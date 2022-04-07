# Databricks notebook source
import mlflow

# COMMAND ----------

dbutils.widgets.text('run_dt','2022-04-06')

# COMMAND ----------

# MAGIC %md
# MAGIC #### Pull Clients with Term-Life Contract Expiring in a Year

# COMMAND ----------

q = f"""
with expiring_ctrcs as (
  select *
  from contracts_management.contract
  where add_months(issue_dt, 12*term_length) = '{dbutils.widgets.get('run_dt')}'
  ),
  
prev_appts as (
  select a.*,
    b.appt_dt as prev_appt_dt,
    row_number() over (partition by a.cust_id_nbr order by b.appt_dt desc) as prev_rn
  from expiring_ctrcs a
  left join contracts_management.sf_appointments b
  on a.emp_id = b.emp_id
  and b.appt_dt < '{dbutils.widgets.get('run_dt')}'
  ),
  
next_appts as (
  select a.*,
    b.appt_dt as next_appt_dt,
    row_number() over (partition by a.cust_id_nbr order by b.appt_dt asc) as next_rn
  from (select * from prev_appts where prev_rn = 1) a
  left join contracts_management.sf_appointments b
  on a.emp_id = b.emp_id
  and b.appt_dt >= '{dbutils.widgets.get('run_dt')}'
)

select ben_ctrc_nbr,
  term_length,
  issue_dt,
  cust_id_nbr,
  emp_id,
  emp_email_id,
  prev_appt_dt,
  next_appt_dt
from next_appts
where next_rn = 1"""

df = spark.sql(q)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Next Best Product

# COMMAND ----------

columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

product_rec_data = (
  spark.read.format('csv').load('/FileStore/product_rec_data.csv', header=True)
  .select(columns+['cust_id_nbr'])
)

loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri="models:/ri-product-rec-test/Production", result_type='double')

df = (
  df
  .join(product_rec_data, on='cust_id_nbr', how='left')
  .withColumn('predictions', loaded_model(*columns))
  .cache()
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Generate HTML Emails for All Clients and Save to S3

# COMMAND ----------

products = ['Whole Life','Variable Annuity', 'Long-term Care']

# COMMAND ----------

recipient = df.first().emp_email_id
recipient_name = recipient.split('@')[0].split('.')[0].title()
cust_id_nbr = df.first().cust_id_nbr
ben_ctrc_nbr = df.first().ben_ctrc_nbr
customer_name = "John"
most_likely_next_product = products[int(df.first().predictions)]
total_contracts_owned = 3
total_aum = 1000000
last_sf_meeting = df.first().prev_appt_dt
scheduled_sf_meeting = df.first().next_appt_dt

# COMMAND ----------

with open('/dbfs/FileStore/email_template.html', 'r') as f:
  html = f.read()
  html = (html
    .replace('MYINSERT_recipient_name', recipient_name)
    .replace('MYINSERT_customer_name', customer_name)
    .replace('MYINSERT_cust_id_nbr', str(cust_id_nbr))
    .replace('MYINSERT_ben_ctrc_nbr', ben_ctrc_nbr)
    .replace('MYINSERT_most_likely_next_product', most_likely_next_product)
    .replace('MYINSERT_total_aum', f"{total_aum:,}")
    .replace('MYINSERT_total_contracts_owned', str(total_contracts_owned))
    .replace('MYINSERT_last_sf_meeting', last_sf_meeting.strftime("%m/%d/%Y"))
    .replace('MYINSERT_scheduled_sf_meeting', scheduled_sf_meeting.strftime("%m/%d/%Y"))
  )

# COMMAND ----------

# displayHTML(html)
