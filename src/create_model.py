# Databricks notebook source
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import mlflow
from sklearn import tree

# COMMAND ----------

iris = load_iris()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Save Dataset for Testing

# COMMAND ----------

feturte_names = [f.replace(' (cm)','').replace(' ', '_') for f in iris['feature_names']]

# COMMAND ----------

df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= feturte_names + ['target'])

df['target_names'] = df['target'].apply(lambda x: list(iris.target_names)[int(x)])

# COMMAND ----------

df.to_csv('/dbfs/FileStore/product_rec_data.csv', index=False)

# COMMAND ----------

# with open('/dbfs/FileStore/product_rec_data.csv') as f:
#   s = f.read()
#   print(s)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Create and Log Dummy Model

# COMMAND ----------

with mlflow.start_run(): 
  
  X, y = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].to_numpy(), df.target
  clf = tree.DecisionTreeClassifier()
  clf = clf.fit(X, y)
  
  mlflow.sklearn.log_model(clf, 'ri-product-rec-test', registered_model_name='ri-product-rec-test')