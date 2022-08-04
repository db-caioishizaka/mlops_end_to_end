# Databricks notebook source
# MAGIC %md
# MAGIC ## Analyze the predicted results

# COMMAND ----------

horizon = (90)

# COMMAND ----------

import mlflow
from mlflow.tracking.client import MlflowClient

client = MlflowClient()
model_name = "forecast-globo"

latest_version_info = client.get_latest_versions(model_name, stages=["Production"])
latest_production_version = latest_version_info[0].version
print("The latest production version of the model '%s' is '%s'." % (model_name, latest_production_version))

# COMMAND ----------

model_production_uri = "models:/{model_name}/production".format(model_name=model_name)
print(model_production_uri)

# COMMAND ----------

# Load the model

loaded_model = mlflow.pyfunc.load_model(model_production_uri)

# COMMAND ----------

import pyspark.pandas as ps
from pyspark.sql.types import *

model = loaded_model._model_impl.python_model
col_types = [StructField(f"{n}", FloatType()) for n in model.get_reserved_cols()]
col_types.append(StructField("ds",TimestampType()))
col_types.append(StructField("ts_id",StringType()))
result_schema = StructType(col_types)

ids = ps.DataFrame(model._model_json.keys(), columns=["ts_id"])
forecast_pd = ids.to_spark().groupby("ts_id").applyInPandas(lambda df: model.model_predict(df), result_schema).cache().pandas_api().set_index("ts_id")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Show the predicted results

# COMMAND ----------

predict_cols = ["ds", "ts_id", "yhat","yhat_lower", "yhat_upper"]
#forecast_pd = forecast_pd.reset_index()
display(forecast_pd[predict_cols])
