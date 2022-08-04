# Databricks notebook source
# MAGIC %md
# MAGIC ![Big picture of MLOps demo Step 1 Load data](files/Step1.png)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Data
# MAGIC 
# MAGIC We are reading straight from a delta table. We could be reading from any source, including files, BigQuery, jdbc connection, Kafka, and many others.

# COMMAND ----------

import pyspark.pandas as ps
df_loaded = ps.DataFrame(spark.table('demand_history'))

# Preview data
df_loaded.head(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Initial trend analysis
# MAGIC 
# MAGIC Let's start taking a look at yearly, monthly and weekly trends

# COMMAND ----------

# DBTITLE 1,Yearly trends
# MAGIC %sql
# MAGIC 
# MAGIC SELECT
# MAGIC   year(date) as year, 
# MAGIC   sum(sales) as sales
# MAGIC FROM demand_history
# MAGIC GROUP BY year(date)
# MAGIC ORDER BY year;

# COMMAND ----------

# DBTITLE 1,Monthly trends
# MAGIC %sql
# MAGIC 
# MAGIC SELECT 
# MAGIC   TRUNC(date, 'MM') as month,
# MAGIC   SUM(sales) as sales
# MAGIC FROM demand_history
# MAGIC GROUP BY TRUNC(date, 'MM')
# MAGIC ORDER BY month;

# COMMAND ----------

# DBTITLE 1,Weekday trends
# MAGIC %sql
# MAGIC 
# MAGIC SELECT
# MAGIC   YEAR(date) as year,
# MAGIC   extract(dayofweek from date) as weekday,
# MAGIC   AVG(sales) as sales
# MAGIC FROM (
# MAGIC   SELECT 
# MAGIC     date,
# MAGIC     SUM(sales) as sales
# MAGIC   FROM demand_history
# MAGIC   GROUP BY date
# MAGIC  ) x
# MAGIC GROUP BY year, weekday
# MAGIC ORDER BY year, weekday;

# COMMAND ----------

import mlflow
import databricks.automl_runtime

target_col = "sales"
time_col = "date"
unit = "day"

id_cols = ["store", "item"]

horizon = 90

# COMMAND ----------

# MAGIC %md
# MAGIC ### Aggregate data by `store`, `item` and `date`
# MAGIC Group the data by `store`, `item` and `date`, and take average if there are multiple `sales` values in the same group.

# COMMAND ----------

group_cols = [time_col] + id_cols
df_aggregated = df_loaded \
  .groupby(group_cols) \
  .agg(y=(target_col, "avg")) \
  .reset_index() \
  .rename(columns={ time_col : "ds" })

df_aggregated = df_aggregated.assign(ts_id=lambda x:x["store"].astype(str)+"-"+x["item"].astype(str))

df_aggregated.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Prepare Feature Store
# MAGIC The Feature store is the best way to prepare and expose data to be trained or be used for model inference.

# COMMAND ----------

# MAGIC %md
# MAGIC First, create the database where the feature tables will be stored.

# COMMAND ----------

# MAGIC %sql 
# MAGIC CREATE DATABASE IF NOT EXISTS feature_store_forecast_example;

# COMMAND ----------

# MAGIC %md
# MAGIC Next, create an instance of the Feature Store client.

# COMMAND ----------

from databricks import feature_store

fs = feature_store.FeatureStoreClient()

# COMMAND ----------

# MAGIC %md
# MAGIC Use either the create_table API (Databricks Runtime 10.2 ML or above) or the create_feature_table API (Databricks Runtime 10.1 ML or below) to define schema and unique ID keys. If the optional argument df (Databricks Runtime 10.2 ML or above) or features_df (Databricks Runtime 10.1 ML or below) is passed, the API also writes the data to Feature Store.

# COMMAND ----------

# MAGIC %md Create Table for the first time
# MAGIC 
# MAGIC     fs.create_table(
# MAGIC       name="feature_store_forecast_example.demand_history",
# MAGIC       primary_keys=["ds", "store","item"],
# MAGIC       df=df_aggregated.to_spark(),
# MAGIC       description="Feature of store sales history",
# MAGIC   )
# MAGIC     

# COMMAND ----------

#fs.drop_table(name="feature_store_forecast_example.demand_history")

# COMMAND ----------

# MAGIC %md When writing, both `merge` and `overwrite` modes are supported.
# MAGIC 
# MAGIC     fs.write_table(
# MAGIC       name="feature_store_forecast_example.demand_history",
# MAGIC       df=df_aggregated,
# MAGIC       mode="overwrite",
# MAGIC     )
# MAGIC     
# MAGIC Data can also be streamed into Feature Store by passing a dataframe where `df.isStreaming` is set to `True`:
# MAGIC 
# MAGIC     fs.write_table(
# MAGIC       name="feature_store_forecast_example.demand_history",
# MAGIC       df=streaming_df,
# MAGIC       mode="merge",
# MAGIC     )
# MAGIC     
# MAGIC You can schedule a notebook to periodically update features using Databricks Jobs ([AWS](https://docs.databricks.com/jobs.html)|[Azure](https://docs.microsoft.com/azure/databricks/jobs)|[GCP](https://docs.gcp.databricks.com/jobs.html)).

# COMMAND ----------

fs.write_table(
      name="feature_store_forecast_example.demand_history",
      df=df_aggregated.to_spark(),
      mode="overwrite",
  )

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC SELECT * FROM feature_store_forecast_example.demand_history LIMIT 10
