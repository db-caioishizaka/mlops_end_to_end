# Databricks notebook source
# MAGIC %md ###Training Prophet

# COMMAND ----------

import mlflow
import databricks.automl_runtime

target_col = "sales"
time_col = "date"
unit = "day"

id_cols = ["store", "item"]

horizon = 90

# COMMAND ----------

import pyspark.pandas as ps
df_loaded = ps.DataFrame(spark.table('demand_history'))

# Preview data
df_loaded.head(5)

# COMMAND ----------

import logging

# disable informational messages from prophet
logging.getLogger("py4j").setLevel(logging.WARNING)

# COMMAND ----------

from pyspark.sql.types import *
import pandas as pd

result_columns = ["ts_id", "model_json", "prophet_params", "start_time", "mse",
                  "rmse", "mae", "mape", "mdape", "smape", "coverage"]
result_schema = StructType([
  StructField("ts_id", StringType()),
  StructField("model_json", StringType()),
  StructField("prophet_params", StringType()),
  StructField("start_time", TimestampType()),
  StructField("mse", FloatType()),
  StructField("rmse", FloatType()),
  StructField("mae", FloatType()),
  StructField("mape", FloatType()),
  StructField("mdape", FloatType()),
  StructField("smape", FloatType()),
  StructField("coverage", FloatType())
  ])

def prophet_training(history_pd):
  from hyperopt import hp
  from databricks.automl_runtime.forecast.prophet.forecast import ProphetHyperoptEstimator

  seasonality_mode = ["additive", "multiplicative"]
  search_space =  {
    "changepoint_prior_scale": hp.loguniform("changepoint_prior_scale", -6.9, -0.69),
    "seasonality_prior_scale": hp.loguniform("seasonality_prior_scale", -6.9, 2.3),
    "holidays_prior_scale": hp.loguniform("holidays_prior_scale", -6.9, 2.3),
    "seasonality_mode": hp.choice("seasonality_mode", seasonality_mode)
  }
  country_holidays = None
  run_parallel = False
 
  hyperopt_estim = ProphetHyperoptEstimator(horizon=horizon, frequency_unit=unit, metric="smape",interval_width=0.8,
                   country_holidays=country_holidays, search_space=search_space, num_folds=3, max_eval=2, trial_timeout=7110,
                   random_state=327032298, is_parallel=run_parallel)

  results_pd = hyperopt_estim.fit(history_pd)
  results_pd["ts_id"] = str(history_pd["ts_id"].iloc[0])
  results_pd["start_time"] = pd.Timestamp(history_pd["ds"].min())
 
  return results_pd[result_columns]

def train_with_fail_safe(df):
  try:
    return prophet_training(df)
  except Exception as e:
    print(f"Encountered an exception while training timeseries: {repr(e)}")
    return pd.DataFrame(columns=result_columns)

# COMMAND ----------

# MAGIC %md ###Load Feature Store Table

# COMMAND ----------

from databricks import feature_store

fs = feature_store.FeatureStoreClient()

# COMMAND ----------

from databricks.feature_store import FeatureLookup

# Start an mlflow run, which is needed for the feature store to log the model
#mlflow.start_run() 

feature_lookups = [
    FeatureLookup(
      table_name = 'feature_store_forecast_example.demand_history',
      feature_names = ['ds','store','item','y','ts_id'],
      lookup_key = 'customer_id')
  ]

# Create the training set that includes the raw input data merged with corresponding features from both feature tables
training_set = fs.create_training_set(
  df = training_df,
  feature_lookups = feature_lookups,
  label = '',
)

# Load the TrainingSet into a dataframe which can be passed into sklearn for training a model
#training_df = training_set.load_df()

# End any existing runs (in the case this notebook is being run for a second time)
#mlflow.end_run()

# COMMAND ----------

training_df = fs.read_table(
  name='feature_store_forecast_example.demand_history',
)

# COMMAND ----------

import mlflow
from databricks.automl_runtime.forecast.prophet.model import mlflow_prophet_log_model, MultiSeriesProphetModel

mlflow.set_experiment("/Users/caio.ishizaka@databricks.com/MLOps - Demand forecast experiments")

with mlflow.start_run(run_name="PROPHET") as mlflow_run:
  run_id = mlflow_run.info.run_id
  mlflow.set_tag("estimator_name", "Prophet")
  mlflow.log_param("interval_width", 0.8)
  mlflow.log_param("num_folds", 3)
  mlflow.log_param("max_eval", 2)
  mlflow.log_param("trial_timeout", 7110)
  
  forecast_results = (training_df.repartition(sc.defaultParallelism, id_cols)
    .groupby(id_cols).applyInPandas(train_with_fail_safe, result_schema)).cache().pandas_api()
   
  # Check whether every time series's model is trained
  ts_models_trained = set(forecast_results["ts_id"].unique().to_list())
  ts_ids = set(forecast_results["ts_id"].unique().tolist())

  if len(ts_models_trained) == 0:
    raise Exception("Trial unable to train models for any identities. Please check the training cell for error details")

  if ts_ids != ts_models_trained:
    mlflow.log_param("partial_model", True)
    print(f"WARNING: Models not trained for the following identities: {ts_ids.difference(ts_models_trained)}")
 
  # Log the metrics to mlflow
  avg_metrics = forecast_results[["mse", "rmse", "mae", "mape", "mdape", "smape", "coverage"]].mean().to_frame(name="mean_metrics").reset_index()
  avg_metrics["index"] = "val_" + avg_metrics["index"].astype(str)
  avg_metrics.set_index("index", inplace=True)
  mlflow.log_metrics(avg_metrics.to_dict()["mean_metrics"])
  #testing logging params
#   mlflow.log_param("prophet_params", forecast_results[["ts_id", "prophet_params"]].to_pandas().set_index("ts_id").to_dict()["prophet_params"])

  # Create mlflow prophet model
  model_json = forecast_results[["ts_id", "model_json"]].to_pandas().set_index("ts_id").to_dict()["model_json"]
  start_time = forecast_results[["ts_id", "start_time"]].to_pandas().set_index("ts_id").to_dict()["start_time"]
  prophet_model = MultiSeriesProphetModel(model_json, start_time, "2017-12-31 00:00:00", horizon, unit, time_col, id_cols)

  # Generate sample input dataframe
  sample_input = df_loaded.head(1).to_pandas()
  sample_input[time_col] = pd.to_datetime(sample_input[time_col])
  sample_input.drop(columns=[target_col], inplace=True)

  mlflow_prophet_log_model(prophet_model, sample_input=sample_input)

# COMMAND ----------

forecast_results.head(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Analyze the predicted results

# COMMAND ----------

from mlflow.tracking.client import MlflowClient
from mlflow.entities import ViewType

experiment_name = "/Users/caio.ishizaka@databricks.com/MLOps - Demand forecast experiments"
experiment = MlflowClient().get_experiment_by_name(experiment_name)
experiment_ids = eval('[' + experiment.experiment_id + ']')
print("Experiments ID:", experiment_ids)

# COMMAND ----------

query = "metrics.val_coverage < 0.99"
runs = MlflowClient().search_runs(experiment_ids, query, ViewType.ALL)

# COMMAND ----------

accuracy_high = None
run_id = None

for run in runs:
  if (accuracy_high == None or run.data.metrics['val_coverage'] > accuracy_high):
    accuracy_high = run.data.metrics['val_coverage']
    run_id = run.info.run_id
    best_run = run
print("Highest Accuracy:", accuracy_high)
print("Run ID:", run_id)

model_uri = "runs:/" + run_id + "/model"

print(model_uri)

# COMMAND ----------

# Load the model

loaded_model = mlflow.pyfunc.load_model(model_uri)

# COMMAND ----------

import pyspark.pandas as ps

model = loaded_model._model_impl.python_model
col_types = [StructField(f"{n}", FloatType()) for n in model.get_reserved_cols()]
col_types.append(StructField("ds",TimestampType()))
col_types.append(StructField("ts_id",StringType()))
result_schema = StructType(col_types)

ids = ps.DataFrame(model._model_json.keys(), columns=["ts_id"])
forecast_pd = ids.to_spark().groupby("ts_id").applyInPandas(lambda df: model.model_predict(df), result_schema).cache().pandas_api().set_index("ts_id")

# COMMAND ----------

# Plotly plots is turned off by default because it takes up a lot of storage.
# Set this flag to True and re-run the notebook to see the interactive plots with plotly
use_plotly = False

# COMMAND ----------

# Choose a random id from `ts_id` for plot
id = set(forecast_pd.index.to_list()).pop()
# Get the prophet model for this id
model = loaded_model._model_impl.python_model.model(id)
predict_pd = forecast_pd.loc[id].to_pandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Plot the forecast with change points and trend

# COMMAND ----------

from prophet.plot import add_changepoints_to_plot, plot_plotly
import datetime

if use_plotly:
    fig = plot_plotly(model, predict_pd, changepoints=True, trend=True, figsize=(1200, 600))
else:
    fig = model.plot(predict_pd)
    a = add_changepoints_to_plot(fig.gca(), model, predict_pd)
fig

fig.savefig('forecast.png')

# COMMAND ----------

current_run = run.info.run_id
with mlflow.start_run(run_id=current_run, nested=True):  
  mlflow.log_artifact("forecast.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Plot the forecast components

# COMMAND ----------

from prophet.plot import plot_components_plotly
if use_plotly:
    fig = plot_components_plotly(model, predict_pd, figsize=(900, 400))
    fig.show()
else:
    fig = model.plot_components(predict_pd)
    
fig.savefig('forecast_components.png')

current_run = run.info.run_id
with mlflow.start_run(run_id=current_run, nested=True):  
  mlflow.log_artifact("forecast_components.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Show the predicted results

# COMMAND ----------

predict_cols = ["ds", "ts_id", "yhat","yhat_lower", "yhat_upper"]
forecast_pd = forecast_pd.reset_index()
display(forecast_pd[predict_cols].tail(horizon))
