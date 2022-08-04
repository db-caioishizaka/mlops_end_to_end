# Databricks notebook source
# MAGIC %md
# MAGIC ![Big picture of MLOps demo Step 3 Manage model](files/Step3.png)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Choosing Best Run for the Experiment
# MAGIC 
# MAGIC We get all runs in the experiment, and grab the one with best target measure. In this caspe, MAPE, which is a good metric for forecast models

# COMMAND ----------

from mlflow.tracking.client import MlflowClient
from mlflow.entities import ViewType
import mlflow

experiment_name = "/Users/caio.ishizaka@databricks.com/MLOps - Demand forecast experiments"
experiment = MlflowClient().get_experiment_by_name(experiment_name)
experiment_ids = eval('[' + experiment.experiment_id + ']')
print("Experiments ID:", experiment_ids)

# COMMAND ----------

query = "metrics.val_mape > 0.01"
runs = MlflowClient().search_runs(experiment_ids, query, ViewType.ALL)

# COMMAND ----------

best_mape = None
run_id = None

for run in runs:
  if (best_mape == None or run.data.metrics['val_mape'] < best_mape):
    best_mape = run.data.metrics['val_mape']
    run_id = run.info.run_id
    best_run = run
print("Best Mape:", best_mape)
print("Run ID:", run_id)

model_uri = "runs:/" + run_id + "/model"

print(model_uri)

# COMMAND ----------

import time

##Check if the model is already registered
client = MlflowClient()
model_name = "forecast-globo"
if len(client.search_model_versions(f"name = '{model_name}'"))== 0:
  registered_model = client.get_registered_model(model_name)
  print('new')
else:
  registered_model = mlflow.register_model(model_uri=model_uri, name=model_name)
  print('update')

##Create the model source
model_source = f"{best_run.info.artifact_uri}/model"

print(model_source)

max_version = 0
for mv in client.search_model_versions(f"name = '{model_name}'"):
  current_version = int(dict(mv)['version'])
  if current_version > max_version:
    max_version = current_version
  if dict(mv)['current_stage'] == 'Production':
    version = dict(mv)['version']
    client.transition_model_version_stage(model_name, version, stage = 'Archived')
    
## Promote the model version to production stage
client.transition_model_version_stage(model_name, max_version, stage = 'Production')
