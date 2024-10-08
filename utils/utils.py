# Databricks notebook source
# MAGIC %run ./init

# COMMAND ----------

import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import OnlineTableSpec,OnlineTableSpecTriggeredSchedulingPolicy
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
from databricks.feature_store.entities.feature_serving_endpoint import EndpointCoreConfig, ServedEntity
from databricks.sdk.service.serving import EndpointStateReady, EndpointStateConfigUpdate


# COMMAND ----------

def create_online_table(fq_table_name : str, primary_key_columns : [str]):
    
    online_table_name = f"{fq_table_name}_online"
    workspace = WorkspaceClient()
    spec = OnlineTableSpec(
        primary_key_columns = primary_key_columns,
        source_table_full_name = fq_table_name,
        run_triggered=OnlineTableSpecTriggeredSchedulingPolicy.from_dict({'triggered': 'true'}), 
        perform_full_copy=True
    )

    try:
        online_table_pipeline = workspace.online_tables.create(name=online_table_name, spec=spec)
        print(f"Online table {online_table_name} created. Please wait for data sync")  
    except Exception as e:
        if "already exists" in str(e):
            print(f"Online table {online_table_name} already exists. Not recreating.")  
        else:
            raise e
        
def create_feature_serving(fq_table_name : str, primary_key_columns : [str]):
    fe = FeatureEngineeringClient()
    
    catalog_name , schema_name, table_name = fq_table_name.split(".")
    online_table_name = f"{fq_table_name}_online"
    feature_spec_name = f"{catalog_name}.{schema_name}.{table_name}_spec"    
    endpoint_name = f"{table_name}_endpoint".replace('_','-')

    try:
        fe.create_feature_spec(
            name= feature_spec_name,
            features=[
                FeatureLookup(
                    table_name=fq_table_name,
                    lookup_key=primary_key_columns
                )]
        )
        print(f"Feature spec {feature_spec_name} created.")  
    except Exception as e:
        if "already exists" in str(e):
            print(f"Feature spec {feature_spec_name} already exists. Not recreating.")  
        else:
            raise e

    try:
        fe.create_feature_serving_endpoint(
        name=endpoint_name,
            config=EndpointCoreConfig(
                served_entities=ServedEntity(
                    feature_spec_name=feature_spec_name,
                    workload_size="Small",
                    scale_to_zero_enabled=True
                )
            )
        )
        print(f"Endpoint {endpoint_name} created. Please wait for endpoint to start")  
    except Exception as e:
        if "already exists" in str(e):
            print(f"Endpoint {endpoint_name} already exists. Not recreating.")  
        else:
            raise e

# COMMAND ----------

import requests
import json

def get_data_from_online_table(fq_table_name, query_object):
    catalog_name , schema_name, table_name = fq_table_name.split(".")
    online_table_name = f"{fq_table_name}_online"
    endpoint_name = f"{table_name}_endpoint".replace('_','-')

    request_url = f"https://{db_host_name}/serving-endpoints/{endpoint_name}/invocations"
    request_headers = {"Authorization": f"Bearer {db_token}", "Content-Type": "application/json"}
    request_data = {
        "dataframe_records": [query_object]
    }
    response = requests.request(method='POST', headers=request_headers, url=request_url, data=json.dumps(request_data, allow_nan=True))
    
    if response.status_code != 200:
        raise Exception(f"Request failed with status {response.status_code}, {response.text}")
    else:
        return response.json()

# COMMAND ----------

import mlflow 

def start_mlflow_experiment(experiment_name):
    user_email = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
    db_host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
    db_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

    #Create an MLflow experiment
    experiment_base_path = f"Users/{user_email}/mlflow_experiments"
    dbutils.fs.mkdirs(f"file:/Workspace/{experiment_base_path}")
    experiment_path = f"/{experiment_base_path}/{experiment_name}"

    # Manually create the experiment so that you can get the ID and can send it to the worker nodes for scaling
    experiment = mlflow.set_experiment(experiment_path)
    return experiment

# COMMAND ----------

from mlflow.tracking.client import MlflowClient

def get_latest_model_version(model_name):
    mlflow_client = MlflowClient(registry_uri="databricks-uc")
    latest_version = 1
    for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
        version_int = int(mv.version)
        if version_int > latest_version:
            latest_version = version_int
    return latest_version


# COMMAND ----------

import mlflow
import mlflow.deployments
from langchain.chat_models import ChatDatabricks
from langchain.llms import Databricks
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.combine_documents import create_stuff_documents_chain

def build_api_chain(model_endpoint_name, prompt_template, qa_chain=False, max_tokens=500, temperature=0.01):
    client = mlflow.deployments.get_deploy_client("databricks")
    endpoint_details = [ep for ep in client.list_endpoints() if ep["name"]==model_endpoint_name]
    if len(endpoint_details)>0:
      endpoint_detail = endpoint_details[0]
      endpoint_type = endpoint_detail["task"]

      if endpoint_type.endswith("chat"):
        llm_model = ChatDatabricks(endpoint=model_endpoint_name, max_tokens = max_tokens, temperature=temperature)
        llm_prompt = ChatPromptTemplate.from_template(prompt_template)

      elif endpoint_type.endswith("completions"):
        llm_model = Databricks(endpoint_name=model_endpoint_name, 
                               model_kwargs={"max_tokens": max_tokens,
                                             "temperature":temperature})
        llm_prompt = PromptTemplate.from_template(prompt_template)
      else:
        raise Exception(f"Endpoint {model_endpoint_name} not compatible ")

      if qa_chain:
        return create_stuff_documents_chain(llm=llm_model, prompt=llm_prompt)
      else:
        return LLMChain(
          llm = llm_model,
          prompt = llm_prompt
        )
      
    else:
      raise Exception(f"Endpoint {model_endpoint_name} not available ")
  


# COMMAND ----------


