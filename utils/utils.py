# Databricks notebook source
# MAGIC %run ./init

# COMMAND ----------

import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import OnlineTableSpec,OnlineTableSpecTriggeredSchedulingPolicy
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
from databricks.feature_store.entities.feature_serving_endpoint import EndpointCoreConfig, ServedEntity
from databricks.sdk.service.serving import EndpointStateReady, EndpointStateConfigUpdate

def endpoint_exists(vsc, vs_endpoint_name):
  try:
    return vs_endpoint_name in [e['name'] for e in vsc.list_endpoints().get('endpoints', [])]
  except Exception as e:
    #Temp fix for potential REQUEST_LIMIT_EXCEEDED issue
    if "REQUEST_LIMIT_EXCEEDED" in str(e):
      print("WARN: couldn't get endpoint status due to REQUEST_LIMIT_EXCEEDED error. The demo will consider it exists")
      return True
    else:
      raise e

def wait_for_vs_endpoint_to_be_ready(vsc, vs_endpoint_name):
  for i in range(180):
    try:
      endpoint = vsc.get_endpoint(vs_endpoint_name)
    except Exception as e:
      #Temp fix for potential REQUEST_LIMIT_EXCEEDED issue
      if "REQUEST_LIMIT_EXCEEDED" in str(e):
        print("WARN: couldn't get endpoint status due to REQUEST_LIMIT_EXCEEDED error. Please manually check your endpoint status")
        return
      else:
        raise e
    status = endpoint.get("endpoint_status", endpoint.get("status"))["state"].upper()
    if "ONLINE" in status:
      return endpoint
    elif "PROVISIONING" in status or i <6:
      if i % 20 == 0: 
        print(f"Waiting for endpoint to be ready, this can take a few min... {endpoint}")
      time.sleep(10)
    else:
      raise Exception(f'''Error with the endpoint {vs_endpoint_name}. - this shouldn't happen: {endpoint}.\n Please delete it and re-run the previous cell: vsc.delete_endpoint("{vs_endpoint_name}")''')
  raise Exception(f"Timeout, your endpoint isn't ready yet: {vsc.get_endpoint(vs_endpoint_name)}")

def index_exists(vsc, endpoint_name, index_full_name):
    try:
        vsc.get_index(endpoint_name, index_full_name).describe()
        return True
    except Exception as e:
        if 'RESOURCE_DOES_NOT_EXIST' not in str(e):
            print(f'Unexpected error describing the index. This could be a permission issue.')
            raise e
    return False
    
def wait_for_index_to_be_ready(vsc, vs_endpoint_name, index_name):
  for i in range(180):
    idx = vsc.get_index(vs_endpoint_name, index_name).describe()
    index_status = idx.get('status', idx.get('index_status', {}))
    status = index_status.get('detailed_state', index_status.get('status', 'UNKNOWN')).upper()
    url = index_status.get('index_url', index_status.get('url', 'UNKNOWN'))
    if "ONLINE" in status:
      return
    if "UNKNOWN" in status:
      print(f"Can't get the status - will assume index is ready {idx} - url: {url}")
      return
    elif "PROVISIONING" in status:
      if i % 40 == 0: print(f"Waiting for index to be ready, this can take a few min... {index_status} - pipeline url:{url}")
      time.sleep(10)
    else:
        raise Exception(f'''Error with the index - this shouldn't happen. DLT pipeline might have been killed.\n Please delete it and re-run the previous cell: vsc.delete_index("{index_name}, {vs_endpoint_name}") \nIndex details: {idx}''')
  raise Exception(f"Timeout, your index isn't ready yet: {vsc.get_index(index_name, vs_endpoint_name)}")

def wait_for_model_serving_endpoint_to_be_ready(ep_name):
    # Wait for it to be ready
    w = WorkspaceClient()
    state = ""
    for i in range(200):
        state = w.serving_endpoints.get(ep_name).state
        if state.config_update == EndpointStateConfigUpdate.IN_PROGRESS:
            if i % 40 == 0:
                print(f"Waiting for endpoint to deploy {ep_name}. Current state: {state}")
            time.sleep(10)
        elif state.ready == EndpointStateReady.READY:
          print('endpoint ready.')
          return
        else:
          break
    raise Exception(f"Couldn't start the endpoint, timeout, please check your endpoint for more details: {state}")

def create_delta_sync_vector_search_index(vector_search_endpoint_name, 
                 index_name, 
                 source_table_name, 
                 primary_key_column, 
                 embedding_source_column, 
                 embedding_endpoint_name,
                 update_mode):
    
    vsc = VectorSearchClient(disable_notice=True)
    
    index_ready = False
    index_exists = False
    try:
        index_info = vsc.get_index(endpoint_name=vector_search_endpoint_name,
                                   index_name=index_name)
        print(f"Index {index_name} already exists")
        index_exists = True
    except:
        print(f"Creating Index {index_name} ")

    if not index_exists:
        index = vsc.create_delta_sync_index(
            endpoint_name=vector_search_endpoint_name,
            source_table_name=source_table_name,
            index_name=index_name,
            pipeline_type=update_mode,
            primary_key=primary_key_column,
            embedding_source_column=embedding_source_column,
            embedding_model_endpoint_name=embedding_endpoint_name
        )

    wait_for_index_to_be_ready(vsc, vector_search_endpoint_name, index_name)

    return vsc.get_index(endpoint_name=vector_search_endpoint_name,index_name=index_name)

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

    #Create an MLFlow experiment
    experiment_base_path = f"Users/{user_email}/mlflow_experiments"
    dbutils.fs.mkdirs(f"file:/Workspace/{experiment_base_path}")
    experiment_path = f"/{experiment_base_path}/{experiment_name}"

    # Manually create the experiment so that you can get the ID and can send it to the worker nodes for scaling
    experiment = mlflow.set_experiment(experiment_path)
    return experiment

# COMMAND ----------

from mlflow.tracking.client import MlflowClient

def get_latest_model_version(model_name: str, env_or_alias: str=""):  
    mlflow.set_registry_uri("databricks-uc")
    client = MlflowClient()

    if env_or_alias == "":
        models = client.search_model_versions(f"name='{model_name}'")
        if len(models) >0:
            return models[0]
        else:
            return None
    else:
        try:
            return client.get_model_version_by_alias(name=model_name,alias=env_or_alias)
        except:
            return None

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


