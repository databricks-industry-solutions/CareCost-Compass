# Databricks notebook source
# MAGIC %md
# MAGIC ##### Define variables and utility functions

# COMMAND ----------

import pytz
from datetime import datetime
#TODO:
####CHANGE ME

#Timezone that you want to use for mlflow logging
timezone_for_logging = "US/Eastern"
logging_timezone = pytz.timezone(timezone_for_logging)

#catalog to use for creating data tables and keeping other resources
#You need necessary privileges to create, delete tables, functions, and models
catalog = "main"
#schema to use for creating data tables and keeping other resources
#You need necessary privileges to create, delete tables, functions, and model
schema = "care_cost"

#The Volume folder where SBC files will be copied to
sbc_folder = "sbc"
#The Volume folder where CPT files will be copied to
cpt_folder = "cpt"

#List of sbc files
sbc_files = ["SBC_client1.pdf","SBC_client2.pdf"]
#imaginary client names for each sbc file
client_names = ["sugarshack","chillystreet"]
#Imaginary Payor name
payor_name = "LemonDrop"

#cpt code file name
cpt_file = "cpt_codes.txt"
#Data table names
member_table_name = "member_enrolment"
member_accumulators_table_name = "member_accumulators"
cpt_code_table_name = "cpt_codes"
procedure_cost_table_name = "procedure_cost"
sbc_details_table_name = "sbc_details"

#Mlflow experiment tag
experiment_tag = f"carecost_compass_agent"

sbc_folder_path = f"/Volumes/{catalog}/{schema}/{sbc_folder}"
cpt_folder_path = f"/Volumes/{catalog}/{schema}/{cpt_folder}"

# COMMAND ----------

current_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
project_root_path = current_path.split("/")[:-1]

# COMMAND ----------

db_host_name = spark.conf.get('spark.databricks.workspaceUrl')
db_host_url = f"https://{db_host_name}"
db_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)

# COMMAND ----------

user_email = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
user_name = user_email.split('@')[0].replace('.','_')
user_prefix = f"{user_name[0:4]}{str(len(user_name)).rjust(3, '0')}"

# COMMAND ----------

#Create mlflow experiment
from datetime import datetime
import mlflow 

mlflow.set_registry_uri("databricks-uc")
mlflow_experiment_base_path = f"Users/{user_email}/mlflow_experiments"

def set_mlflow_experiment(experiment_tag):
    dbutils.fs.mkdirs(f"file:/Workspace/{mlflow_experiment_base_path}")
    experiment_path = f"/{mlflow_experiment_base_path}/{experiment_tag}"
    return mlflow.set_experiment(experiment_path)


# COMMAND ----------

print(f"Using catalog: {catalog}")
print(f"Using schema: {schema}")
print(f"Project root: {project_root_path}")
print(f"MLFlow Experiment Path: {mlflow_experiment_base_path}")
