# Databricks notebook source
# MAGIC %run ./utils/init

# COMMAND ----------

# MAGIC %md
# MAGIC ##### First lets collect the endpoint details to connect to the app

# COMMAND ----------

from databricks.sdk import WorkspaceClient


w = WorkspaceClient()

all_endpoints = w.serving_endpoints.list()
care_cost_endpoint = [e for e in all_endpoints if "carecost" in e.name][0]


# COMMAND ----------

print(f"Endpoint Name: {care_cost_endpoint.name}")


# COMMAND ----------

yaml_string = f"""
command: [
  "streamlit", 
  "run",
  "app.py"
]

env:
  - name: STREAMLIT_BROWSER_GATHER_USAGE_STATS
    value: "false"
  - name: "SERVING_ENDPOINT"
    value: "{care_cost_endpoint.name}"
"""


with open("app/app.yaml", "w") as file:
    file.write(yaml_string)


# COMMAND ----------

from utils.apps_helper import LakehouseAppHelper

app_name = f"carecost-compass-chatbot"

helper = LakehouseAppHelper()
app_details = helper.create(app_name, app_description="A Chatbot for providing Precedure Cost Estimations")

# COMMAND ----------

# MAGIC %md
# MAGIC ### NOTE: Wait until the app is in STARTED state

# COMMAND ----------

import yaml
from pathlib import Path

app_config = yaml.safe_load(Path('app/app.yaml').read_text())
endpoint_name = [ env["value"] for env in app_config["env"] if env["name"]=="SERVING_ENDPOINT"][0]

# COMMAND ----------

#please wait until the app is created
helper.add_dependencies(
    app_name=app_name,
    dependencies=[
        {
            "name": "llm-endpoint",
            "serving_endpoint": {
                "name": endpoint_name,
                "permission": "CAN_QUERY",
            },
        }
    ],
    overwrite=False # if False dependencies will be appended to existing ones
)

# COMMAND ----------

import os
helper.deploy(app_name, os.path.join(os.getcwd(), 'app'))
displayHTML(helper.details(app_name))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Almost there! Last Step
# MAGIC ##### Add `CAN QUERY` permission to the App Service Principal on the `carecost_compass` enpoint

# COMMAND ----------


