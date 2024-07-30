# Databricks notebook source
# MAGIC %md
# MAGIC ##### Define variables and utility functions

# COMMAND ----------

catalog = "main"
schema = "care_cost"
sbc_folder = "sbc"
cpt_folder = "cpt"

# COMMAND ----------

current_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
project_root_path = current_path.split("/")[:-1]

# COMMAND ----------

user_email = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
user_name = user_email.split('@')[0].replace('.','_')
user_prefix = f"{user_name[0:4]}{str(len(user_name)).rjust(3, '0')}"

# COMMAND ----------

print(f"Using catalog: {catalog}")
print(f"Using schema: {schema}")
print(f"Project root: {project_root_path}")

