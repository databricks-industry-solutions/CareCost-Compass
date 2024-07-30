# Databricks notebook source
# MAGIC %md
# MAGIC #### Create catalog, schema and volumes

# COMMAND ----------

# MAGIC %run ./init

# COMMAND ----------

spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.{sbc_folder}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.{cpt_folder}")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Copy Files to Volume
# MAGIC

# COMMAND ----------

#Let us first copy the SBC files 
sbc_folder_path = f"/Volumes/{catalog}/{schema}/{sbc_folder}"
sbc_files = ["SBC_client1.pdf","SBC_client2.pdf"]
for sbc_file in sbc_files:
  dbutils.fs.cp(f"file:/Workspace/{'/'.join(project_root_path)}/resources/{sbc_file}",sbc_folder_path,True)

# COMMAND ----------

#Now lets copy the cpt codes file
cpt_file = "cpt_codes.txt"
cpt_folder_path = f"/Volumes/{catalog}/{schema}/{cpt_folder}"
dbutils.fs.cp(f"file:/Workspace/{'/'.join(project_root_path)}/resources/{cpt_file}",cpt_folder_path,True)
