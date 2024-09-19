# Databricks notebook source
# MAGIC %md
# MAGIC #### Utility methods

# COMMAND ----------

# MAGIC %run ./utils/utils

# COMMAND ----------

# MAGIC %md
# MAGIC ####Create a Vector Search endpoint

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

vector_search_endpoint_name = "care_cost_vs_endpoint" 
embedding_endpoint_name = "databricks-bge-large-en" 

sbc_source_data_table = f"{catalog}.{schema}.{sbc_details_table_name}"
sbc_source_data_table_id_field = "id"  
sbc_source_data_table_text_field = "content" 
sbc_vector_index_name = f"{sbc_source_data_table}_index"

cpt_source_data_table = f"{catalog}.{schema}.{cpt_code_table_name}"
cpt_source_data_table_id_field = "id"  
cpt_source_data_table_text_field = "description" 
cpt_vector_index_name = f"{cpt_source_data_table}_index"


# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
vsc = VectorSearchClient(disable_notice=True)

if not endpoint_exists(vsc, vector_search_endpoint_name):
    vsc.create_endpoint(name=vector_search_endpoint_name, endpoint_type="STANDARD")

wait_for_vs_endpoint_to_be_ready(vsc, vector_search_endpoint_name)
print(f"Endpoint named {vector_search_endpoint_name} is ready.")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Check if embeddings endpoint exists
# MAGIC
# MAGIC We will use the existing `databricks-bge-large-en` endpoint for embeddings

# COMMAND ----------

import mlflow
import mlflow.deployments

client = mlflow.deployments.get_deploy_client("databricks")

# COMMAND ----------

[ep for ep in client.list_endpoints() if ep["name"]==embedding_endpoint_name]

# COMMAND ----------

client.predict(endpoint="databricks-bge-large-en", inputs={"input": ["What is Apache Spark?"]})

# COMMAND ----------

# MAGIC %md
# MAGIC ####Create Vector Search Index

# COMMAND ----------

# MAGIC %md
# MAGIC In order for vector search to automatically sync updates, we need to enable ChangeDataFeed on the source table.

# COMMAND ----------

spark.sql(f"ALTER TABLE {sbc_source_data_table} SET TBLPROPERTIES (delta.enableChangeDataFeed = true) ")
spark.sql(f"ALTER TABLE {cpt_source_data_table} SET TBLPROPERTIES (delta.enableChangeDataFeed = true) ")

# COMMAND ----------

sbc_index = create_delta_sync_vector_search_index(
  vector_search_endpoint_name=vector_search_endpoint_name,
  index_name=sbc_vector_index_name,
  source_table_name=sbc_source_data_table,
  primary_key_column=sbc_source_data_table_id_field,
  embedding_source_column=sbc_source_data_table_text_field,
  embedding_endpoint_name=embedding_endpoint_name,
  update_mode="TRIGGERED"
)

# COMMAND ----------

cpt_index = create_delta_sync_vector_search_index(
  vector_search_endpoint_name=vector_search_endpoint_name,
  index_name=cpt_vector_index_name,
  source_table_name=cpt_source_data_table,
  primary_key_column=cpt_source_data_table_id_field,
  embedding_source_column=cpt_source_data_table_text_field,
  embedding_endpoint_name=embedding_endpoint_name,
  update_mode="TRIGGERED"
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Quick Test of Indexes

# COMMAND ----------

results = sbc_index.similarity_search(
  query_text="I need to do a shoulder MRI. How much will it cost me?",
  filters={"client":"sugarshack"},
  columns=["id", "content"],
  num_results=1
)

if results["result"]["row_count"] >0:
  display(results["result"]["data_array"])
else:
  print("No records")

# COMMAND ----------

results = cpt_index.similarity_search(
  query_text="How much does Xray of shoulder cost?",
  columns=["id", "code", "description"],
  num_results=3
)

if results["result"]["row_count"] >0:
  display(results["result"]["data_array"])
else:
  print("No records")
