# Databricks notebook source
# MAGIC %md
# MAGIC ###Mosaic AI Vector Search and Vector Indexes
# MAGIC Mosaic AI Vector Search is a vector database that is built into the Databricks Data Intelligence Platform and integrated with its governance and productivity tools. A vector database is a database that is optimized to store and retrieve embeddings. Embeddings are mathematical representations of the semantic content of data, typically text or image data. [Read More](https://docs.databricks.com/en/generative-ai/vector-search.html)
# MAGIC
# MAGIC ### Vector Indexes
# MAGIC Let us start creating vector indexes
# MAGIC
# MAGIC <img src="./resources/build_3.png" alt="Vector Indexes" width="900"/>
# MAGIC
# MAGIC We will be creating two vector indexes for this project.
# MAGIC 1. Vector Index for the parsed Summary of Benefits and Coverage chunks
# MAGIC 2. Vector Index for CPT codes and descriptions
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC #####Install libraries and import utility methods

# COMMAND ----------

# MAGIC %run ./utils/utils

# COMMAND ----------

# MAGIC %md
# MAGIC ####Create a Vector Search endpoint

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

#name for the vector search endpoint
vector_search_endpoint_name = "care_cost_vs_endpoint" 

#We are using an embedding endpoint available in Databricks Workspace
#If needed we can use custom embedding endpoints as well
embedding_endpoint_name = "databricks-bge-large-en" 

#Define the source tables, index name and key fields
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

#create the vector search endpoint if it does not exist
#same endpoint can be used to serve both the indexes
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

# MAGIC %md
# MAGIC #### Test the embeddings endpoint

# COMMAND ----------

client.predict(endpoint="databricks-bge-large-en", inputs={"input": ["What is Apache Spark?"]})

# COMMAND ----------

# MAGIC %md
# MAGIC ####Create Vector Search Index
# MAGIC
# MAGIC To learn more about creating Vector Indexes, visit this [link](https://docs.databricks.com/en/generative-ai/create-query-vector-search.html). 
# MAGIC
# MAGIC We will now create the vector indexes. Our vector index will be of `Delta Sync Index` type. [[Read More](https://docs.databricks.com/en/generative-ai/create-query-vector-search.html#create-a-vector-search-index)] 
# MAGIC
# MAGIC We will use a Sync Mode of `TRIGGERED` as our table updates are not happening frequently and sync latency is not an issue for us. [[Read More](https://docs.databricks.com/en/generative-ai/create-query-vector-search.html#create-a-vector-search-index:~:text=embedding%20table.-,Sync%20mode%3A,-Continuous%20keeps%20the)]

# COMMAND ----------

# MAGIC %md
# MAGIC **NOTE:** In order for vector search to automatically sync updates, we need to enable ChangeDataFeed on the source table.

# COMMAND ----------

spark.sql(f"ALTER TABLE {sbc_source_data_table} SET TBLPROPERTIES (delta.enableChangeDataFeed = true) ")
spark.sql(f"ALTER TABLE {cpt_source_data_table} SET TBLPROPERTIES (delta.enableChangeDataFeed = true) ")

# COMMAND ----------

# MAGIC %md
# MAGIC #####Create SBC Vector Index

# COMMAND ----------

# MAGIC %md
# MAGIC ######NOTE
# MAGIC Below section creates a vector search index and does an initial sync. Some time this could take longer and the cell execution might timeout. You can re-run the cell to finish to completion

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

# MAGIC %md
# MAGIC #####Create CPT Code Vector Index

# COMMAND ----------

# MAGIC %md
# MAGIC ######NOTE
# MAGIC Below section creates a vector search index and does an initial sync. Some time this could take longer and the cell execution might timeout. You can re-run the cell to finish to completion

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
