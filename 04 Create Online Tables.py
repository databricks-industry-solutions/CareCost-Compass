# Databricks notebook source
# MAGIC %md
# MAGIC ###Online Tables and Feature Serving
# MAGIC An online table is a read-only copy of a Delta Table that is stored in row-oriented format optimized for online access. Online tables are fully serverless tables that auto-scale throughput capacity with the request load and provide low latency and high throughput access to data of any scale. Online tables are designed to work with Mosaic AI Model Serving, Feature Serving, and agentic applications where they are used for fast data lookups.
# MAGIC
# MAGIC We will be needing online tables for our `member_enrolment`, `member_accumulatord` and `procedure_cost` table.

# COMMAND ----------

# MAGIC %run ./utils/utils

# COMMAND ----------

# MAGIC %md
# MAGIC ###### Online table for `member_enrolment`

# COMMAND ----------

create_online_table(f"{catalog}.{schema}.{member_table_name}", ["member_id"])
create_feature_serving(f"{catalog}.{schema}.{member_table_name}", ["member_id"])

# COMMAND ----------

# MAGIC %md
# MAGIC ###### Online table for  `procedure_cost`

# COMMAND ----------

create_online_table(f"{catalog}.{schema}.{procedure_cost_table_name}", ["procedure_code"])
create_feature_serving(f"{catalog}.{schema}.{procedure_cost_table_name}",["procedure_code"])

# COMMAND ----------

# MAGIC %md
# MAGIC ###### Online table for `member_accumulators` 

# COMMAND ----------

create_online_table(f"{catalog}.{schema}.{member_accumulators_table_name}", ["member_id"])
create_feature_serving(f"{catalog}.{schema}.{member_accumulators_table_name}", ["member_id"])

# COMMAND ----------

# MAGIC %md
# MAGIC ######Query the endpoints to test

# COMMAND ----------

get_data_from_online_table(f"{catalog}.{schema}.{member_table_name}", {"member_id":"1234"})

# COMMAND ----------

get_data_from_online_table(f"{catalog}.{schema}.{procedure_cost_table_name}", {"procedure_code":"23920"})

# COMMAND ----------

get_data_from_online_table(f"{catalog}.{schema}.{member_accumulators_table_name}", {"member_id":"1234"})

# COMMAND ----------

