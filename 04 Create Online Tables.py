# Databricks notebook source
# MAGIC %run ./utils/utils

# COMMAND ----------

# MAGIC %md
# MAGIC ###### Online table for `member`

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


