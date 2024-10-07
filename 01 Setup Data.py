# Databricks notebook source
# MAGIC %md
# MAGIC #Prepare Data
# MAGIC ###### Let us start by creating some synthetic data to work with.
# MAGIC <img src="./resources/build_1.png" alt="Prepare Data" width="900" style="border:2px;"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC This notebook will create CATALOG and SCHEMA if it does not exist and create the below data tables.
# MAGIC
# MAGIC **member_enrolment**: Table containing member enrolment information like client and plan_id
# MAGIC
# MAGIC **member_accumulators**: Table containing member accumulators like deductibles and out of pocket spent
# MAGIC
# MAGIC **cpt_codes**: Table containing CPT codes and descriptions
# MAGIC
# MAGIC **procedure_cost**: Table containing negotiated cost of each procedure. 
# MAGIC
# MAGIC In addition to these tables, this notebook creates a Unity Catalog Volume and store the Summary of Benefit PDF files and CPT Code CSV files in appropriate folders
# MAGIC
# MAGIC We are using synthetic data as example. In reality robust Data Ingestion Pipelines will be used to manage this data in a Lakehouse.
# MAGIC
# MAGIC #####Read More:
# MAGIC * [Databricks Volumes](https://docs.databricks.com/en/sql/language-manual/sql-ref-volumes.html)
# MAGIC * [Ingest Data into Databricks Lakehouse](https://docs.databricks.com/en/ingestion/index.html)
# MAGIC * [Data Pipelines in Databricks](https://docs.databricks.com/en/getting-started/data-pipeline-get-started.html)
# MAGIC

# COMMAND ----------

# MAGIC %run ./utils/init

# COMMAND ----------

# MAGIC %md
# MAGIC ####Create Catalog and Schema

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

for sbc_file in sbc_files:
  dbutils.fs.cp(f"file:/Workspace/{project_root_path}/resources/{sbc_file}",sbc_folder_path,True)

# COMMAND ----------

#Now lets copy the cpt codes file
#Downloaded from https://www.cdc.gov/nhsn/xls/cpt-pcm-nhsn.xlsx

dbutils.fs.cp(f"file:/Workspace/{project_root_path}/resources/{cpt_file}",cpt_folder_path,True)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Create Data Tables
# MAGIC - Member Enrolment Table: Contains member details including the client id
# MAGIC - Member Accumulator Table: Contain member year to date deductible accumulator
# MAGIC - Procedure Cost Table: Contain estimated cost of all the covered procedures
# MAGIC
# MAGIC

# COMMAND ----------

import pandas as pd
from pyspark.sql.types import StructType, StructField, StringType, DateType, DoubleType, IntegerType, LongType
import datetime

# COMMAND ----------

# MAGIC %md
# MAGIC #####`member_enrolment`

# COMMAND ----------

member_table_schema = StructType([
    StructField("member_id",StringType(), nullable=False),
    StructField("client_id",StringType(), nullable=False),   
    StructField("plan_id",StringType(), nullable=False),
    StructField("plan_start_date",DateType(), nullable=False),
    StructField("plan_end_date",DateType(), nullable=False),
    StructField("active_ind",StringType(), nullable=False),    
])

member_data = [
    ("1234",client_names[0],"P1", datetime.date(2024,1,1), datetime.date(2024,12,31),"Y" ),
    ("2345",client_names[0],"P1", datetime.date(2024,1,1), datetime.date(2024,12,31),"Y" ),
    ("7890",client_names[1],"P2", datetime.date(2024,1,1), datetime.date(2024,12,31),"Y" ),
]

member = spark.createDataFrame(member_data, schema=member_table_schema)

spark.sql(f"DROP TABLE IF EXISTS {catalog}.{schema}.{member_table_name}")

spark.catalog.createTable(f"{catalog}.{schema}.{member_table_name}", schema=member_table_schema)

member.write.mode("append").saveAsTable(f"{catalog}.{schema}.{member_table_name}")

spark.sql(f"ALTER TABLE {catalog}.{schema}.{member_table_name} ADD CONSTRAINT {member_table_name}_pk PRIMARY KEY( member_id )")

# COMMAND ----------

# MAGIC %md
# MAGIC ###### Inspect and Verify `Data`

# COMMAND ----------

display(spark.table(f"{catalog}.{schema}.{member_table_name}"))

# COMMAND ----------

# MAGIC %md
# MAGIC #####`member_accumulators`

# COMMAND ----------


member_accumulators_schema = StructType([
    StructField("member_id",StringType(), nullable=False),
    StructField("oop_max",DoubleType(), nullable=False),
    StructField("fam_deductible",DoubleType(), nullable=False),
    StructField("mem_deductible",DoubleType(), nullable=False),
    StructField("oop_agg",DoubleType(), nullable=False),
    StructField("mem_ded_agg",DoubleType(), nullable=False),
    StructField("fam_ded_agg",DoubleType(), nullable=False),
])

member_accumulators_data = [
    ('1234', 2500.00, 1500.00, 1000.00, 500.00, 500.00, 750.00),
    ('2345', 2500.00, 1500.00, 1000.00, 250.00, 250.00, 750.00),
    ('7890', 3000.00, 2500.00, 2000.00, 3000.00, 2000.00, 2000.00),
]

member_accumulators = spark.createDataFrame(member_accumulators_data, schema=member_accumulators_schema)

spark.sql(f"DROP TABLE IF EXISTS {catalog}.{schema}.{member_accumulators_table_name}")

spark.catalog.createTable(f"{catalog}.{schema}.{member_accumulators_table_name}", schema=member_accumulators_schema)

member_accumulators.write.mode("append").saveAsTable(f"{catalog}.{schema}.{member_accumulators_table_name}")

spark.sql(f"ALTER TABLE {catalog}.{schema}.{member_accumulators_table_name} ADD CONSTRAINT {member_accumulators_table_name}_pk PRIMARY KEY( member_id)")

# COMMAND ----------

# MAGIC %md
# MAGIC ###### Inspect and Verify Data

# COMMAND ----------

display(spark.table(f"{catalog}.{schema}.{member_accumulators_table_name}"))

# COMMAND ----------

# MAGIC %md
# MAGIC #####`cpt_codes`
# MAGIC

# COMMAND ----------

from pyspark.sql.functions import monotonically_increasing_id

cpt_codes_file = f"{cpt_folder_path}/{cpt_file}"

cpt_codes_file_schema = (StructType()
    .add("code",StringType(),True)
    .add("description",StringType(),True)
)

cpt_codes_table_schema = (StructType()
    .add("id",LongType(),False)
    .add("code",StringType(),True)
    .add("description",StringType(),True)
)


cpt_df = (spark
          .read
          .option("header", "false")
          .option("delimiter", "\t")
          .schema(cpt_codes_file_schema)
          .csv(cpt_codes_file)
          .repartition(1)
          .withColumn("id",monotonically_increasing_id())
          .select("id","code","description")
)

spark.sql(f"DROP TABLE IF EXISTS {catalog}.{schema}.{cpt_code_table_name}")

spark.catalog.createTable(f"{catalog}.{schema}.{cpt_code_table_name}", schema=cpt_codes_table_schema)

cpt_df.write.mode("append").saveAsTable(f"{catalog}.{schema}.{cpt_code_table_name}")

spark.sql(f"ALTER TABLE {catalog}.{schema}.{cpt_code_table_name} ADD CONSTRAINT {cpt_code_table_name}_pk PRIMARY KEY( id )")

# COMMAND ----------

# MAGIC %md
# MAGIC ###### Inspect and Verify Data

# COMMAND ----------

display(cpt_df)

# COMMAND ----------

# MAGIC %md
# MAGIC #####`procedure_cost` 
# MAGIC Table containing negotiated cost of each procedure.
# MAGIC For simiplicity we will assign a random cost to each procedure

# COMMAND ----------

from pyspark.sql.functions import rand,round, pow, ceil,col

procedure_cost_schema = StructType([
    StructField("procedure_code",StringType(), nullable=False),
    StructField("cost",DoubleType(), nullable=False)
])

spark.sql(f"DROP TABLE IF EXISTS {catalog}.{schema}.{procedure_cost_table_name}")

spark.catalog.createTable(f"{catalog}.{schema}.{procedure_cost_table_name}", schema=procedure_cost_schema)

#Read the procedure codes and assign some cost to it
#In a production scenario it could be a complex procedure to calculate the expected cost
procedure_cost = (
    spark
    .table(f"{catalog}.{schema}.{cpt_code_table_name}")
    .withColumn("pow", ceil(rand(seed=1234) * 10) % 3 + 2 )
    .withColumn("cost", round(rand(seed=2345) *  pow(10, "pow") + 20 ,2)  )
    .select(col("code").alias("procedure_code"),"cost")
)

procedure_cost.write.mode("append").saveAsTable(f"{catalog}.{schema}.{procedure_cost_table_name}")

spark.sql(f"ALTER TABLE {catalog}.{schema}.{procedure_cost_table_name} ADD CONSTRAINT {procedure_cost_table_name}_pk PRIMARY KEY( procedure_code )")


# COMMAND ----------

# MAGIC %md
# MAGIC ###### Inspect and Verify Data

# COMMAND ----------

display(spark.table(f"{catalog}.{schema}.{procedure_cost_table_name}"))
