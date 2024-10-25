# Databricks notebook source
# MAGIC %md
# MAGIC # Parsing and Chunking Summary of benefits
# MAGIC ######Next step of our building process is to build the the Summary of Benefit Parsing and implement an appropriate chunking strategy.
# MAGIC <img src="./resources/build_2.png" alt="Parse and Chunk" width="900"/>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ###Chunking Strategy
# MAGIC
# MAGIC In this notebook we will try to ingest the Coverage Summary documents in PDF format. This example makes the assumption that the coverage summary document is in the below format.
# MAGIC
# MAGIC First page is the summary of coverage as shown below
# MAGIC
# MAGIC <img src="./resources/img_summary.png" alt="drawing" width="700"/>
# MAGIC
# MAGIC Remaining pages has the details of coverage as shown below
# MAGIC
# MAGIC <img src="./resources/img_details.png" alt="drawing" width="700"/>
# MAGIC
# MAGIC Our aim is to extract this tabular data from PDF and create full text summary of each line item so that it captures the details appropriately. Below is an example
# MAGIC
# MAGIC For the line item
# MAGIC <img src="./resources/img_line.png" alt="drawing" width="700"/> we want to generate two paragraphs as below 
# MAGIC
# MAGIC **If you have a test, for Diagnostic test (x-ray, blood work) you will pay $10 copay/test In Network and 40% coinsurance Out of Network.**
# MAGIC
# MAGIC and 
# MAGIC
# MAGIC **If you have a test, for Imaging (CT/PET scans, MRIs) you will pay $50 copay/test In Network and 40% coinsurance Out of Network.**
# MAGIC
# MAGIC We have to create more pipelines and parsing logic for different kind of summary formats
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Read PDF documents

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Setting up `camelot`
# MAGIC [Camelot](https://camelot-py.readthedocs.io/en/master/) is one of the Python librareis that can help extract tabular data from PDFs. Camelot provides:
# MAGIC
# MAGIC **Configurability**: Camelot gives you control over the table extraction process with tweakable settings.
# MAGIC
# MAGIC **Metrics**: You can discard bad tables based on metrics like accuracy and whitespace, without having to manually look at each table.
# MAGIC
# MAGIC **Output**: Each table is extracted into a pandas DataFrame, which seamlessly integrates into ETL and data analysis workflows. You can also export tables to multiple formats, which include CSV, JSON, Excel, HTML, Markdown, and Sqlite.
# MAGIC
# MAGIC **NOTE:** Camelot only works with text-based PDFs and not scanned documents. For processing scanned PDF documents, we might have to change the PDF reading library
# MAGIC
# MAGIC **NOTE:** The `lattice` mode of Camelot relies on having a PDF rendering backend. Here we will create a custom backend using `pdfplumber` library. If you wish to install other backends like `GhostScript` or `Poppler`. Please follow the [instructions](https://camelot-py.readthedocs.io/en/master/user/install-deps.html) and install the appropriate backend. 
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC #####Install other libraries and import utility methods

# COMMAND ----------

# MAGIC %run ./utils/init

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Implement pdf reading using `camelot`
# MAGIC
# MAGIC Let us create some utility methods to read each data sections from the Summary of Benefits and Coverage (SBC) document. 
# MAGIC
# MAGIC **NOTE**:These methods are kept simple for demonstration, but could be extended to generalize for different SBC formats.

# COMMAND ----------

import pandas as pd
import camelot
import pdfplumber

#lets create a custom backend for camelot to convert pdf to png
class SBCConversionBackend(object):
    def convert(self,pdf_path, png_path):
        pdf = pdfplumber.open(pdf_path)
        pdf.pages[0].to_image(resolution=600).save(png_path)

def get_summary(pdf_name : str) -> pd.DataFrame :
    #assuming first page is summary
    tables = camelot.read_pdf(pdf_name,
                              pages="1",
                              backend=SBCConversionBackend(),
                              flavor="lattice")
    summary_df = tables[0].df
    summary_df = summary_df.tail(-1)
    summary_df.columns= ["Questions","Answer","Why this matters"] 
    return summary_df

def format_coverage_page(page_df:pd.DataFrame,skip_lines:int) -> pd.DataFrame :
    if len(page_df.columns) == 5:
        page_df.columns= ["Medical Event","Service","In Network Amount","Out of Network Amount", "Limitations,Exceptions and Important Information"] 
        page_df=page_df.mask(page_df == '')
        page_df=page_df.fillna(method='ffill')
        return page_df
    else:
        return None

def get_coverage(pdf_name : str, summary_page = True) -> pd.DataFrame :
    
    #There are custom processing that is needed for each PDF
    #In a production usecase, you would have algorithms detect each characteristic and process accordingly
    #clien1 pdf has two header lines we need to skip
    #client2 pdf has an extra example tables at the end we need to skip
    skip_lines = 2 if "client1" in pdf_name else 1
    skip_tables = 8 if "client2" in pdf_name else 2

    tables = camelot.read_pdf(pdf_name,
                              pages="2-end" if summary_page else "1-end",                              
                              backend=SBCConversionBackend(),
                              flavor="lattice")

    page_df_list = [table.df for table in tables]
    
    #we also need to process the excluded services and other covered services differently
    #we are ignoring them for this example
    covered_services = page_df_list[: -skip_tables]

    #format covered services
    page_df = pd.concat([df.tail(-skip_lines) for df in covered_services])
    page_df_formatted = format_coverage_page(page_df, skip_lines)
    
    return page_df_formatted

# COMMAND ----------

# MAGIC %md
# MAGIC ######Quick test of data extraction

# COMMAND ----------

#lets test our methods
"""  Uncomment this with a PDF backend library installed
pdf_name = f"{sbc_folder_path}/SBC_client2.pdf"
display(get_summary(pdf_name))
"""

# COMMAND ----------

"""  Uncomment this with a PDF backend library installed
display(get_coverage(pdf_name))
"""

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Summarizing 
# MAGIC Now we have the data in tabular form, we need to summarize each row into text and create a document. Then we will embed the document

# COMMAND ----------

from pyspark.sql.types import ArrayType, StringType

def clean(text):
    return str(text).replace('\n','')

def summarize_summary_row(row):
    return f" {clean(row['Questions'])}\n Answer is {clean(row['Answer'])}. Why it matters to you is because {clean(row['Why this matters'])}"

def get_extra_coverage_info(row):
    extra = clean(row['Limitations,Exceptions and Important Information'])
    return f"Also {extra}" if "none" not in extra.lower() else ""

def summarize_coverage_row(row):
    return f" {clean(row['Medical Event'])}, for {clean(row['Service'])} you will pay {clean(row['In Network Amount']) } In Network and {clean(row['Out of Network Amount'])} Out of Network. {get_extra_coverage_info(row)}"

def summary_to_document(summary_df):
    return summary_df.apply(summarize_summary_row, axis=1).values.tolist()

def coverage_to_document(coverage_df):
    return coverage_df.apply(summarize_coverage_row, axis=1).values.tolist()

@udf(returnType=ArrayType(StringType()))
def pdf_to_document(pdf_file):
    summary_df = get_summary(pdf_file)
    coverage_df = get_coverage(pdf_file)
    return summary_to_document(summary_df) + coverage_to_document(coverage_df)


# COMMAND ----------

import pandas as pd

doc_list = [f"{sbc_folder_path}/{sbc_files[0]}",f"{sbc_folder_path}/{sbc_files[1]}"]

pd_sbc_details = pd.DataFrame({
        "client" : client_names, 
        "sbc_file_name": doc_list
    })


# COMMAND ----------

display(pd_sbc_details)

# COMMAND ----------

# MAGIC %md
# MAGIC Using Spark to load and chunk the PDF documents for scalability

# COMMAND ----------

from pyspark.sql.functions import explode, array, monotonically_increasing_id

sbc_details = (spark
               .createDataFrame(pd_sbc_details)               
               .withColumn("content",pdf_to_document("sbc_file_name"))
               .select("client", explode("content").alias("content"))
               .repartition(1)
               .withColumn("id", monotonically_increasing_id())
               .select("id","client","content")

)

# COMMAND ----------

"""  Uncomment this with a PDF backend library installed
display(sbc_details)
"""

# COMMAND ----------

# MAGIC %md
# MAGIC ###Save the SBC data to a Delta table in Unity Catalog

# COMMAND ----------

spark.sql(f"DROP TABLE IF EXISTS {catalog}.{schema}.{sbc_details_table_name}")
sbc_details.write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.{sbc_details_table_name}")

# COMMAND ----------

display(spark.table(f"{catalog}.{schema}.{sbc_details_table_name}"))

# COMMAND ----------


