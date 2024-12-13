# Databricks notebook source
# MAGIC %md
# MAGIC #Create All Tools
# MAGIC Tools are interfaces that an agent, chain, or LLM can use to interact with the world. [Read More](https://python.langchain.com/v0.1/docs/modules/tools/)
# MAGIC
# MAGIC ###Identify Tools
# MAGIC
# MAGIC While constructing Agents, we might be leveraging many functions to perform specific actions. In our application we have the below functions that we need to implement
# MAGIC * Retrieve `member_id` from context
# MAGIC * Classifier to categorize the question
# MAGIC * A lookup function to get `client_id` from `member_id` from member enrolment table
# MAGIC * A RAG module to lookup Benefit from Summary of Benefits index for the `client_id`
# MAGIC * A semantic search module to lookup appropriate procedure code for the question
# MAGIC * A lookup function to get procedure cost for the retrieved `procedure_code` from procedure cost table
# MAGIC * A lookup function to get member accumulators for the `member_id` from member accumulators table
# MAGIC * A pythom function to calculate out of pocket cost given the information from the previous steps
# MAGIC * A summarizer to summarize the calculation in a professional manner ans send it to the user
# MAGIC
# MAGIC While develpoing Agentic Applications, thse functions will be developed as **Tools** so that the Agent Executor can use them to process the user request. 
# MAGIC
# MAGIC In this notebook we will develop thse functions as LangChain [tools](https://python.langchain.com/v0.1/docs/modules/tools/), so that we can potentially use these tools in a LangChain agent.
# MAGIC
# MAGIC **NOTE:** In a real enterprise application many of these tools could be complex functions or REST api calls to other services. The scope of this notebook is to illustrate the feature and can be extended any way possible.
# MAGIC
# MAGIC <img src="https://raw.githubusercontent.com/databricks-industry-solutions/CareCost-Compass/refs/heads/main/resources/build_5.png" alt="Create Tools" width="900" />

# COMMAND ----------

# MAGIC %md
# MAGIC ###MLflow Tracing for Agents
# MAGIC Using MLflow Tracing you can log, analyze, and compare traces across different versions of generative AI applications. It allows you to debug your generative AI Python code and keep track of inputs and responses. Doing so can help you discover conditions or parameters that contribute to poor performance of your application. MLflow Tracing is tightly integrated with Databricks tools and infrastructure, allowing you to store and display all your traces in Databricks notebooks or the MLflow experiment UI as you run your code. [Read More](https://docs.databricks.com/en/mlflow/mlflow-tracing.html)
# MAGIC
# MAGIC We will be using MLflow Tracing throughout this project. Look for the `@mlflow.trace` decorator on some of the methods
# MAGIC

# COMMAND ----------

# MAGIC %run ./utils/utils

# COMMAND ----------

# MAGIC %md
# MAGIC ## Adding building blocks for the Gen AI Application
# MAGIC
# MAGIC All the building blocks will be developed as Tools

# COMMAND ----------

import mlflow
import mlflow.deployments
import os
import pandas as pd
import requests
import json

from typing import Optional, Type, List, Union

from pydantic import BaseModel, Field

from langchain.tools import BaseTool, StructuredTool, tool
from langchain.callbacks.manager import (AsyncCallbackManagerForToolRun, CallbackManagerForToolRun)
from langchain.chat_models import ChatDatabricks
from langchain.llms import Databricks
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_core.documents.base import Document
from langchain.output_parsers import PydanticOutputParser


from databricks.vector_search.client import VectorSearchClient
from databricks.vector_search.index import VectorSearchIndex



# COMMAND ----------

# MAGIC %md
# MAGIC #### Let us create some helper classes and utility methods

# COMMAND ----------

class BaseCareCostToolBuilder:
    name:str = None
    description:str = None
    args_schema : Type[BaseModel] = None
    def execute(self, **kwargs):
        raise NotImplementedError("Please Implement this method")
        
    def get(self):
        return StructuredTool.from_function(func=self.execute,
                                            name=self.name,
                                            description=self.description,
                                            args_schema=self.args_schema)

class RetrieverConfig(BaseModel):
    """A data class for passing around vector index configuration"""
    vector_search_endpoint_name:str
    vector_index_name:str
    vector_index_id_column:str
    retrieve_columns:List[str]

def build_api_chain(model_endpoint_name, prompt_template, qa_chain=False, max_tokens=500, temperature=0.01):
    client = mlflow.deployments.get_deploy_client("databricks")
    endpoint_details = [ep for ep in client.list_endpoints() if ep["name"]==model_endpoint_name]
    if len(endpoint_details)>0:
      endpoint_detail = endpoint_details[0]
      endpoint_type = endpoint_detail["task"]

      if endpoint_type.endswith("chat"):
        llm_model = ChatDatabricks(endpoint=model_endpoint_name, max_tokens = max_tokens, temperature=temperature)
        llm_prompt = ChatPromptTemplate.from_template(prompt_template)

      elif endpoint_type.endswith("completions"):
        llm_model = Databricks(endpoint_name=model_endpoint_name, 
                               model_kwargs={"max_tokens": max_tokens,
                                             "temperature":temperature})
        llm_prompt = PromptTemplate.from_template(prompt_template)
      else:
        raise Exception(f"Endpoint {model_endpoint_name} not compatible ")

      if qa_chain:
        return create_stuff_documents_chain(llm=llm_model, prompt=llm_prompt)
      else:
        return LLMChain(
          llm = llm_model,
          prompt = llm_prompt
        )
      
    else:
      raise Exception(f"Endpoint {model_endpoint_name} not available ")


def get_data_from_online_table(fq_table_name, query_object):
    catalog_name , schema_name, table_name = fq_table_name.split(".")
    online_table_name = f"{fq_table_name}_online"
    endpoint_name = f"{table_name}_endpoint".replace('_','-')


    client = mlflow.deployments.get_deploy_client("databricks")
    response = client.predict(
      endpoint = endpoint_name,
      inputs = {
        "dataframe_records": [query_object]
      }
    )
    return response

# COMMAND ----------

# MAGIC %md
# MAGIC ###Add Member Id Retriever
# MAGIC
# MAGIC The function of `MemberIdRetriever` is to retrieve the value of `member_id` from the chat messages input to the model.
# MAGIC
# MAGIC We can easily implement this using a simple Zero Shot prompt
# MAGIC

# COMMAND ----------

class MemberIdRetrieverInput(BaseModel):
    """Data class for tool input"""
    question: str = Field(description="Sentence containing member_id")

class  MemberIdRetriever(BaseCareCostToolBuilder):
    """A tool to extract member id from question"""
    name : str = "MemberIdRetriever"
    description : str = "useful for extracting member id from question"
    args_schema : Type[BaseModel] = MemberIdRetrieverInput
    model_endpoint_name:str = None

    prompt:str = "Extract the member id from the question. \
        Only respond with a single word which is the member id. \
        Do not include any other  details in response.\
        Question:{question}"

    def __init__(self, model_endpoint_name : str):
        super().__init__()
        self.model_endpoint_name = model_endpoint_name
    
    @mlflow.trace(name="get_member_id", span_type="func")
    def execute(self, question:str) -> str: 
        chain = build_api_chain(self.model_endpoint_name, self.prompt)
        category = chain.run(question=question)
        return category.strip()
    

# COMMAND ----------

# MAGIC %md
# MAGIC ###Adding Question Classifier
# MAGIC
# MAGIC The `QuestionClassifier` acts as a content filter that checks if the question from user is appropriate and relevant. We will create the content filter component as a Natural Language Classfier using simple Zero Shot prompt.
# MAGIC

# COMMAND ----------



class QuestionClassifierInput(BaseModel):
    """Data class for tool input"""
    questions: List[str] = Field(description="Question to be classified")

class QuestionClassifier(BaseCareCostToolBuilder):
    """A tool to classify questions into categories"""
    name : str = "QuestionClassifier"
    description : str = "useful for classifying questions into categories"
    args_schema : Type[BaseModel] = QuestionClassifierInput
    model_endpoint_name:str = None
    categories_and_description:dict = None
    category_str: str = ""

    prompt:str = "Classify the question into one of below the categories. \
        {categories}\
        Only respond with a single word which is the category code. \
        Do not include any other  details in response.\
        Question:{question}"

    def __init__(self, model_endpoint_name : str, categories_and_description : dict[str:str]):
        super().__init__()
        self.model_endpoint_name = model_endpoint_name
        self.categories_and_description = categories_and_description
        self.category_str = "\n".join([ f"{c}:{self.categories_and_description[c]}" for c in self.categories_and_description])
    
    @mlflow.trace(name="get_question_category", span_type="func")
    def execute(self, questions:[str]) -> [str]: 
        chain = build_api_chain(self.model_endpoint_name, self.prompt)
        categories = []
        for question in questions:
            category = chain.run(categories=self.category_str, question=question)
            categories.append(category.strip())
        
        return categories

    

# COMMAND ----------

# MAGIC %md
# MAGIC ###Adding Benefit RAG
# MAGIC
# MAGIC The function of `BenefitsRetriever` is to 
# MAGIC * Retrieve the appropriate Benefit clause from Summary of Benefits Index 
# MAGIC * Parse the clause, retrieve in-network and out-of-network benefits and convert into a structured dataclass object
# MAGIC
# MAGIC We will implement this as a RAG application with Zero Shot prompting

# COMMAND ----------

class BenefitsRetriever():    
    """A retriever class to do Vector Index Search"""
    retriever_config: RetrieverConfig = None
    vector_index: VectorSearchIndex = None

    def __init__(self, retriever_config: RetrieverConfig):
        super().__init__()
        self.retriever_config = retriever_config
        
        vsc = VectorSearchClient()
        
        self.vector_index = vsc.get_index(endpoint_name=self.retriever_config.vector_search_endpoint_name,
                                          index_name=self.retriever_config.vector_index_name)

    @mlflow.trace(name="get_benefit_retriever", span_type="func")
    def get_benefits(self, client_id:str, question:str):
        query_results = self.vector_index.similarity_search(
            query_text=question,
            filters={"client":client_id},
            columns=self.retriever_config.retrieve_columns,
            num_results=1)
        
        return query_results


class BenefitsRAGInput(BaseModel):
    """Data class for tool input"""
    client_id : str = Field(description="Client ID for which the benefits need to be retrieved")
    question: str = Field(description="Question for which the benefits need to be retrieved")

class Benefit(BaseModel):
    """Data class for tool output"""
    text:str = Field(description="Full text as provided in the context as-is without changing anything")
    in_network_copay:float = Field(description="In Network copay amount. Set to -1 if not covered or has coinsurance")
    in_network_coinsurance:float= Field(description="In Network coinsurance amount without the % sign. Set to -1 if not covered or has copay")
    out_network_copay:float = Field(description="Out of Network copay amount. Set to -1 if not covered or has coinsurance")
    out_network_coinsurance:float = Field(description="Out of Network coinsurance amount without the % sign. Set to -1 if not covered or has copay")
    
class BenefitsRAG(BaseCareCostToolBuilder):
    """Tool class implementing the benefits retriever"""
    name : str = "BenefitsRAG"
    description : str = "useful for retrieving benefits from a vector search index in json format"
    args_schema : Type[BaseModel] = BenefitsRAGInput
    model_endpoint_name:str = None
    retriever_config: RetrieverConfig = None    
    retrieved_documents:List[Document] = None
    prompt_coverage_qa:str = "Get the member medical coverage benefits from the input sentence at the end:\
        The output should only contain the formatted JSON instance that conforms to the JSON schema below.\
        Do not provide any extra information other than the json object.\
        {pydantic_parser_format_instruction}\
        Input Sentence:{context}"
    

    def __init__(self,
                 model_endpoint_name : str,
                 retriever_config: RetrieverConfig):
        super().__init__()
        self.model_endpoint_name = model_endpoint_name
        self.retriever_config = retriever_config
        
    @mlflow.trace(name="get_benefits", span_type="func")
    def execute(self, client_id:str, question:str) -> str:

        retriever = BenefitsRetriever(self.retriever_config)        
        self.retrieved_documents = None
        query_results = retriever.get_benefits(client_id, question)
        
        if query_results["result"]["row_count"] > 0:
            coverage_records = [Document(page_content=data[1]) for data in query_results["result"]["data_array"]]
            #save the records for evaluation
            self.retrieved_documents = coverage_records

            qa_chain = build_api_chain(model_endpoint_name=self.model_endpoint_name,
                                       prompt_template=self.prompt_coverage_qa,
                                       qa_chain=True)
            parser = PydanticOutputParser(pydantic_object=Benefit)

            answer = qa_chain.invoke({"context": coverage_records,
                               "pydantic_parser_format_instruction": parser.get_format_instructions()})
            return answer.replace('`','')# Benefit.model_validate_json(answer)
        else:
            raise Exception("No coverage found")


# COMMAND ----------

# MAGIC %md
# MAGIC ###Adding Procedure Retriever
# MAGIC
# MAGIC `ProcedureRetriever` tool will be used to retrieve procedure code that corresponds to the question. It's implemented as simple seamntic search using vector search api

# COMMAND ----------

class ProcedureRetrieverInput(BaseModel):
    """Data class for tool input"""
    question: str = Field(description="Question for which the procedure need to be retrieved")

class ProcedureRetriever(BaseCareCostToolBuilder):
    """A retriever class to do Vector Index Search"""
    name : str = "ProcedureRetriever"
    description : str = "useful for retrieving an appropriate procedure code for the given question"
    args_schema : Type[BaseModel] = ProcedureRetrieverInput

    retriever_config: RetrieverConfig = None
    vector_index: VectorSearchIndex = None

    def __init__(self, retriever_config: RetrieverConfig):
        super().__init__()
        self.retriever_config = retriever_config
        
        vsc = VectorSearchClient()
        
        self.vector_index = vsc.get_index(endpoint_name=self.retriever_config.vector_search_endpoint_name,
                                          index_name=self.retriever_config.vector_index_name)

    @mlflow.trace(name="get_procedure_details", span_type="func")
    def execute(self, question:str) -> (str,str):
        query_results = self.vector_index.similarity_search(
            query_text=question,
            columns=self.retriever_config.retrieve_columns,
            num_results=1)

        if query_results["result"]["row_count"] > 0:      
            procedure_detail = query_results["result"]["data_array"][0]
            return (procedure_detail[0],procedure_detail[1])
        else:
            raise Exception("No procedure found.")


# COMMAND ----------

# MAGIC %md
# MAGIC ###Adding Client Id Lookup
# MAGIC
# MAGIC `ClientIdLookup` tool will provide an Online Table look up to retrieve `client_id` for a given `member_id`. We will implment this using a feature lookup on the `member_enrolment` Online Table

# COMMAND ----------

class ClientIdLookupInput(BaseModel):
    """Data class for tool input"""
    member_id: str = Field(description="Member ID using which we need to lookup client id")

class ClientIdLookup(BaseCareCostToolBuilder):    
    """A class to do online table lookup to retrieve client_id gievn member_id"""
    name : str = "ClientIdLookup"
    description : str = "useful for retrieving a client id given a member id"
    args_schema : Type[BaseModel] = ClientIdLookupInput
    fq_member_table_name:str = None

    def __init__(self, fq_member_table_name:str):
        super().__init__()
        self.fq_member_table_name = fq_member_table_name
    
    @mlflow.trace(name="get_client_id", span_type="func")
    def execute(self, member_id:str) -> str:
        member_data = get_data_from_online_table(self.fq_member_table_name, 
                                                 {"member_id":member_id})
        print(member_data)
        return member_data["outputs"][0]["client_id"]


# COMMAND ----------

# MAGIC %md
# MAGIC ###Adding Procedure Cost Lookup
# MAGIC
# MAGIC Next we will implement a function to retrieve the negotiated Procedure Cost given a procedure code. 
# MAGIC
# MAGIC This function is a very simplistic implementation of something that can be quite complex and requires properly constructed data engineering pipelines. In this example we have made an assumption that the aggregation of the procedure cost from various providers are already completed and is available in a delta table.

# COMMAND ----------

class ProcedureCostLookupInput(BaseModel):
    """Data class for tool input"""
    procedure_code: str = Field(description="Procedure Code for which to find the cost")

class ProcedureCostLookup(BaseCareCostToolBuilder):    
    """A class to do online table lookup to retrieve procedure cost given procedure code"""
    name : str = "ProcedureCostLookup"
    description : str = "useful for retrieving the cost of a procedure given the procedure code"
    args_schema : Type[BaseModel] = ProcedureCostLookupInput
    fq_procedure_cost_table_name:str = None

    def __init__(self, fq_procedure_cost_table_name:str):
        super().__init__()
        self.fq_procedure_cost_table_name = fq_procedure_cost_table_name
    
    @mlflow.trace(name="get_procedure_cost", span_type="func")
    def execute(self, procedure_code:str) -> float:
        procedure_cost_data = get_data_from_online_table(self.fq_procedure_cost_table_name,
                                                         {"procedure_code":procedure_code})
        return procedure_cost_data["outputs"][0]["cost"]


# COMMAND ----------

# MAGIC %md
# MAGIC ###Adding Member Accumulators Lookup
# MAGIC
# MAGIC Next important function that we need to implement is a lookup to retrieve the member accumulators like deductibles and YTD out of pocket costs. In this example we are implementing it as a simple table lookup with member_id but usually the implementation could be different, for eg: could be a call to a micro service api
# MAGIC

# COMMAND ----------

class MemberAccumulatorsLookupInput(BaseModel):
    """Data class for tool input"""
    member_id: str = Field(description="Member Id for which we need to lookup the accumulators")

class MemberAccumulatorsLookup(BaseCareCostToolBuilder):    
    """A class to do online table lookup to retrieve member accumulators given member id"""
    name : str = "MemberAccumulatorsLookup"
    description : str = "useful for retrieving the accumulators like deductibles given a member id"
    args_schema : Type[BaseModel] = MemberAccumulatorsLookupInput
    fq_member_accumulators_table_name:str = None

    def __init__(self, fq_member_accumulators_table_name:str):
        super().__init__()
        self.fq_member_accumulators_table_name = fq_member_accumulators_table_name
    
    @mlflow.trace(name="get_member_accumulators", span_type="func")
    def execute(self, member_id:str) -> dict[str, Union[float,str] ]:
        accumulator_data = get_data_from_online_table(self.fq_member_accumulators_table_name,
                                                      {"member_id":member_id})
        return accumulator_data["outputs"][0]


# COMMAND ----------

# MAGIC %md
# MAGIC ###Adding Cost Calculator
# MAGIC Cost Calculator is a deterministic method that takes member benefits, deductibles and procedure cost to calculate the out of pocket cost that member could be paying for the procedure
# MAGIC
# MAGIC This is again a very simplistic implementation of a calculation which could be much complex in reality. As mentioned before, the idea is to illustrate patterns that can be easily extended as needed

# COMMAND ----------

class MemberCost(BaseModel):
  """Data class for member cost which will be the model output"""
  in_network_cost : float = Field(description="In-Network cost of the procedure")
  out_network_cost : float = Field(description="Out-Network cost of the procedure")
  notes : List[str] = Field(description="Notes about the cost calculation")


class MemberCostCalculatorInput(BaseModel):
  """Data class for tool input"""
  benefit:Benefit = Field(description="Benefit object for the member")
  procedure_cost:float = Field(description="Cost of the procedure")
  member_deductibles:dict[str, Union[float,str] ] = Field(description="Accumulators for the member")


class MemberCostCalculator(BaseCareCostToolBuilder):
    """A class to calculate the member out of pocket cost given the benefits, procedure cost and deductibles"""
    name : str = "MemberCostCalculator"
    description : str = "calculates the estimated member out of pocket cost given the benefits, procedure cost and deductibles"
    args_schema : Type[BaseModel] = MemberCostCalculatorInput

    def __init__(self):
        super().__init__()

    @mlflow.trace(name="get_member_out_of_pocket_cost", span_type="func")
    def execute(self, 
                  benefit:Benefit,
                  procedure_cost:float,
                  member_deductibles:dict[str, Union[float,str] ]) -> MemberCost:
        """
        Method to get estimated member out of pocket cost
        """
        in_network_cost = benefit.in_network_copay if benefit.in_network_copay > 0 else benefit.in_network_coinsurance
        out_network_cost = benefit.out_network_copay if benefit.out_network_copay > 0 else benefit.out_network_coinsurance
        in_network_cost_type = "copay" if benefit.in_network_copay > 0 else "coinsurance"
        out_network_cost_type = "copay" if benefit.out_network_copay > 0 else "coinsurance"
        notes=[benefit.text]

        #If oop_max has met member has to pay anything
        if member_deductibles["mem_ded_agg"] < member_deductibles["oop_max"]:
          notes.append("Out of pocket maximum is not met.")
          #if annual deductible is met, only pay copay/coinsurance
          if member_deductibles["mem_ded_agg"] >= member_deductibles["mem_deductible"]:
            notes.append("Deductible is met.")
            if in_network_cost > 0:
              notes.append("This procedure is covered In-Network.")

              if in_network_cost_type == "copay":
                in_network_cost = in_network_cost 
                notes.append("You will pay only your copay amount")
              else:
                in_network_cost = (float(procedure_cost)*in_network_cost)/100
                notes.append("You will pay a percentage of procedure cost as coinsurance In-Network")

            else:
              notes.append("This procedure is not covered In-Network. You need to pay the full cost of the procedure if done In-Network")
              in_network_cost = procedure_cost

            if out_network_cost > 0:
              notes.append("This procedure is covered Out-Of-Network.")

              if out_network_cost_type == "copay":
                out_network_cost = out_network_cost 
                notes.append("You will pay only your copay amount")
              else:
                out_network_cost = (float(procedure_cost)*out_network_cost)/100
                notes.append("You will pay a percentage of procedure cost as coinsurance Out-Of-network")

            else:
              notes.append("This procedure is not covered Out-Of-Network. You need to pay the full cost of the procedure if done Out-Of-Network")
              out_network_cost = procedure_cost
            
          else:
            notes.append("Deductible not met. You need to pay the full cost of the procedure")
            in_network_cost = procedure_cost
            out_network_cost = procedure_cost

        notes.append(f"Your cost if procedure is done In-Network is {in_network_cost}")
        notes.append(f"Your cost if procedure is done Out-Of-Network is {out_network_cost}")
        member_cost = MemberCost(in_network_cost=in_network_cost, out_network_cost=out_network_cost, notes=notes)
        return member_cost


# COMMAND ----------

# MAGIC %md
# MAGIC ### Adding Summarizer
# MAGIC
# MAGIC `ResponseSummarizer` will take the calculation notes produced by the cost calculator tool and summarize it as response to the user.
# MAGIC It is implemented using simple Zero Shot prompt

# COMMAND ----------

class ResponseSummarizerInput(BaseModel):
    notes:List[str] = Field(description="MemberCost object for the member")    

class ResponseSummarizer(BaseCareCostToolBuilder):
    name : str = "ResponseSummarizer"
    description : str = "useful for summarizing the response of the member cost calculation"
    args_schema : Type[BaseModel] = ResponseSummarizerInput
    model_endpoint_name:str = None
    prompt:str = "Summarize the below notes in a professional manner explaining the details.\
        At the end provide the in-network and out-of-network cost that was calculated.\
        Only return the summmary as answer and reply in plain text without any special characters.\
        Notes: {notes}"

    def __init__(self, model_endpoint_name : str):
        super().__init__()
        self.model_endpoint_name = model_endpoint_name
    
    @mlflow.trace(name="summarize", span_type="func")
    def execute(self,  notes:List[str]) -> str: 
        chain = build_api_chain(self.model_endpoint_name, self.prompt)
        summary = chain.run(notes="\n\n".join(notes))
        return summary.strip()


# COMMAND ----------

# MAGIC %md
# MAGIC #Assemble the Care Cost Compass Application
# MAGIC
# MAGIC ####Now it's time to assemble all the components that we have built so far and build the Agent. 
# MAGIC <img src="https://raw.githubusercontent.com/databricks-industry-solutions/CareCost-Compass/refs/heads/main/resources/build_6.png" alt="Assemble Agent" width="900"/>
# MAGIC
# MAGIC Since we made our components as LangChain Tools, we can use an AgentExecutor to run the process. 
# MAGIC
# MAGIC But since its a very straight forward process, for the sake of reducing latency of response and to improve accuracy, we can use a custom PyFunc model to build our Agent application and deploy it on Databricks Model Serving.
# MAGIC
# MAGIC ####MLflow Python Function
# MAGIC MLflow’s Python function, pyfunc, provides flexibility to deploy any piece of Python code or any Python model. The following are example scenarios where you might want to use the guide.
# MAGIC
# MAGIC * Your model requires preprocessing before inputs can be passed to the model’s predict function.
# MAGIC * Your model framework is not natively supported by MLflow.
# MAGIC * Your application requires the model’s raw outputs to be post-processed for consumption.
# MAGIC * The model itself has per-request branching logic.
# MAGIC * You are looking to deploy fully custom code as a model.
# MAGIC
# MAGIC [Read More](https://docs.databricks.com/en/machine-learning/model-serving/deploy-custom-models.html)
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ###Load Tools

# COMMAND ----------

#a utility function to use logging api instead of print
import logging
def log_print(msg):
    logging.warning(f"=====> {msg}")


# COMMAND ----------

# MAGIC %md
# MAGIC ###Build Agent
# MAGIC ####CareCostCompassAgent
# MAGIC `CareCostCompassAgent` is our Python Function that will implement the logic necessary for our Agent.
# MAGIC
# MAGIC There are two required functions that we need to implement:
# MAGIC
# MAGIC `load_context` - anything that needs to be loaded just one time for the model to operate should be defined in this function. This is critical so that the system minimize the number of artifacts loaded during the predict function, which speeds up inference.
# MAGIC We will be instantiating all the tools in this method
# MAGIC
# MAGIC `predict` - this function houses all the logic that is run every time an input request is made. We will implement the application logic here.
# MAGIC
# MAGIC ####Model Input and Output
# MAGIC Our model is being built as Chat Agent and that dictates the model signature that we are going to use. So, request will be `ChatCompletionRequest`
# MAGIC
# MAGIC The `data` input to a pyfunc model can be a Pandas DataFrame , Pandas Series , Numpy Array, List or a Dictionary. For our implementation we will be expecting a Pandas DataFrame as input. Since its a Chat agent, it will be having the schema of `mlflow.models.rag_signatures.Message`.
# MAGIC
# MAGIC Our response will be just a `mlflow.models.rag_signatures.StringMessage` 
# MAGIC
# MAGIC ####Workflow
# MAGIC We will implement the below workflow in the `predict` method of pyfunc model.
# MAGIC
# MAGIC <img src="https://raw.githubusercontent.com/databricks-industry-solutions/CareCost-Compass/refs/heads/main/resources/logic_workflow.png" width="700">

# COMMAND ----------

import json
import numpy as np
import pandas as pd
from dataclasses import asdict
from mlflow.pyfunc import PythonModel, PythonModelContext
from mlflow.models.rag_signatures import (
    ChatCompletionRequest,
    ChatCompletionResponse, 
    Message,
    StringResponse
)
import asyncio

class CareCostCompassAgent(PythonModel):
  """Agent that can answer questions about medical procedure cost."""

  #lets define the categories for our question classifier
  invalid_question_category = {
    "PROFANITY": "Content has inappropriate language",
    "RACIAL": "Content has racial slur.",
    "RUDE": "Content has angry tone and has unprofessional language.",
    "IRRELEVANT": "The question is not about a medical procedure cost.",
    "GOOD": "Content is a proper question about a cost of medical procedure."
  }

  def load_context(self, context):    
    """
    Loads the context and initilizes the connections
    """

    #get the config from context
    model_config = context.model_config
    
    #instrumentation for feedback app as it does not let you post multiple messages
    #below variables are so that we can use it for review app
    self.environment = model_config["environment"]
    self.default_parameter_json_string = model_config["default_parameter_json_string"]
    
    self.question_classifier_model_endpoint_name = model_config["question_classifier_model_endpoint_name"]
    self.benefit_retriever_model_endpoint_name = model_config["benefit_retriever_model_endpoint_name"]
    self.benefit_retriever_config = RetrieverConfig(**model_config["benefit_retriever_config"])
    self.procedure_code_retriever_config = RetrieverConfig(**model_config["procedure_code_retriever_config"])
    self.summarizer_model_endpoint_name = model_config["summarizer_model_endpoint_name"]
    self.member_table_name = model_config["member_table_name"]
    self.procedure_cost_table_name = model_config["procedure_cost_table_name"]
    self.member_accumulators_table_name = model_config["member_accumulators_table_name"]

    #Start instantiating tools                                    
    self.question_classifier = QuestionClassifier(model_endpoint_name=self.question_classifier_model_endpoint_name,
                            categories_and_description=self.invalid_question_category).get()
    
    self.client_id_lookup = ClientIdLookup(fq_member_table_name=self.member_table_name).get()
    
    self.benefit_rag = BenefitsRAG(model_endpoint_name=self.benefit_retriever_model_endpoint_name,
                              retriever_config=self.benefit_retriever_config).get()
    
    self.procedure_code_retriever = ProcedureRetriever(retriever_config=self.procedure_code_retriever_config).get()

    self.procedure_cost_lookup = ProcedureCostLookup(fq_procedure_cost_table_name=self.procedure_cost_table_name).get()

    self.member_accumulator_lookup = MemberAccumulatorsLookup(fq_member_accumulators_table_name=self.member_accumulators_table_name).get()

    self.member_cost_calculator = MemberCostCalculator().get()

    self.summarizer = ResponseSummarizer(model_endpoint_name=self.summarizer_model_endpoint_name).get()
  
  #we will create three flows that can run parallely
  async def __benefit_flow(self, member_id:str, question:str) -> Benefit:
      ##########################################
      ####Get client id
      log_print("Getting client id:")
      client_id = await self.client_id_lookup.arun({"member_id": member_id})
      if client_id is None:
        raise Exception("Member not found")

      ##########################################
      ####Get Coverage details
      log_print("Getting Coverage details:")
      benefit_json = await self.benefit_rag.arun({"client_id":client_id,"question":question})
      benefit = Benefit.model_validate_json(benefit_json)
      log_print("Coverage details:")
      log_print(benefit_json)
      return benefit
    
  async def __procedure_flow(self, question:str) -> float:
      ##########################################
      ####Get procedure code and description
      proc_code, proc_description = await self.procedure_code_retriever.arun({"question":question})
      log_print("Procedure")
      log_print(f"{proc_code}:{proc_description}")
      
      ##########################################
      ####Get procedure cost
      proc_cost = await self.procedure_cost_lookup.arun({"procedure_code":proc_code})
      if proc_cost is None:
        raise Exception(f"Procedure code {proc_code} not found")
      else:
        log_print(f"Procedure Cost: {proc_cost}")
      
      return proc_cost

  async def __member_accumulator_flow(self, member_id:str) -> dict:
      ##########################################
      ####Get member deductibles"
      member_deductibles = await self.member_accumulator_lookup.arun({"member_id":member_id})
      if member_deductibles is None:
        raise Exception("Member not found")
      else:
        log_print("Member deductibles")
        log_print(member_deductibles)
      
      return member_deductibles

  async def __async_run(self, member_id, question) -> []:
      """Runs the three flows in parallel"""
      tasks = [
        asyncio.create_task(self.__benefit_flow(member_id, question)),
        asyncio.create_task(self.__procedure_flow(question)),
        asyncio.create_task(self.__member_accumulator_flow(member_id))
      ]      
      return await asyncio.gather(*tasks)
      

  @mlflow.trace(name="predict", span_type="func")
  def predict(self, context:PythonModelContext, model_input: pd.DataFrame, params:dict) -> StringResponse:
    """
    Generate answer for the question.

    Args:
        context: The PythonModelContext for the model
        model_input: we will not use this input
        params: Question and member id

    Returns:
        Predicted answer: string
    """
    try:

      log_print("Inside predict")
      ##########################################
      ####Get rows of dataframe as list of messages
      if isinstance(model_input, pd.DataFrame):
          model_input = model_input.to_dict(orient="records")
      else:
        raise Exception("Invalid input: Expecting a pandas.DataFrame")

      
      ##########################################
      ####Get member id and question

      messages = model_input[0]["messages"]
      
      member_id_sentence = None
      question = None
      
      for message in messages:
        if self.environment in ["dev", "test"]:
          ##This workaround is for making our agent work with review app
          ##This is required only because we need two messages in the request
          ##one for member id and other for question
          ##Currently review app does not support multiple messages in the request

          #In non production env, we will use a hardcoded member id
          #dev/test
          parameters = json.loads(self.default_parameter_json_string)
        else:
          #production
          if message["role"] == "system":
            parameter_json = message["content"]
            parameters = json.loads(parameter_json)

        if message["role"] == "user":
          question = message["content"]

      ##########################################      
      ####Filter the question to only those that are valid
      log_print("Filtering:")
      question_category = self.question_classifier.run({"questions":[question]})[0]
      log_print("Question is :{question_category}")
      if question_category != "GOOD":
        log_print(f"Question is invalid: Category: {question_category}")
        error_categories = [c.strip() for c in question_category.split(',')]
        categories = [self.invalid_question_category[c] 
                    if c in self.invalid_question_category else "Unsuitable question" 
                  for c in error_categories]
        error_message = "\n".join(categories)
        raise Exception(error_message)

      ##########################################
      ####Get member id
      log_print("Getting member id:")
      member_id = parameters["member_id"]#self.member_id_retriever.get_member_id(member_id_sentence)
      if member_id is None:
        raise Exception("Invalid member id {member_id}")
      else:
        log_print(f"Member id: {member_id}")

      ############################################
      #### Run the flows, namely benefit, procedure, member_accumulator parallely

      async_results = asyncio.run(self.__async_run(member_id, question))

      benefit = async_results[0]
      proc_cost = async_results[1]
      member_deductibles = async_results[2]

      ##########################################
      ####Calculate member out of pocket cost
      member_cost_calculation = self.member_cost_calculator.run({"benefit":benefit,
                                                                  "procedure_cost":proc_cost,
                                                                  "member_deductibles":member_deductibles
                                                                  })
      log_print("Calculated cost")
      log_print(f"in_network_cost:{member_cost_calculation.in_network_cost}")
      log_print(f"out_network_cost:{member_cost_calculation.out_network_cost}")
      
      return_message = self.summarizer.run({"notes":member_cost_calculation.notes})

    except Exception as e:
      error_string = f"Failed: {repr(e)}"
      logging.error(error_string)
      if len(e.args)>0:
        return_message = f"Sorry, I cannot answer that question because of following reasons:\n {e.args[0]}"
      else:
        return_message = f"Sorry, I cannot answer that question because of an error.\n{repr(e)}"
    
    return_message = asdict(StringResponse(return_message))

    return return_message
  

# COMMAND ----------

## Tell MLflow logging where to find your chain.
mlflow.models.set_model(model=CareCostCompassAgent())
