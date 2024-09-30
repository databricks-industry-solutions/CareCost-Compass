# Databricks notebook source
# MAGIC %md
# MAGIC ###Create All Tools
# MAGIC
# MAGIC <img src="./resources/build_05.png" alt="Create Tools" width="900" />

# COMMAND ----------

# MAGIC %run ./utils/utils

# COMMAND ----------

# MAGIC %md
# MAGIC ## Adding building blocks for the Gen AI Application
# MAGIC
# MAGIC All the building blocks will be developed as Tools

# COMMAND ----------


from typing import Optional, Type, List, Union

from pydantic import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool
from langchain.callbacks.manager import (AsyncCallbackManagerForToolRun, CallbackManagerForToolRun)

import os
from databricks.vector_search.client import VectorSearchClient
from databricks.vector_search.index import VectorSearchIndex

from langchain_core.documents.base import Document
from langchain.output_parsers import PydanticOutputParser

# COMMAND ----------

import pandas as pd

class ToolRunner:
    """A helper class to run a Tool and get results from the tool."""
    def __init__(self,
                 tool:BaseTool,
                 evaluation_data:pd.DataFrame,
                 tool_input_columns:str,
                 tool_additional_output:str=None,
                 iterate_rows:bool=False):
        
        self.tool = tool
        self.evaluation_data = evaluation_data
        self.tool_input_columns = tool_input_columns
        self.tool_additional_output = tool_additional_output
        self.iterate_rows = iterate_rows
        self.tool_output = []

    def run_tool(self, data:pd.DataFrame):
        if self.iterate_rows:
            tool_result = []
            tool_output= []
            for index, row in data.iterrows():
                input_dict = { col:row[col] for col in self.tool_input_columns}
                print(f"Running tool with input: {input_dict}")
                tool_result.append(self.tool.run(input_dict))

                if self.tool_additional_output is not None:
                    tool_output.append(getattr(self.tool, self.tool_additional_output))
                    setattr(self, self.tool_additional_output, tool_output)
        else:
            input_dict = { col:self.evaluation_data[col].tolist()  for col in self.tool_input_columns}
            print(f"Running tool with input: {input_dict}")
            tool_result = self.tool.run(input_dict)

            if self.tool_additional_output is not None:
                tool_output = getattr(self.tool, self.tool_additional_output)
                setattr(self, self.tool_additional_output, tool_output)
        
        return tool_result


class RetrieverConfig(BaseModel):
    """A data class for passing around vector index configuration"""
    vector_search_endpoint_name:str
    vector_index_name:str
    vector_index_id_column:str
    retrieve_columns:List[str]

# COMMAND ----------

# MAGIC %md
# MAGIC ###Add Member Id Retriever
# MAGIC

# COMMAND ----------

class MemberIdRetrieverInput(BaseModel):
    question: str = Field(description="Sentence containing member_id")

class  MemberIdRetriever(BaseTool):
    name : str = " MemberIdRetriever"
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
    def get_member_id(self, question:str) -> str: 
        chain = build_api_chain(self.model_endpoint_name, self.prompt)
        category = chain.run(question=question)
        return category.strip()
    
    def _run(self, question:str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        return self.get_member_id(question)
    
    def _arun(self, question:str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        return self.get_member_id(question)


# COMMAND ----------

# MAGIC %md
# MAGIC ###Adding Content Filter
# MAGIC We will create the content filter component as a Natural Language Classfier
# MAGIC

# COMMAND ----------



class QuestionClassifierInput(BaseModel):
    questions: List[str] = Field(description="Question to be classified")

class QuestionClassifier(BaseTool):
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
    def get_question_category(self, question:str) -> str: 
        chain = build_api_chain(self.model_endpoint_name, self.prompt)
        category = chain.run(categories=self.category_str, question=question)
        return category.strip()
    
    def _run(self, questions:[str], run_manager: Optional[CallbackManagerForToolRun] = None) -> [str]:
        return [self.get_question_category(question) for question in questions]
    
    def _arun(self, questions:[str], run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> [str]:
        return [self.get_question_category(question) for question in questions]
    

# COMMAND ----------

# MAGIC %md
# MAGIC ###Adding Benefit RAG

# COMMAND ----------

class BenefitsRetriever():    
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
    client_id : str = Field(description="Client ID for which the benefits need to be retrieved")
    question: str = Field(description="Question for which the benefits need to be retrieved")

class Benefit(BaseModel):
    text:str = Field(description="Full text as provided in the context as-is without changing anything")
    in_network_copay:float = Field(description="In Network copay amount. Set to -1 if not covered or has coinsurance")
    in_network_coinsurance:float= Field(description="In Network coinsurance amount without the % sign. Set to -1 if not covered or has copay")
    out_network_copay:float = Field(description="Out of Network copay amount. Set to -1 if not covered or has coinsurance")
    out_network_coinsurance:float = Field(description="Out of Network coinsurance amount without the % sign. Set to -1 if not covered or has copay")
    
class BenefitsRAG(BaseTool):
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
    def get_benefits(self, client_id:str, question:str) -> str:

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
            return answer# Benefit.model_validate_json(answer)
        else:
            raise Exception("No coverage found")

    def _run(self, client_id:str, question:str,run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        return self.get_benefits(client_id, question)
    
    def _arun(self, client_id:str, question:str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        return self.get_benefits(client_id, question)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Adding Procedure Retriever

# COMMAND ----------

class ProcedureRetrieverInput(BaseModel):
    question: str = Field(description="Question for which the procedure need to be retrieved")

#Expects DATABRICKS_TOKEN and DATABRICKS_HOST env vars
class ProcedureRetriever(BaseTool):    
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
    def get_procedure_details(self, question:str) -> (str,str):
        query_results = self.vector_index.similarity_search(
            query_text=question,
            columns=self.retriever_config.retrieve_columns,
            num_results=1)

        if query_results["result"]["row_count"] > 0:      
            procedure_detail = query_results["result"]["data_array"][0]
            return (procedure_detail[0],procedure_detail[1])
        else:
            raise Exception("No procedure found.")
                            
    def _run(self, question:str,run_manager: Optional[CallbackManagerForToolRun] = None) -> (str,str):
        return self.get_procedure_details(question)
    
    def _arun(self, question:str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> (str,str):
        return self.get_procedure_details(question)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Adding Client Id Lookup

# COMMAND ----------

class ClientIdLookupInput(BaseModel):
    member_id: str = Field(description="Member ID using which we need to lookup client id")

class ClientIdLookup(BaseTool):    
    name : str = "ClientIdLookup"
    description : str = "useful for retrieving a client id given a member id"
    args_schema : Type[BaseModel] = ClientIdLookupInput
    fq_member_table_name:str = None

    def __init__(self, fq_member_table_name:str):
        super().__init__()
        self.fq_member_table_name = fq_member_table_name
    
    @mlflow.trace(name="get_client_id", span_type="func")
    def get_client_id(self, member_id:str) -> str:
        member_data = get_data_from_online_table(self.fq_member_table_name, {"member_id":member_id})
        print(member_data)
        return member_data["outputs"][0]["client_id"]

    def _run(self, member_id:str,run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        return self.get_client_id(member_id)
    
    def _arun(self, member_id:str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        return self.get_client_id(member_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Adding Procedure Cost Lookup

# COMMAND ----------

class ProcedureCostLookupInput(BaseModel):
    procedure_code: str = Field(description="Procedure Code for which to find the cost")

class ProcedureCostLookup(BaseTool):    
    name : str = "ProcedureCostLookup"
    description : str = "useful for retrieving the cost of a procedure given the procedure code"
    args_schema : Type[BaseModel] = ProcedureCostLookupInput
    fq_procedure_cost_table_name:str = None

    def __init__(self, fq_procedure_cost_table_name:str):
        super().__init__()
        self.fq_procedure_cost_table_name = fq_procedure_cost_table_name
    
    @mlflow.trace(name="get_procedure_cost", span_type="func")
    def get_procedure_cost(self, procedure_code:str) -> float:
        procedure_cost_data = get_data_from_online_table(self.fq_procedure_cost_table_name, {"procedure_code":procedure_code})
        return procedure_cost_data["outputs"][0]["cost"]

    def _run(self, procedure_code:str,run_manager: Optional[CallbackManagerForToolRun] = None) -> float:
        return self.get_procedure_cost(procedure_code)
    
    def _arun(self, procedure_code:str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> float:
        return self.get_procedure_cost(procedure_code)

# COMMAND ----------

rr = pd.DataFrame( {
    "messages" : [
        {"content":"Member id is 1234.","role":"user" },
        {"content":"an mri of shoulder is needed. How much will it cost me?","role":"user" }
        ]
})

rr.to_dict(orient="records")

# COMMAND ----------

# MAGIC %md
# MAGIC ###Adding Member Accumulators Lookup

# COMMAND ----------

class MemberAccumulatorsLookupInput(BaseModel):
    member_id: str = Field(description="Member Id for which we need to lookup the accumulators")

class MemberAccumulatorsLookup(BaseTool):    
    name : str = "MemberAccumulatorsLookup"
    description : str = "useful for retrieving the accumulators like deductibles given a member id"
    args_schema : Type[BaseModel] = MemberAccumulatorsLookupInput
    fq_member_accumulators_table_name:str = None

    def __init__(self, fq_member_accumulators_table_name:str):
        super().__init__()
        self.fq_member_accumulators_table_name = fq_member_accumulators_table_name
    
    @mlflow.trace(name="get_member_accumulators", span_type="func")
    def get_member_accumulators(self, member_id:str) -> dict[str, Union[float,str] ]:
        accumulator_data = get_data_from_online_table(self.fq_member_accumulators_table_name, {"member_id":member_id})
        return accumulator_data["outputs"][0]

    def _run(self, member_id:str,run_manager: Optional[CallbackManagerForToolRun] = None) -> dict[str, Union[float,str] ]:
        return self.get_member_accumulators(member_id)
    
    def _arun(self, member_id:str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> dict[str, Union[float,str] ]:
        return self.get_member_accumulators(member_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Adding Cost Calculator
# MAGIC Cost Calculator is a deterministic method that takes member benefits, deductibles and procedure cost to calculate the out of pocket cost that member could be paying for the procedure

# COMMAND ----------

class MemberCost(BaseModel):
    in_network_cost : float = Field(description="In-Network cost of the procedure")
    out_network_cost : float = Field(description="Out-Network cost of the procedure")
    notes : List[str] = Field(description="Notes about the cost calculation")


class MemberCostCalculatorInput(BaseModel):
    benefit:Benefit = Field(description="Benefit object for the member")
    procedure_cost:float = Field(description="Cost of the procedure")
    member_deductibles:dict[str, Union[float,str] ] = Field(description="Accumulators for the member")


class MemberCostCalculator(BaseTool):
    name : str = "MemberCostCalculator"
    description : str = "calculates the estimated member out of pocket cost given the benefits, procedure cost and deductibles"
    args_schema : Type[BaseModel] = MemberCostCalculatorInput

    def __init__(self):
        super().__init__()

    @mlflow.trace(name="get_member_out_of_pocket_cost", span_type="func")
    def get_member_out_of_pocket_cost(self, 
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

    def _run(self,
             benefit:Benefit,
             procedure_cost:float,
             member_deductibles:dict[str, Union[float,str] ],
             run_manager: Optional[CallbackManagerForToolRun] = None) -> MemberCost:
        return self.get_member_out_of_pocket_cost(benefit,procedure_cost,member_deductibles)
    
    def _run(self,
             benefit:Benefit,
             procedure_cost:float,
             member_deductibles:dict[str, Union[float,str] ],
             run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> MemberCost:
        return self.get_member_out_of_pocket_cost(benefit,procedure_cost,member_deductibles)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Adding Summarizer

# COMMAND ----------

class ResponseSummarizerInput(BaseModel):
    notes:List[str] = Field(description="MemberCost object for the member")    

class ResponseSummarizer(BaseTool):
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
    def summarize(self,  notes:List[str]) -> str: 
        chain = build_api_chain(self.model_endpoint_name, self.prompt)
        summary = chain.run(notes="\n\n".join(notes))
        return summary.strip()
    
    def _run(self, notes:List[str], run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        return self.summarize(notes)
    
    def _arun(self, notes:List[str], run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        return self.summarize(notes)
