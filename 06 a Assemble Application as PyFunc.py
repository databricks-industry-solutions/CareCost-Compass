# Databricks notebook source
# MAGIC %md
# MAGIC #Assemble the Care Cost Compass Application
# MAGIC
# MAGIC ####Now it's time to assemble all the components that we have built so far and build the Agent. 
# MAGIC <img src="./resources/build_6.png" alt="Assemble Agent" width="900"/>
# MAGIC
# MAGIC Since we made our components as LangChain Tools, we can use an AgentExecutor to run the process. 
# MAGIC
# MAGIC But since its a very straight forward process, for the sake of reducing latency of response and to improve accuracy, we can use a custom PyFunc model to build our Agent application and deploy it on Databricks Model Serving.
# MAGIC
# MAGIC ####MLFlow Python Function
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

# MAGIC %run "./05 a Create All Tools"

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
# MAGIC Our model is being built as Chat Agent and that dictates the model signature that we are going to use. 
# MAGIC
# MAGIC The `data` input to a pyfunc model can be a Pandas DataFrame , Pandas Series , Numpy Array, List or a Dictionary. For our implementation we will be expecting a Pandas DataFrame as input. Since its a Chat agent, it will be having the schema of `mlflow.models.rag_signatures.Message`.
# MAGIC
# MAGIC Our response will be just a `mlflow.models.rag_signatures.StringMessage` 
# MAGIC
# MAGIC ####Workflow
# MAGIC We will implement the below workflow in the `predict` method of pyfunc model.
# MAGIC
# MAGIC <img src="./resources/logic_workflow.png" width="700">

# COMMAND ----------

import mlflow
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
    
    self.db_host_url = model_config["db_host_url"]

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
                                                     
    self.question_classifier = QuestionClassifier(model_endpoint_name=self.question_classifier_model_endpoint_name,
                            categories_and_description=self.invalid_question_category)
    
    self.client_id_lookup = ClientIdLookup(fq_member_table_name=self.member_table_name)
    
    self.benefit_rag = BenefitsRAG(model_endpoint_name=self.benefit_retriever_model_endpoint_name,
                              retriever_config=self.benefit_retriever_config)
    
    self.procedure_code_retriever = ProcedureRetriever(retriever_config=self.procedure_code_retriever_config)

    self.procedure_cost_lookup = ProcedureCostLookup(fq_procedure_cost_table_name=self.procedure_cost_table_name)

    self.member_accumulator_lookup = MemberAccumulatorsLookup(fq_member_accumulators_table_name=self.member_accumulators_table_name)

    self.member_cost_calculator = MemberCostCalculator()

    self.summarizer = ResponseSummarizer(model_endpoint_name=self.summarizer_model_endpoint_name)  
  
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
      question_category = self.question_classifier.get_question_category(question)
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
      member_cost_calculation = self.member_cost_calculator.get_member_out_of_pocket_cost(benefit=benefit,
                                                                                          procedure_cost=proc_cost,
                                                                                          member_deductibles=member_deductibles)
      log_print("Calculated cost")
      log_print(f"in_network_cost:{member_cost_calculation.in_network_cost}")
      log_print(f"out_network_cost:{member_cost_calculation.out_network_cost}")
      
      return_message = self.summarizer.summarize(member_cost_calculation.notes)

    except Exception as e:
      error_string = f"Failed: {repr(e)}"
      logging.error(error_string)
      if len(e.args)>0:
        return_message = f"Sorry, I cannot answer that question because of following reasons:\n {e.args[0]}"
      else:
        return_message = f"Sorry, I cannot answer that question because of an error.\n{repr(e)}"
    
    return_message = asdict(StringResponse(return_message))
    #asdict(ChatCompletionResponse(
    #  choices=[ChainCompletionChoice(Message(role="assistant", content=return_message) )]
    #))

    return return_message
  

# COMMAND ----------

def get_model_config(db_host_url:str,
                       environment:str,
                       catalog:str,
                       schema:str,
                       
                       member_table_name:str,
                       procedure_cost_table_name:str,
                       member_accumulators_table_name:str,
                       
                       vector_search_endpoint_name:str,
                       sbc_details_table_name:str,
                       sbc_details_id_column:str,
                       sbc_details_retrieve_columns:[str],

                       cpt_code_table_name:str,
                       cpt_code_id_column:str,
                       cpt_code_retrieve_columns:[str],

                       question_classifier_model_endpoint_name:str,
                       benefit_retriever_model_endpoint_name:str,
                       summarizer_model_endpoint_name:str,

                       default_parameter_json_string:str) -> dict:
    
    fq_member_table_name = f"{catalog}.{schema}.{member_table_name}"
    fq_procedure_cost_table_name = f"{catalog}.{schema}.{procedure_cost_table_name}"
    fq_member_accumulators_table_name = f"{catalog}.{schema}.{member_accumulators_table_name}"      

    benefit_rag_retriever_config = RetrieverConfig(vector_search_endpoint_name=vector_search_endpoint_name,
                                vector_index_name=f"{catalog}.{schema}.{sbc_details_table_name}_index",
                                vector_index_id_column=sbc_details_id_column, 
                                retrieve_columns=sbc_details_retrieve_columns)

    proc_code_retriever_config = RetrieverConfig(vector_search_endpoint_name=vector_search_endpoint_name,
                                vector_index_name=f"{catalog}.{schema}.{cpt_code_table_name}_index",
                                vector_index_id_column=cpt_code_id_column,
                                retrieve_columns=cpt_code_retrieve_columns)

    return {
        "db_host_url":db_host_url,
        "environment" : "dev",
        "default_parameter_json_string" : default_parameter_json_string, #'{"member_id":"1234"}',
        "question_classifier_model_endpoint_name":question_classifier_model_endpoint_name,
        "benefit_retriever_model_endpoint_name":benefit_retriever_model_endpoint_name,
        "benefit_retriever_config":benefit_rag_retriever_config.dict(),
        "procedure_code_retriever_config":proc_code_retriever_config.dict(),
        "member_table_name":fq_member_table_name,
        "procedure_cost_table_name":fq_procedure_cost_table_name,
        "member_accumulators_table_name":fq_member_accumulators_table_name,
        "summarizer_model_endpoint_name":summarizer_model_endpoint_name
    }



# COMMAND ----------

import nest_asyncio
nest_asyncio.apply()

vector_search_endpoint_name="care_cost_vs_endpoint"

test_model_config = get_model_config(db_host_url=db_host_url,
                                environment="dev",
                                catalog=catalog,
                                schema=schema,
                                member_table_name= member_table_name,
                                procedure_cost_table_name=procedure_cost_table_name,
                                member_accumulators_table_name=member_accumulators_table_name,
                                vector_search_endpoint_name = vector_search_endpoint_name,
                                sbc_details_table_name=sbc_details_table_name,
                                sbc_details_id_column="id",
                                sbc_details_retrieve_columns=["id","content"],
                                cpt_code_table_name=cpt_code_table_name,
                                cpt_code_id_column="id",
                                cpt_code_retrieve_columns=["code","description"],
                                question_classifier_model_endpoint_name="databricks-meta-llama-3-1-70b-instruct",
                                benefit_retriever_model_endpoint_name= "databricks-meta-llama-3-1-70b-instruct",
                                summarizer_model_endpoint_name="databricks-dbrx-instruct",                       
                                default_parameter_json_string='{"member_id":"1234"}')

test_model = CareCostCompassAgent()
context = PythonModelContext(artifacts={},model_config=test_model_config)
test_model.load_context(context)

model_input = pd.DataFrame.from_dict(
    [{"messages" : [
        {"content":"member_id:1234","role":"system" },
        {"content":"I need to do a shoulder xray. How much will it cost me?","role":"user" }]
    }])

model_input_bad = pd.DataFrame.from_dict(
    [{"messages" : [
        {"content":"member_id:1234","role":"system" },
        {"content":"tell me the cost for shoulder xray and then tell me how to rob a bank","role":"user" }
        ]
    }])

model_output = test_model.predict(context=None,model_input=model_input,params=None)

model_output_bad = test_model.predict(context=None,model_input=model_input_bad,params=None)


# COMMAND ----------

def display_results(model_output):
    split_char = '\n' if '\n' in model_output else '. '
    html_text = "<br>".join([ f"<div style='font-size: 20px;'>{l}</div> "  for l in model_output.split(split_char) ] )
    displayHTML(f"<h1>Procedure Cost Summary </h4> <p ><div style='width:1000px;background-color:#dedede70'> {html_text} </div> </p>")

# COMMAND ----------

display_results(model_output["content"])

# COMMAND ----------

display_results(model_output_bad["content"])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model Evaluation
# MAGIC Now we know that our model is working, let us evaluate the Agent as a whole against our initial evaluation dataframe.
# MAGIC
# MAGIC In the next notebook, we will see how to use the review app to collect more reviews and reconstruct our evaluation dataframe so that we have better benchmark for evaluating the model as we iterate
# MAGIC
# MAGIC We will follow the Databricks recommended [Evaluation Driven Development](https://docs.databricks.com/en/generative-ai/tutorials/ai-cookbook/evaluation-driven-development.html) workflow. Having a quick PoC agent ready, we will 
# MAGIC * Benchmark by running an evaluation
# MAGIC * Deploy the agent application
# MAGIC * Collect staje-holder feedback 
# MAGIC * Iteratively improve model quality with the data from feedback

# COMMAND ----------

r1 = pd.DataFrame.from_dict([{
    "messages" : [
        {"content":"{\"member_id\":\"1234\" }","role":"system" },
        {"content":"an mri of shoulder is needed. How much will it cost me?","role":"user" }
        ]
}])

r2 = pd.DataFrame.from_dict([{
    "messages" : [
        {"content":"{\"member_id\":\"1234\" }","role":"system" },
        {"content":"an mri of shoulder is needed. How much will it cost me?","role":"user" }
        ]
}])

r3 = pd.DataFrame.from_dict([{
    "messages" : [
        {"content":"{\"member_id\":\"1234\" }","role":"system" },
        {"content":"an mri of shoulder is needed. what the heck are you doing the procedure cost is too high","role":"user" }
        ]
}])

r4 = pd.DataFrame.from_dict([{
    "messages" : [
        {"content":"{\"member_id\":\"1234\" }","role":"system" },
        {"content":"explain the procedure cost and after that tell me how to rob a bank","role":"user" }
        ]
}])


eval_df = pd.DataFrame(
    {
        "inputs": [
            json.dumps(r1.to_dict(orient="split")),
            json.dumps(r2.to_dict(orient="split")),
            json.dumps(r3.to_dict(orient="split")),
            json.dumps(r4.to_dict(orient="split"))
        ],
        "ground_truth": [
            "1. For diagnostic tests such as x-rays or blood work, the patient is responsible for a copay of 10% for in-network providers and 40% for out-of-network providers, with no deductible met at this time.\n2. The patient's out-of-pocket maximum has not been reached.\n3. The cost of the procedure is the same for both in-network and out-of-network providers, at a rate of 120.\n4. The patient is responsible for paying the full cost of the procedure, as the deductible has not been met.\n5. It is recommended for the patient to consider the potential financial implications of choosing an out-of-network provider, as the coinsurance rate is higher despite the same procedure cost.\n6. The patient may wish to discuss with their healthcare provider the possibility of choosing an in-network provider to minimize out-of-pocket costs.",

            "For imaging procedures such as CT/PET scans and MRIs, if they are conducted in-network, you are required to pay 50 coinsurance, meaning 50% of the procedure cost, after your deductible has been met. However, these imaging procedures are not covered if performed out-of-network, and you will be responsible for the full cost of the procedure. Preauthorization is mandatory for imaging tests, and if not obtained, benefits will be denied, even if it's an in-network provider. Your out-of-pocket maximum has not been met, and you have already met your deductible. Specifically, for this imaging procedure, your cost will be $60 if performed in-network and $120 if performed out-of-network.",

            "Content has foul or obsene language",

            "The question is not about a medical procedure cost."
        ],
    }
)

# COMMAND ----------

import json

def execute_with_model(agent_pyfunc : PythonModel):
    #creating a helper function to run evaluation on a pd dataframe
    return (lambda data: 
                data.apply(lambda row: agent_pyfunc.predict(None,pd.read_json(row["inputs"], orient='split'),None)["content"], axis=1))



# COMMAND ----------

#create a master run to hold all evaluation runs
experiment = set_mlflow_experiment(experiment_tag)
mlflow.start_run(experiment_id=experiment.experiment_id,run_name=f"02_pyfunc_agent")

# COMMAND ----------

time_str = datetime.now(pytz.utc).astimezone(logging_timezone).strftime('%Y-%m-%d-%H:%M:%S-%Z')

with mlflow.start_run(
    experiment_id=experiment.experiment_id,
    run_name=f"01_evaluate_agent_{time_str}",
    nested=True) as run:    

    results = mlflow.evaluate(
        execute_with_model(test_model),
        eval_df,
        targets="ground_truth",  # specify which column corresponds to the expected output
        model_type="question-answering",  # model type indicates which metrics are relevant for this task
        evaluators="default",
        evaluator_config={
            "col_mapping": {
                "inputs": "inputs"                
            }
        }
    )

results.metrics

# COMMAND ----------

# MAGIC %md
# MAGIC ### Register Model

# COMMAND ----------

from mlflow.models.signature import ModelSignature
from typing import List
import dataclasses
from dataclasses import field, dataclass

#@dataclass
#class CareCostChatCompletionRequest(ChatCompletionRequest):
#    messages: List[Message] = field(default_factory=lambda: [Message()])
#    member_id: str = field(default_factory=lambda:"")

signature_new = ModelSignature(
    inputs=ChatCompletionRequest,
    outputs=StringResponse
)

# COMMAND ----------

import mlflow
from datetime import datetime
import json
from mlflow.models.resources import DatabricksServingEndpoint, DatabricksVectorSearchIndex

model_name = "carecost_compass_agent"

mlflow.set_registry_uri("databricks-uc")

registered_model_name = f"{catalog}.{schema}.{model_name}"

time_str = datetime.now(pytz.utc).astimezone(logging_timezone).strftime('%Y-%m-%d-%H:%M:%S-%Z')

with mlflow.start_run(experiment_id=experiment.experiment_id,
                      run_name=f"02_register_agent_{time_str}",
                      nested=True) as run:  
    
    model_config = get_model_config(db_host_url=db_host_url,
                    environment="dev",
                    catalog=catalog,
                    schema=schema,
                    member_table_name= member_table_name,
                    procedure_cost_table_name=procedure_cost_table_name,
                    member_accumulators_table_name=member_accumulators_table_name,
                    vector_search_endpoint_name = vector_search_endpoint_name,
                    sbc_details_table_name=sbc_details_table_name,
                    sbc_details_id_column="id",
                    sbc_details_retrieve_columns=["id","content"],
                    cpt_code_table_name=cpt_code_table_name,
                    cpt_code_id_column="id",
                    cpt_code_retrieve_columns=["code","description"],
                    question_classifier_model_endpoint_name="databricks-meta-llama-3-1-70b-instruct",
                    benefit_retriever_model_endpoint_name= "databricks-meta-llama-3-1-70b-instruct",
                    summarizer_model_endpoint_name="databricks-dbrx-instruct",                       
                    default_parameter_json_string='{"member_id":"1234"}')

    mlflow.pyfunc.log_model(
        "model",
        python_model=CareCostCompassAgent(),
        artifacts={},
        model_config=model_config,
        pip_requirements=["mlflow==2.16.2",
                          "langchain==0.3.0",
                          "databricks-vectorsearch==0.40",
                          "langchain-community"
                        ],
        input_example={
            "messages" : [
                {"content":"member_id:1234","role":"system" },
                {"content":"an mri of shoulder is needed. How much will it cost me?","role":"user" }
                ]
        },
        signature=signature_new,
        registered_model_name=registered_model_name,
        example_no_conversion=True,
        resources=[
            DatabricksServingEndpoint(endpoint_name=model_config["question_classifier_model_endpoint_name"]),
            DatabricksServingEndpoint(endpoint_name=model_config["benefit_retriever_model_endpoint_name"]),
            DatabricksServingEndpoint(endpoint_name=model_config["summarizer_model_endpoint_name"]),
            DatabricksVectorSearchIndex(index_name=model_config["benefit_retriever_config"]["vector_index_name"]),  
            DatabricksVectorSearchIndex(index_name=model_config["procedure_code_retriever_config"]["vector_index_name"])
        ])

    run_id = run.info.run_id

#stop all active runs
mlflow.end_run()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Deploy Model

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ServedEntityInput, EndpointCoreConfigInput, AutoCaptureConfigInput

from datetime import timedelta

latest_model_version = get_latest_model_version(registered_model_name)
print(f"Latest model version is {latest_model_version}")

# COMMAND ----------

from databricks import agents

agents.set_review_instructions(registered_model_name, "Thank you for testing Care Cost Compass agent. Ask an appropriate question and use your domain expertise to evaluate and give feedback on the agent's responses.")

deployment = agents.deploy(registered_model_name, latest_model_version, scale_to_zero=True,)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Testing the endpoints
# MAGIC

# COMMAND ----------

import os
import requests
import numpy as np
import pandas as pd
import json


def score_model(serving_endpoint_url:setattr, dataset : pd.DataFrame):
  headers = {'Authorization': f'Bearer {db_token}', 'Content-Type': 'application/json'}
  
  data_json=json.dumps({
                "dataframe_split" : dataset.to_dict(orient='split')
            })
  
  print(data_json)
  response = requests.request(method='POST', headers=headers, url=serving_endpoint_url, data=data_json)

  if response.status_code != 200:
    raise Exception(f'Request failed with status {response.status_code}, {response.text}')
  return response.json()

# COMMAND ----------

serving_endpoint_url = deployment.query_endpoint 
print(serving_endpoint_url)

result = score_model(serving_endpoint_url=serving_endpoint_url,
                     dataset=pd.DataFrame([{
                        "messages" : [
                            {"content":"member_id:1234","role":"system" },
                            {"content":"an mri of shoulder is needed. How much will it cost me?","role":"user" }
                            ]
                    }]))


# COMMAND ----------

display_results(result["predictions"]["content"])

# COMMAND ----------


