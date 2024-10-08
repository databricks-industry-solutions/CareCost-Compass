# Databricks notebook source
# MAGIC %pip install pyautogen
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run "./05 a Create All Tools and Model"

# COMMAND ----------

def generate_autogen_config(langchain_tools: List[StructuredTool]):
    function_schema = []
    function_map = {}
    
    for tool in langchain_tools:
        function_schema.append({
            "name": tool.name.lower().replace (' ', '_'),
            "description": tool.description,
            "parameters": {
                "type": "object",
                "properties": {} if tool.args is None else tool.args,
                "required": [],
            },
        })
        function_map[tool.name.lower().replace (' ', '_')]=tool._run

    print("****************")
    print(function_schema)
    print("****************")
    print(function_map)

    return function_schema, function_map

# COMMAND ----------

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import AIMessage, HumanMessage
from autogen import ConversableAgent, UserProxyAgent, AssistantAgent, config_list_from_json

os.environ['DATABRICKS_HOST'] = db_host_url
os.environ['DATABRICKS_TOKEN'] = db_token
os.environ["AUTOGEN_USE_DOCKER"] = "False"

class CareCostReactAgent:
    
    max_tokens=4096
    temperature=0.01
    invalid_question_category = {
        "PROFANITY": "Content has inappropriate language",
        "RACIAL": "Content has racial slur.",
        "RUDE": "Content has angry tone and has unprofessional language.",
        "IRRELEVANT": "The question is not about a medical procedure cost.",
        "GOOD": "Content is a proper question about a cost of medical procedure."
    }
    agent_prompt = "You are a helpful assistant who can answer questions about medical procedure costs.\
                        Only use the functions you have been provided with. Reply TERMINATE when the task is done.\
                            First classify the question and if only GOOD proceed with the question. If not, terminate the conversation.\
                                Once you have all inputs use member cost calculation and summarize the response. " 

    def __init__(self, model_config:dict):
        self.db_host_url = model_config["db_host_url"]

        #instrumentation for feedback app as it does not let you post multiple messages
        #below variables are so that we can use it for review app
        self.environment = model_config["environment"]
        self.default_parameter_json_string = model_config["default_parameter_json_string"]

        self.agent_chat_model_endpoint_name = model_config["agent_chat_model_endpoint_name"]
        self.member_id_retriever_model_endpoint_name = model_config["member_id_retriever_model_endpoint_name"]        
        self.question_classifier_model_endpoint_name = model_config["question_classifier_model_endpoint_name"]
        self.benefit_retriever_model_endpoint_name = model_config["benefit_retriever_model_endpoint_name"]
        self.benefit_retriever_config = RetrieverConfig(**model_config["benefit_retriever_config"])
        self.procedure_code_retriever_config = RetrieverConfig(**model_config["procedure_code_retriever_config"])
        self.summarizer_model_endpoint_name = model_config["summarizer_model_endpoint_name"]
        self.member_table_name = model_config["member_table_name"]
        self.procedure_cost_table_name = model_config["procedure_cost_table_name"]
        self.member_accumulators_table_name = model_config["member_accumulators_table_name"]
                                                        
        self.member_id_retriever = MemberIdRetriever(model_endpoint_name=self.member_id_retriever_model_endpoint_name)

        self.question_classifier = QuestionClassifier(model_endpoint_name=self.question_classifier_model_endpoint_name,
                                categories_and_description=self.invalid_question_category
                                )
        
        self.client_id_lookup = ClientIdLookup(fq_member_table_name=self.member_table_name)
        
        self.benefit_rag = BenefitsRAG(model_endpoint_name=self.benefit_retriever_model_endpoint_name,
                                retriever_config=self.benefit_retriever_config
                                )
        
        self.procedure_code_retriever = ProcedureRetriever(retriever_config=self.procedure_code_retriever_config)

        self.procedure_cost_lookup = ProcedureCostLookup(fq_procedure_cost_table_name=self.procedure_cost_table_name)

        self.member_accumulator_lookup = MemberAccumulatorsLookup(fq_member_accumulators_table_name=self.member_accumulators_table_name)

        self.member_cost_calculator = MemberCostCalculator()

        self.summarizer = ResponseSummarizer(model_endpoint_name=self.summarizer_model_endpoint_name)

        self.langchain_tools = [
            self.member_id_retriever,
            self.question_classifier,
            self.client_id_lookup,
            self.benefit_rag,
            self.procedure_code_retriever,
            self.procedure_cost_lookup,
            self.member_accumulator_lookup,
            self.member_cost_calculator,
            self.summarizer
        ]

        autogen_functions, autogen_function_maps = generate_autogen_config(self.langchain_tools)

        self.chat_model = ChatDatabricks(
            endpoint=self.agent_chat_model_endpoint_name,
            max_tokens = self.max_tokens,
            temperature=self.temperature
        )

        self.agent_config = {
            "functions": autogen_functions,
            "config_list": [{
                "model" : "srijit_nair_openai", #"databricks-meta-llama-3-1-70b-instruct",
                "api_key": str(os.environ["DATABRICKS_TOKEN"]),
                "base_url": str(os.getenv("DATABRICKS_HOST"))+"/serving-endpoints"
            }],
            "timeout": 120,
            }


        self.agent = AssistantAgent(
            name="carecost_agent",
            system_message=self.agent_prompt,
            llm_config=self.agent_config,
        )

        self.user_proxy = UserProxyAgent(
            name="user_proxy",
            is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
            human_input_mode="NEVER",
            max_consecutive_auto_reply=20,
            code_execution_config={"work_dir": "coding"},
        )

        self.user_proxy.register_function(
            function_map=autogen_function_maps
        )

    def answer(self, member_id:str ,input_question:str) -> str:

        self.user_proxy.initiate_chat(
            self.agent,
            message=f"Member id is {member_id}, Question is: {input_question}"
        )





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

                       agent_chat_model_endpoint_name:str,
                       member_id_retriever_model_endpoint_name:str,
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
        "agent_chat_model_endpoint_name":agent_chat_model_endpoint_name,
        "member_id_retriever_model_endpoint_name":member_id_retriever_model_endpoint_name,
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


care_cst_agent = CareCostReactAgent(model_config=get_model_config(db_host_url=db_host_url,
                                environment="dev",
                                catalog=catalog,
                                schema=schema,
                                member_table_name= member_table_name,
                                procedure_cost_table_name=procedure_cost_table_name,
                                member_accumulators_table_name=member_accumulators_table_name,
                                vector_search_endpoint_name = "care_cost_vs_endpoint",
                                sbc_details_table_name=sbc_details_table_name,
                                sbc_details_id_column="id",
                                sbc_details_retrieve_columns=["id","content"],
                                cpt_code_table_name=cpt_code_table_name,
                                cpt_code_id_column="id",
                                cpt_code_retrieve_columns=["code","description"],
                                agent_chat_model_endpoint_name="srijit_nair_openai" ,# "databricks-meta-llama-3-1-405b-instruct",
                                member_id_retriever_model_endpoint_name="databricks-mixtral-8x7b-instruct",
                                question_classifier_model_endpoint_name="databricks-meta-llama-3-1-70b-instruct",
                                benefit_retriever_model_endpoint_name= "databricks-meta-llama-3-1-70b-instruct",
                                summarizer_model_endpoint_name="databricks-dbrx-instruct",                       
                                default_parameter_json_string='{"member_id":"1234"}'))

# COMMAND ----------

care_cst_agent.answer(member_id = "1234", input_question="What is the total cost of a shoulder mri")

# COMMAND ----------


