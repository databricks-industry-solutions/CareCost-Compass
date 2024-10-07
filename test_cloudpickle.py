# Databricks notebook source
# MAGIC %pip install langchain==0.3.0 langchain-community==0.3.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from typing import Optional, Type, List, Union
import langchain
import langchain.tools
from langchain.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

import cloudpickle
import mlflow
import asyncio


# COMMAND ----------

def say_hello(member:str):
    """Useful function to say hello"""
    print(f"hello {member}")

class MyToolInput(BaseModel):
    member: str = Field(description="member id to say hello")

class MyTool(StructuredTool):
    name:str = "MyTool"
    description:str = "useful tool for saying hello"
    args_schema : Type[BaseModel] = MyToolInput
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _run(self, member:str, run_manager: Optional[CallbackManagerForToolRun] = None):
        say_hello(member)
    
    async def _arun(self,member:str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None):
        say_hello(member)



# COMMAND ----------

t1 = StructuredTool.from_function(func=say_hello)
t2 = MyTool()

print(isinstance(t1, StructuredTool))
print(isinstance(t2, StructuredTool))


# COMMAND ----------

class ToolUser(BaseModel):
    tool1 : Type[BaseTool] = None
    tool2 : Type[BaseTool] = None

    def load_context(self):
        self.tool1 = t1
        self.tool2 = t2
        print(type(self.tool1).__mro__)
        print("====================")
        print(type(self.tool2).__mro__)

    def use_tool(self):
        asyncio.run(self.tool1.arun({"member":"Srijit from Tool 1"}))
        asyncio.run(self.tool2.arun({"member":"Srijit from Tool 2"}))

# COMMAND ----------

import nest_asyncio
nest_asyncio.apply()

tool_user = ToolUser()
tool_user.load_context()

print("**********************")
tool_user.use_tool()

# COMMAND ----------

with open("/tmp/pickly.pkl", "wb") as f:  
    cloudpickle.dump(tool_user, f)

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import cloudpickle

with open("/tmp/pickly.pkl", "rb") as f: 
    unpickly = cloudpickle.load(f)

import nest_asyncio
nest_asyncio.apply()

unpickly.load_context()

print("**********************")
unpickly.use_tool()

# COMMAND ----------


