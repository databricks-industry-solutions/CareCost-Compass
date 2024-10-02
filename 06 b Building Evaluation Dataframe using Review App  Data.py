# Databricks notebook source
# MAGIC %md
# MAGIC #Evaluation Driven Development 
# MAGIC ![Evaluation Driven Workflow](https://docs.databricks.com/en/_images/workflow.png)
# MAGIC
# MAGIC Evaluation Driven Workflow is based on the Mosaic Research team’s recommended best practices for building and evaluating high-quality RAG applications.
# MAGIC
# MAGIC Databricks recommends the following evaluation-driven workflow:
# MAGIC 1. Define the requirements.
# MAGIC 2. Collect stakeholder feedback on a rapid proof of concept (POC).
# MAGIC 3. Evaluate the POC’s quality.
# MAGIC 4. Iteratively diagnose and fix quality issues.
# MAGIC 5. Deploy to production.
# MAGIC 6. Monitor in production.
# MAGIC
# MAGIC [Read More](https://docs.databricks.com/en/generative-ai/tutorials/ai-cookbook/evaluation-driven-development.html)
# MAGIC
# MAGIC #Review App and Getting Human Feedback
# MAGIC The Databricks Review App stages the agent in an environment where expert stakeholders can interact with it - in other words, have a conversation, ask questions, and so on. In this way, the review app lets you collect feedback on your application, helping to ensure the quality and safety of the answers it provides.
# MAGIC
# MAGIC Stakeholders can chat with the application bot and provide feedback on those conversations, or provide feedback on historical logs, curated traces, or agent outputs.
# MAGIC
# MAGIC [Read more](https://docs.databricks.com/en/generative-ai/agent-evaluation/human-evaluation.html)
# MAGIC
# MAGIC One of the core concepts of Evaluation Driven Workflow is to objectively measure quality of the model. Reviews captured by the review app can now serve as a basis to create a better evaluation dataframe for our Agent as we iterate to improve quality. This notebook demonstrates that process.

# COMMAND ----------


