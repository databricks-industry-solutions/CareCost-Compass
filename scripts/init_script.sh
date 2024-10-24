#!/bin/bash
pip install --quiet opencv-python==4.8.0.74 camelot-py
pip install --quiet typing-inspect==0.8.0 typing-extensions==4.12.2
pip install -q mlflow==2.16.2 databricks-vectorsearch==0.40 databricks-sdk==0.28.0 langchain==0.3.0 langchain-community==0.3.0 mlflow[databricks] databricks-agents==0.6.0
pip install camelot-py[cv]
pip install 'PyPDF2<3.0'