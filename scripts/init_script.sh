#!/bin/bash
pip install --quiet opencv-python==4.8.0.74 camelot-py==0.11.0 pdfplumber==0.11.4
pip install --quiet typing-inspect==0.8.0 typing-extensions==4.12.2
pip install -q mlflow==2.16.2 databricks-vectorsearch==0.40 langchain==0.3.0 langchain-community==0.3.0 mlflow[databricks] databricks-agents==0.6.0 databricks-sdk==0.28.0 
pip install camelot-py[cv]==0.11.0
pip install 'PyPDF2<3.0'