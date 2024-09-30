#!/bin/bash

apt-get update
apt-get install -y --fix-missing ghostscript python3-tk

pip install -q mlflow==2.16.2 databricks-vectorsearch==0.40 databricks-sdk==0.28.0 langchain==0.3.0 langchain-community mlflow[databricks] 
pip install databricks-agents

pip install --quiet opencv-python==4.8.0.74 ghostscript camelot-py
pip install --quiet typing-inspect==0.8.0 typing-extensions==4.12.2