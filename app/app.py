import logging
import os
import requests
import numpy as np
import pandas as pd
import json
import streamlit as st
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure environment variable is set correctly
assert os.getenv('SERVING_ENDPOINT'), "SERVING_ENDPOINT must be set in app.yaml."
HOSTNAME = os.getenv('DATABRICKS_HOST')

def access_token_for_service_principal():    
    # Set your environment variables or replace them directly here
    CLIENT_ID = os.getenv("DATABRICKS_CLIENT_ID")
    CLIENT_SECRET = os.getenv("DATABRICKS_CLIENT_SECRET")
    ENDPOINT_ID = os.getenv("SERVING_ENDPOINT")
    ACTION = "query_inference_endpoint" # Can also be `manage_inference_endpoint`

    # Token endpoint URL
    TOKEN_URL = f"https://{HOSTNAME}/oidc/v1/token"

    # Build the payload, note the creation of authorization_details
    payload = { 'grant_type': 'client_credentials', 'scope': 'all-apis' }

    print(f"TOKEN_URL: {TOKEN_URL}")
    print(payload)

    # Make the POST request with basic auth
    response = requests.post( TOKEN_URL, auth=(CLIENT_ID, CLIENT_SECRET), data=payload )

    # Check the response
    if response.ok:
        token_response = response.json()
        access_token = token_response.get("access_token")

        if access_token:
            return access_token
        else:
            print("access_token not found in response.")
            raise Exception("Access token not found in response.")
    else: 
        print(f"Failed to fetch token: {response.status_code} {response.text}")
        raise Exception(f"Failed to fetch token: {response.status_code} {response.text}")


def get_user_info():
    headers = st.context.headers
    user_access_token = headers.get("X-Forwarded-Access-Token")
    user_name=headers.get("X-Forwarded-Preferred-Username")
    user_display_name = ""
    if user_access_token:
        # Initialize WorkspaceClient with the user's token
        w = WorkspaceClient(token=user_access_token, auth_type="pat")
        # Get current user information
        current_user = w.current_user.me()
        # Display user information
        user_display_name = current_user.display_name
        

    return dict(
        user_name=user_name,
        user_email=headers.get("X-Forwarded-Email"),
        user_id=headers.get("X-Forwarded-User"),
        user_access_token = headers.get("X-Forwarded-Access-Token"),
        user_display_name = user_display_name if user_display_name != "" else user_name
    )

def score_model(serving_endpoint_url:str, message_dict : dict, token: str):
      headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}  
      data_json = json.dumps(message_dict)
      logger.info(serving_endpoint_url)
      logger.info(data_json)
      response = requests.request(method='POST', headers=headers, url=serving_endpoint_url, data=data_json)

      if response.status_code != 200:
        raise Exception(f'Request failed with status {response.status_code}, {response.text}')
      return response.json()
    
def on_preset_message_change():
    st.session_state.preset_message = st.session_state.preset_message_pill
    st.session_state.preset_message_pill = None


user_info = get_user_info()
print(user_info)

# Streamlit app
if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

c1, c2 = st.columns([2,1])
with c1:
    st.markdown("### :material/ecg: Care Cost Compass ")
with c2:
    st.write(f"Hello, {user_info['user_display_name'].split(' ')[0]}")
    st.write(f"Member Id: 1234")

st.markdown("#### Procedure Cost Estimator")
st.write("I can help you with questions related to Procedure Cost.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

st.divider()

if 'preset_message' not in st.session_state:
    st.session_state.preset_message = None

preset_message = st.pills("You can ask questions like:",
         ["How much will a chest X Ray cost?", "What will be the cost of a knee replacement?", "How much will a shoulder MRI cost?"],
         default=None, key="preset_message_pill", on_change=on_preset_message_change)

prompt = st.chat_input(key="chat_input")

if st.session_state.preset_message is not None:
    preset_message_selection = st.session_state.preset_message
    st.session_state.preset_message = None
else:
    preset_message_selection = None

# Accept user input
if preset_message_selection or prompt:
    user_query = preset_message_selection if preset_message_selection else prompt
    #hard coded member id to mimic an internal app
    system_message_with_member_id = {"role": "system", "content": "member_id:1234"}
    #actual query
    query_message = {"role": "user", "content": user_query}
    # Add user message to chat history
    st.session_state.messages.append(query_message)
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(user_query)

    messages = { "messages": [system_message_with_member_id, query_message] } 
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("Calculating..."):
            # Query the Databricks serving endpoint
            try:
                serving_endpoint_url = f"https://{HOSTNAME}/serving-endpoints/{os.getenv('SERVING_ENDPOINT')}/invocations"
                response = score_model(serving_endpoint_url, messages, access_token_for_service_principal() )#user_info['user_access_token'])
                logger.info(response)
                assistant_response = response["content"]
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                #st.write(assistant_response)                
                st.rerun()
            except Exception as e:
                st.error(f"Error querying model: {e}")
                assistant_response="An error occured"
    

    

