import streamlit as st
import requests
import json
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom styling to match the image
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    body, .stApp, .chat-container, .header-container, .message-container, .bot-name, .bot-message, .stTextInput input {
        font-family: 'Inter', monospace;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .stApp {
        background-color: #daecfe !important;
    }
    
    /* Chat container */
    .chat-container {
        display: flex;
        flex-direction: column;
    }
    
    /* Header styling */
    .header-container {
        background-color: #f3faff;
        padding: 15px;
        border-radius: 15px 15px 0 0;
        display: flex;
        align-items: center;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 0;
    }
    
    /* Message styling */
    .message-container {
        padding: 10px 15px;
        background-color: #FFFFFFFF;
        border-radius: 0 0 15px 15px;
        margin-bottom: 0;
        margin-top: 0;
    }
    
    .bot-icon {
        color: #daecfe;
        background-color: #daecfe;
        padding: 5px;
        border-radius: 50%;
        margin-right: 10px;
        font-size: 20px;
    }
    
    .bot-name {
        color: #0a2e5c;
        margin-left: 5px;
        font-weight: 500;
    }
    
    .bot-message {
        background-color: #0046c7;
        border-radius: 15px;
        padding: 12px;
        margin: 3px 0;
        display: inline-block;
        max-width: 80%;
        color: white;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    /* User message styling */
    .user-message {
        background-color: #0046c7;
        color: white;
        border-radius: 15px;
        padding: 12px;
        margin: 3px 0;
        display: inline-block;
        max-width: 80%;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        float: right;
    }
    
    /* User message container */
    .user-container {
        display: flex;
        flex-direction: column;
        align-items: flex-end;
        width: 100%;
        padding: 10px 15px;
        background-color: #FFFFFFFF;
    }
    
    /* User name styling */
    .user-name {
        color: #0a2e5c;
        margin-right: 5px;
        font-weight: 500;
        margin-bottom: 2px;
    }
     /* User name styling */
    .input {
        background-color: #FFFFFFFF;
        
    }
    
    /* Input styling */
    .stTextInput input {
        border-radius: 20px;
        border: 1px solid #e0e0e0;
        padding: 10px 15px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        background-color: #0046c7;
        color: white;
    }
    
    /* Message group container to ensure header and messages are connected */
    .message-group {
        display: flex;
        flex-direction: column;
        margin-bottom: 0;
    }
    
    /* Force chat input area background to white */
    .stChatInput, .stChatInput > div, .stChatInput > div > div {
        background-color: #FFFFFF !important;
    }
    
    /* Style the chat input container */
    .stChatInputContainer {
        background-color: #FFFFFF !important;
        padding: 15px;
        border-radius: 0 0 15px 15px;
        margin-top: 0;
    }

    </style>
""", unsafe_allow_html=True)

# Databricks endpoint configuration
DATABRICKS_ENDPOINT = "https://<DATABRICKS URL>/serving-endpoints/agents_dbdemos_mv-care_cost-carecost_compass_agent/invocations"
# Use the provided PAT token
DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN", "<PAT TOKEN>")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add initial system message
    st.session_state.system_message = {"content": "member_id:1234", "role": "system"}
    # Add welcome message
    st.session_state.messages.append({
        "role": "doctor", 
        "content": "Hello! I'm your HealthSpend Bot. I can help you understand your healthcare costs and coverage. What would you like to know about?"
    })

# Function to call Databricks endpoint
def query_databricks_endpoint(messages):
    try:
        headers = {
            "Authorization": f"Bearer {DATABRICKS_TOKEN}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "messages": [st.session_state.system_message] + messages
        }
        
        logger.info(f"Sending request to Databricks: {json.dumps(payload)}")
        
        response = requests.post(DATABRICKS_ENDPOINT, headers=headers, json=payload)
        
        # Log the response status and headers for debugging
        logger.info(f"Response status: {response.status_code}")
        logger.info(f"Response headers: {response.headers}")
        
        if response.status_code != 200:
            logger.error(f"Error response: {response.text}")
            return {"content": f"I'm sorry, I encountered an error (Status code: {response.status_code}). Please try again later."}
        
        # Try to parse the response as JSON
        try:
            result = response.json()
            logger.info(f"Response content: {result}")
            return result
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Response text: {response.text}")
            return {"content": "I'm sorry, I received an invalid response format. Please try again later."}
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error: {str(e)}")
        return {"content": f"I'm sorry, I encountered a connection error: {str(e)}. Please try again later."}
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return {"content": "I'm sorry, I encountered an unexpected error. Please try again later."}

# Chat container
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Create a message group to ensure header and messages are connected
st.markdown('<div class="message-group">', unsafe_allow_html=True)

# Header
st.markdown('''
    <div class="header-container">
        <div style="background-color: #f3faff; padding: 15px; border-radius: 50%; margin-right: 15px;">
            <span style="color: #0046c7; font-size: 24px; font-weight: 600;">+</span>
        </div>
        <div>
            <div style="font-weight: 600; color: #0046c7; font-size: 24px;">💰 HealthSpend Bot</div>
            <div style="background-color: #e6f0ff; width: 120px; height: 10px; border-radius: 5px; margin-top: 5px;"></div>
        </div>
        <div style="margin-left: auto; color: #0046c7; font-size: 24px;">×</div>
    </div>
''', unsafe_allow_html=True)

# Message container
for message in st.session_state.messages:
    if message["role"] == "doctor":
        st.markdown(f'''
            <div class="message-container">
                <div style="display: flex; align-items: center; margin-bottom: 2px;">
                    <div style="color: #e6f0ff; font-size: 24px; margin-right: 5px;">+</div>
                    <div class="bot-name">💰 HealthSpend Bot</div>
                </div>
                <div class="bot-message">{message["content"]}</div>
            </div>
        ''', unsafe_allow_html=True)
    elif message["role"] == "user":
        st.markdown(f'''
            <div class="user-container">
                <div style="display: flex; align-items: center; margin-bottom: 2px;">
                    <div class="user-name">User</div>
                    <div style="color: #0a2e5c; font-size: 18px; margin-left: 5px;">👤</div>
                </div>
                <div class="user-message">{message["content"]}</div>
            </div>
        ''', unsafe_allow_html=True)

# Close message group
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div style="background-color: #FFFFFF !important; padding: 15px; border-radius: 0 0 15px 15px;">', unsafe_allow_html=True)

# Text input for chat
if prompt := st.chat_input("How can I help you with healthcare costs?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    try:
        # Format messages for the API call
        formatted_messages = [{"role": m["role"] if m["role"] != "doctor" else "assistant", 
                              "content": m["content"]} 
                             for m in st.session_state.messages]
        
        # Call Databricks endpoint
        response = query_databricks_endpoint(formatted_messages)
        
        # Add bot response to chat history
        st.session_state.messages.append({"role": "doctor", "content": response["content"]})
        
        st.rerun()
    except Exception as e:
        logger.error(f"Error in main flow: {str(e)}")
        st.error(f"An error occurred: {str(e)}")

st.markdown('</div>', unsafe_allow_html=True)

# Add a sidebar for configuration (hidden in production)
with st.sidebar:
    st.title("Configuration")
    member_id = st.text_input("Member ID", "1234")
    if st.button("Update Member ID"):
        st.session_state.system_message = {"content": f"member_id:{member_id}", "role": "system"}
        st.success(f"Member ID updated to {member_id}")
    
    # Add a debug section
    st.divider()
    st.subheader("Debug Information")
    if st.button("Test Connection"):
        test_message = [{"role": "user", "content": "Test connection"}]
        test_response = query_databricks_endpoint(test_message)
        st.write("Response:", test_response)
    
    st.divider()
    st.caption("For development only - will be removed in production")  