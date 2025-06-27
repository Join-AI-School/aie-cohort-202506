"""
uv add streamlit
streamlit run streamlit_app/ui.py
"""

import streamlit as st
import requests

# API_URL = "http://localhost:8000/chat"  # Change this to your public endpoint if deployed
API_URL = "http://host.docker.internal:8000/chat"
# API_URL = "http://34.133.6.80/chat"

st.set_page_config(page_title="FIA", page_icon="🤖")
st.title("Financial Insight Advisor (FIA)")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "ai", "content": "Hi, how can I help you today?"}]

# Show messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Type your message here..."):
    # Add user message to history
    st.session_state.messages.append({"role": "human", "content": prompt})
    with st.chat_message("human"):
        st.markdown(prompt)

    # Send to FastAPI
    payload = {
        "messages": st.session_state.messages,
        "user_id": "user_123",
        "thread_id": "conversation_1"
    }

    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        reply = response.json()["response"]
    except Exception as e:
        reply = f"⚠️ API error: {e}"

    # Add AI reply to history
    st.session_state.messages.append({"role": "ai", "content": reply})
    with st.chat_message("ai"):
        st.markdown(reply)
