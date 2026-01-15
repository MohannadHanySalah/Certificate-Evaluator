import streamlit as st
import os
import base64
import mimetypes
import json
from langchain_core.messages import HumanMessage, AIMessage
from main import build_graph, AgentState

# Page configuration
st.set_page_config(
    page_title="Certificate Evaluation Agent",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Helper function to encode image/pdf to base64
@st.cache_data
def get_base64_encoded_file(content):
    return base64.b64encode(content).decode("utf-8")

def get_document_message(file):
    mime_type, _ = mimetypes.guess_type(file.name)
    if not mime_type:
        if file.name.lower().endswith('.pdf'):
            mime_type = "application/pdf"
        else:
            mime_type = "image/jpeg"
    
    encoded = get_base64_encoded_file(file.getvalue())
    
    return HumanMessage(
        content=[
            {"type": "text", "text": f"Here is the certificate file ({file.name}):"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{encoded}"}
            }
        ]
    )

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent_state" not in st.session_state:
    st.session_state.agent_state = {
        "messages": [],
        "extracted_data": {},
        "evaluation_criteria": {},
        "evaluation_result": {},
        "conversation_context": {}
    }

if "app" not in st.session_state:
    st.session_state.app = build_graph()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "active_chat_id" not in st.session_state:
    st.session_state.active_chat_id = None

if "sent_files" not in st.session_state:
    st.session_state.sent_files = []

# Sidebar
with st.sidebar:
    st.markdown("<h2 style='margin-bottom: 2rem; font-family: Outfit, sans-serif;'>Certificate Agent</h2>", unsafe_allow_html=True)
    
    if st.button("+ New chat", use_container_width=True):
        # Only save if it's a NEW chat that has messages
        if st.session_state.messages and st.session_state.active_chat_id is None:
            title = st.session_state.messages[0]["content"][:30] + "..."
            chat_id = len(st.session_state.chat_history)
            st.session_state.chat_history.append({
                "id": chat_id,
                "title": title, 
                "messages": st.session_state.messages, 
                "state": st.session_state.agent_state,
                "sent_files": st.session_state.sent_files
            })
            
        st.session_state.messages = []
        st.session_state.agent_state = {
            "messages": [],
            "extracted_data": {},
            "evaluation_criteria": {},
            "evaluation_result": {},
            "conversation_context": {}
        }
        st.session_state.sent_files = []
        st.session_state.active_chat_id = None
        st.rerun()
    
    st.markdown("### Recent")
    if not st.session_state.chat_history:
        st.caption("No previous chats")
    else:
        # Show in reverse order
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            col_chat, col_del = st.columns([5, 1])
            with col_chat:
                if st.button(chat["title"], key=f"hist_{chat['id']}", use_container_width=True, type="secondary"):
                    st.session_state.messages = chat["messages"]
                    st.session_state.agent_state = chat["state"]
                    st.session_state.sent_files = chat.get("sent_files", [])
                    st.session_state.active_chat_id = chat["id"]
                    st.rerun()
            with col_del:
                if st.button("🗑️", key=f"del_{chat['id']}", help="Delete this chat"):
                    st.session_state.chat_history = [c for c in st.session_state.chat_history if c["id"] != chat["id"]]
                    if st.session_state.active_chat_id == chat["id"]:
                        st.session_state.messages = []
                        st.session_state.active_chat_id = None
                    st.rerun()
    
    st.markdown("---")
    st.markdown("### 📄 Documents")
    uploaded_files = st.file_uploader(
        "Upload Certificates", 
        type=["pdf", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key="file_uploader",
        label_visibility="collapsed"
    )
    
    if uploaded_files:
        st.success(f"{len(uploaded_files)} files uploaded")

    st.markdown("---")

# Main Content

if not st.session_state.messages:
    # Centered Greeting
    st.markdown("<div style='height: 25vh;'></div>", unsafe_allow_html=True)
    st.markdown("""
        <div style='text-align: center;'>
            <h1 style='font-size: 4rem; font-weight: 600; background: linear-gradient(90deg, #4285f4, #91c4f9); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>Hello, User</h1>
            <p style='color: #b4b4b4; font-size: 1.5rem; margin-top: 1rem;'>This is you Certificate Evaulation Agent</p>
            <p style='color: #b4b4b4; font-size: 1.5rem; margin-top: 1rem;'>How can I help you evaluate your certificates today?</p>
        </div>
    """, unsafe_allow_html=True)
else:
    # Chat History Container
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Ask Certificate Agent..."):
    # Add user message to UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun()

# Processing Logic (Triggered after rerun if prompt exists)
# In Streamlit, it's often better to handle the agent call within the prompt block
# but since ST reruns the whole script, we check if the last message is from user.

if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    user_prompt = st.session_state.messages[-1]["content"]
    
    # Prepare agent messages
    current_agent_messages = st.session_state.agent_state["messages"]
    
    # Handle files
    if uploaded_files:
        for file in uploaded_files:
            if file.name not in st.session_state.sent_files:
                current_agent_messages.append(get_document_message(file))
                st.session_state.sent_files.append(file.name)
    
    current_agent_messages.append(HumanMessage(content=user_prompt))
    
    # Show status and invoke
    with st.chat_message("assistant"):
        with st.spinner("Processing..."):
            try:
                st.session_state.agent_state["messages"] = current_agent_messages
                result = st.session_state.app.invoke(st.session_state.agent_state)
                
                last_msg = result['messages'][-1]
                response_text = last_msg.content if isinstance(last_msg.content, str) else \
                                ' '.join([c.get('text', '') for c in last_msg.content if c.get('type') == 'text'])
                
                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                st.session_state.agent_state = result
                st.rerun()
            except Exception as e:
                st.error(f"Error communicating with agent: {e}")
