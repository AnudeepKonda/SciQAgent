import os
import streamlit as st
from rag_agent import SciQAgent
from __init__ import setup_logger

# Initialize logger
setup_logger()
# Set page title and icon
st.set_page_config(page_title="Scientific QA Agent", page_icon="", layout="centered")

# Custom styles
st.markdown(
    """
    <style>
        .title {
            text-align: center;
            font-size: 2rem;
            font-weight: bold;
            color: #4CAF50;
        }
        .subtitle {
            text-align: center;
            font-size: 1.2rem;
            color: #777;
        }
        .chat-container {
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 10px;
        }
        .user-message {
            background-color: #e1f5fe;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 5px;
        }
        .bot-message {
            background-color: #c8e6c9;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 5px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Title and subtitle
st.markdown('<div class="title">Scientific QA Agent</div>', unsafe_allow_html=True)
st.markdown("---")

# Ensure API key is set only once
if "api_key_set" not in st.session_state:
    st.session_state.api_key_set = False

if not st.session_state.api_key_set:
    openai_api_key = st.text_input("🔑 Enter your OpenAI API Key:", type="password")
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
        st.session_state.api_key_set = True
        st.success("✅ API Key set successfully!")
        st.rerun()
    else:
        st.error("⚠️ Please enter a valid API Key.")
else:
    # Persist SciQAgent instance
    if "rag_agent" not in st.session_state:
        st.session_state.rag_agent = SciQAgent()

    # Initialize conversation state if not already set
    if "rag_state" not in st.session_state:
        st.session_state.rag_state = {
            "messages": [],
            "query": "",
            "retrieved_context": "",
            "feedback": "",
            "generated_answer": "",
        }

    # Chat input
    st.markdown("### Ask a question:")
    user_input = st.text_input("", placeholder="Ask and press Enter...", key="user_input")

    # Send message when user presses Enter
    if user_input:
        if user_input.strip().lower() == "exit":
            st.markdown("✅ **Conversation ended. Refresh the page to start a new session.**")
            st.stop()

        # Update state with new query
        st.session_state.rag_state["query"] = user_input
        st.session_state.rag_state["retrieved_context"] = ""
        st.session_state.rag_state["feedback"] = ""
        st.session_state.rag_state["generated_answer"] = ""
        st.session_state.rag_state["refinement_count"] = 0

        st.session_state.rag_state["messages"].append({"role": "user", "content": user_input})

        # Invoke the persisted RAG Agent
        response = st.session_state.rag_agent.invoke(st.session_state.rag_state)
        st.session_state.rag_state["messages"].append({"role": "assistant", "content": response["generated_answer"]})

    # Display chat history
    st.markdown("### 📝 Conversation History")
    with st.container():
        for message in st.session_state.rag_state["messages"]:
            if message["role"] == "user":
                st.markdown(f'<div class="chat-container user-message"><strong>🧑‍💻 You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-container bot-message"><strong>🤖 Agent:</strong> {message["content"]}</div>', unsafe_allow_html=True)
