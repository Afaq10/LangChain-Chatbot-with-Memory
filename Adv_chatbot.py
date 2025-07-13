from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

st.set_page_config(page_title="DeepSeek Chatbot", layout="centered")

with st.sidebar:
    # Streamlit config
    st.title("👻 Chatbot")

    # Temperature control
    temperature = st.slider("🔧 Set Model Temperature", 0.0, 1.0, 0.6)

    # Clear Chat Button
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = [("system", "You are a helpful AI assistant. Please respond to the user's queries clearly and concisely.")]
        st.rerun()



# Apply basic styling
st.markdown("""
    <style>
    .reportview-container {
        background-color: #fff;
    }
    .stTextInput>div>div>input {
        font-size: 18px;
    }
     .user-bubble {
        background-color: #DCF8C6;
        padding: 10px 15px;
        border-radius: 15px;
        margin: 5px 0;
        max-width: 80%;
        align-self: flex-end;
        color: black;
    }
    .bot-bubble {
        background-color: #F1F0F0;
        padding: 10px 15px;
        border-radius: 15px;
        margin: 5px 0;
        max-width: 80%;
        align-self: flex-start;
        color: black;
    }
    .chat-container {
        display: flex;
        flex-direction: column;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [("system", "You are a helpful AI assistant. Please respond to the user's queries clearly and concisely.")]


# Model and parser setup
llm = Ollama(model="deepseek-r1:1.5b", temperature=temperature)
output_parser = StrOutputParser()


# Chat history display
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

for role, message in st.session_state.chat_history:
    if role == "user":
        st.markdown(f'<div class="user-bubble">🧑 {message}</div>', unsafe_allow_html=True)
    elif role == "assistant":
        st.markdown(f'<div class="bot-bubble">🤖 {message}</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# 👉 Input field shown **after** chat history
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("💬 How may I assist you today?", key="user_input")
    submitted = st.form_submit_button("Send")

# If message submitted
if submitted and user_input:
    st.session_state.chat_history.append(("user", user_input))
    prompt = ChatPromptTemplate.from_messages(st.session_state.chat_history)
    chain = prompt | llm | output_parser

    with st.spinner("🤔 Thinking..."):
        try:
            response = chain.invoke({})
            st.session_state.chat_history.append(("assistant", response))
            st.rerun()  # Refresh to show response and move input down
        except Exception as e:
            st.session_state.chat_history.append(("assistant", f"Error: {e}"))
            st.rerun()
