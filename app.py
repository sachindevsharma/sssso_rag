import os
import uuid
import yaml
import streamlit as st
# from yaml.loader import SafeLoader
# import streamlit_authenticator as stauth
from src import pRAGma
from config import Config
from dotenv import load_dotenv

load_dotenv()

# os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

CONFIG = Config()
CHATBOT = pRAGma(CONFIG)

st.set_page_config(
    page_title="pRAGma",
    layout="wide"
)

def init_state():
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []

if len(st.session_state.items()) == 0:
    init_state()

# with open("auth.yaml") as file:
#     auth = yaml.load(file, Loader=SafeLoader)

# authenticator = stauth.Authenticate(auth["credentials"])
# name, authentication_status, username = authenticator.login()
# print(name, authentication_status, username)


def UI():

    st.title("pRAGma 💬")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        print("prompt", prompt)
        with st.spinner("Thinking..."):
            output = CHATBOT.invoke(prompt)
            
        with st.chat_message("assistant"):
            st.markdown(output)
        st.session_state.messages.append({"role": "assistant", "content": output})

UI()