from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace
from dotenv import load_dotenv
import streamlit as st


load_dotenv()

st.header("Research Tool")

user_input = st.text_input("Enter your Prompt")

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm )
summarise = st.button("summarise")

if summarise:
    result = model.invoke(user_input)
    st.write (result.content)