import streamlit as st
from langchain_helper import get_qa_chain, create_vectordb

st.title("Product QnA")
btn = st.button("Create Knowledgebase")
if btn:
    create_vectordb()

question = st.text_input("Question: ")

if question:
    chain = get_qa_chain()
    response = chain(question)

    st.header("Answer")
    st.write(response["result"])