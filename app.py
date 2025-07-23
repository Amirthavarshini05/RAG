import streamlit as st
from rag_pipeline import get_response

st.title("Offline RAG Chatbot")
query = st.text_input("Enter your question")
if st.button("Ask"):
    answer = get_response(query)
    st.write(answer)
