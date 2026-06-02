import streamlit as st

from main import ask_question

st.set_page_config(
    page_title="Agentic RAG",
    page_icon="🤖",
    layout="wide"
)

st.title("📚 Agentic RAG with LangGraph + Pinecone")

question = st.text_area(
    "Ask a question",
    height=120
)

if st.button("Submit"):

    if question.strip():

        with st.spinner("Thinking..."):

            answer = ask_question(question)

        st.markdown("### Answer")

        st.write(answer)