import os
import shutil
import tempfile
import time

import streamlit as st

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI

load_dotenv()

st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📚",
    layout="wide"
)

st.title("📚 PDF Question Answering Chatbot")
st.write("Upload a PDF and ask questions about it.")

uploaded_pdf = st.sidebar.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

llm = ChatMistralAI(
    model="mistral-small-latest"
)

prompt = ChatPromptTemplate.from_messages(
[
(
"system",
"""
You are a helpful AI assistant.

Answer ONLY using the given context.

If the answer is not present in the context, reply:

'I could not find the answer in the document.'
"""
),

(
"human",
"""
Context:
{context}

Question:
{question}
"""
)
]
)

if uploaded_pdf:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(uploaded_pdf.read())
        pdf_path = f.name

    with st.spinner("Reading PDF..."):

        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(docs)

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory="chroma_db"
        )

        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k":4,
                "fetch_k":10,
                "lambda_mult":0.5
            }
        )

    st.success("PDF processed successfully!")

    question = st.chat_input("Ask a question...")

    if question:

        with st.chat_message("user"):
            st.write(question)

        start = time.time()

        docs = retriever.invoke(question)

        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        final_prompt = prompt.invoke(
            {
                "context":context,
                "question":question
            }
        )

        with st.spinner("Generating answer..."):
            response = llm.invoke(final_prompt)

        end = time.time()

        with st.chat_message("assistant"):
            st.write(response.content)

            st.caption(f"Response Time : {end-start:.2f} sec")

        with st.expander("Retrieved Context"):

            for i, doc in enumerate(docs):

                st.markdown(f"### Chunk {i+1}")

                st.write(doc.page_content[:1000])