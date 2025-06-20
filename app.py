import streamlit as st
import os
import hashlib
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatOllama

st.set_page_config(page_title="RAG Chatbot")
st.title("RAG Chatbot")

def get_file_hash(file):
    content = file.read()
    file.seek(0)
    return hashlib.md5(content).hexdigest()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer" 
    )

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])
user_query = st.chat_input("Ask a question...")

if uploaded_file:
    file_hash = get_file_hash(uploaded_file)
    persist_directory = f"chroma_db/{file_hash}"

    if not os.path.exists(persist_directory):
        with st.spinner("Indexing PDF..."):
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.read())

            loader = PyPDFLoader("temp.pdf")
            documents = loader.load()

            for doc in documents:
                if "page" not in doc.metadata:
                    doc.metadata["page"] = doc.metadata.get("page_number", 0)

            splitter = CharacterTextSplitter(chunk_size=250, chunk_overlap=100)
            docs = splitter.split_documents(documents)

            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectordb = Chroma.from_documents(
                docs,
                embedding=embeddings,
                persist_directory=persist_directory,
                collection_name="ragpdf"
            )
            vectordb.persist()
            st.success("Document indexed and stored in Chroma.")
    else:
        st.info("Using existing Chroma index.")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name="ragpdf"
    )

    st.session_state.retriever = vectordb.as_retriever()
    llm = ChatOllama(model="gemma3:1b")  

    st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=st.session_state.retriever,
        memory=st.session_state.memory,
        return_source_documents=True,
        output_key="answer" 
    )


if user_query and st.session_state.qa_chain:
    with st.spinner("Generating answer..."):
        result = st.session_state.qa_chain({"question": user_query})
        answer = result["answer"]
        sources = result.get("source_documents", [])

        page_numbers = sorted({doc.metadata.get("page", "N/A") for doc in sources})
        pages_text = ", ".join(str(p) for p in page_numbers)

        
        st.session_state.chat_history.append(("You", user_query))
        st.session_state.chat_history.append(("Gemma", f"{answer}\n\n **Sources:** Page(s) {pages_text}"))

for speaker, msg in st.session_state.chat_history:
    with st.chat_message(speaker):
        st.markdown(msg)

if os.path.exists("temp.pdf"):
    os.remove("temp.pdf")
