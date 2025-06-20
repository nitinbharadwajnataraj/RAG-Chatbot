
# 🧠 RAG Chatbot with Streamlit and Ollama

This project is a **Retrieval-Augmented Generation (RAG) chatbot** that allows users to upload PDF documents and interact with their contents through natural language queries. Built with **Streamlit**, **LangChain**, and **Ollama**, this chatbot performs real-time document indexing, semantic search, and conversational question answering.

---

## 📌 Features

- 🔍 **PDF Upload & Parsing**: Users can upload a PDF file, which is automatically parsed and indexed.
- 🧠 **Semantic Chunking**: Documents are split into overlapping text chunks for better retrieval performance.
- 📚 **Vector Search**: Uses `Chroma` as a vector store backed by `sentence-transformers/all-MiniLM-L6-v2`.
- 💬 **Conversational Memory**: Maintains context with `ConversationBufferMemory` for multi-turn interactions.
- 🤖 **Local LLM**: Powered by `ChatOllama` using the lightweight `gemma3:1b` model.
- 📄 **Source Reference**: Displays the source page numbers from which answers are derived.

---

## 🛠️ Tech Stack

| Component             | Tool/Library                                  |
|----------------------|-----------------------------------------------|
| Web UI               | [Streamlit](https://streamlit.io)             |
| LLM                  | [Ollama](https://ollama.com) (`gemma3:1b`)    |
| Embeddings           | [HuggingFace Transformers](https://huggingface.co) |
| Vector Store         | [Chroma DB](https://www.trychroma.com)        |
| PDF Loader           | `PyPDFLoader` via `langchain_community`       |
| RAG Engine           | [LangChain](https://www.langchain.com)        |

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/nitinbharadwajnataraj/RAG-Chatbot.git
cd RAG-Chatbot
```

### 2. Install Dependencies

Make sure you have Python 3.9+ and Ollama installed.

```bash
pip install -r requirements.txt
```

> You also need to have `ollama` running with the `gemma3:1b` model downloaded:
```bash
ollama run gemma3:1b
```

### 3. Run the App

```bash
streamlit run app.py
```

---

## 🧪 How It Works

1. **PDF Upload**: The user uploads a PDF file via Streamlit.
2. **Hashing**: A unique hash of the file is generated to identify and reuse existing vector indices.
3. **Chunking & Embedding**: The PDF is split into overlapping text chunks and embedded using MiniLM.
4. **Vector Storage**: Chunks are stored in a persistent Chroma vector database.
5. **Retrieval & QA**: On query input, the top matching chunks are retrieved and passed to the LLM for answer generation.
6. **Contextual Dialogue**: Conversation history is maintained across turns for a coherent chat experience.

---

## 📁 Folder Structure

```
rag-chatbot/
│
├── app.py                  # Main Streamlit application
├── chroma_db/              # Folder to store Chroma persistent DB
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## ✅ Example Use Case

1. Upload a PDF (e.g., technical manual, research paper).
2. Ask questions like:  
   - *"What is the main conclusion?"*  
   - *"Summarize the content on page 3."*
3. Get contextual answers with page-level source references.

---

## 🧹 Cleanup

Temporary files like `temp.pdf` are automatically deleted after processing.

---

## 📌 Notes

- Ensure Ollama is running in the background before starting the Streamlit app.
- To reset memory or clear chat history, simply restart the Streamlit app.
