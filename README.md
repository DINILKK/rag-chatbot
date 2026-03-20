# 🧠 RAG Document Chatbot

A fully free Retrieval-Augmented Generation (RAG) chatbot that lets you 
chat with your PDF documents using natural language.

Built with LLaMA 3, semantic embeddings, and Streamlit.

## 🔴 Live Demo
[Add your Streamlit Cloud link here]

## 🧠 How It Works

1. Upload a PDF document
2. Document is split into overlapping chunks of 300 words
3. Each chunk is converted into a 384-dimensional vector using sentence transformers
4. When you ask a question, it gets embedded using the same model
5. Cosine similarity finds the 8 most relevant chunks
6. Retrieved chunks + your question are sent to LLaMA 3 via Groq API
7. LLaMA generates an answer grounded in your document

## ⚡ Stack

| Layer | Tool |
|---|---|
| LLM | LLaMA 3.3 70B via Groq (free) |
| Embeddings | all-MiniLM-L6-v2 via sentence-transformers |
| Retrieval | Cosine similarity with numpy |
| UI | Streamlit |

## 🚀 Run Locally

**1. Clone the repo**
git clone https://github.com/DINILKK/rag-chatbot.git
cd rag-chatbot

**2. Create virtual environment**
python -m venv venv
venv\Scripts\activate

**3. Install dependencies**
pip install -r requirements.txt

**4. Get free Groq API key**
Go to console.groq.com and create a free account

**5. Run**
python -m streamlit run app.py

## 📁 Project Structure

rag-chatbot/
├── app.py           # Streamlit web app
├── pipeline.py      # Core RAG pipeline (built from scratch)
├── requirements.txt
└── README.md

## 💡 Key Concepts

**Chunking** — Documents are split into 300 word overlapping chunks
so embeddings capture focused meaning rather than mixed context

**Semantic Search** — Unlike keyword search, embeddings understand
meaning. "courage" matches "bravery" even without the exact word

**RAG vs Fine-tuning** — RAG is better for custom documents because
it doesn't require retraining. Just index and query.

## 🔧 Potential Improvements
- Reranking with cross-encoder for better retrieval accuracy
- Hybrid search combining TF-IDF and semantic search
- Persistent vector storage with FAISS or ChromaDB
- Multi-document support
- Chat history memory

## 👤 Author
Dinil KK
github.com/DINILKK