# 🧠 RAG Chatbot — Free LLM Document Q&A

A fully free Retrieval-Augmented Generation (RAG) chatbot that lets you chat with your PDF or text documents. Built with Groq (free LLaMA 3), FAISS, and Streamlit.

## ⚡ Stack
| Layer | Tool |
|---|---|
| LLM | LLaMA 3.3 70B via **Groq** (free) |
| Embeddings | `all-MiniLM-L6-v2` via HuggingFace (local, free) |
| Vector Store | **FAISS** (local) |
| UI | **Streamlit** |

## 🚀 Setup & Run

### 1. Clone / download this project

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Get your FREE Groq API key
- Go to [console.groq.com](https://console.groq.com)
- Sign up and create an API key (completely free)

### 4. Run the app
```bash
streamlit run app.py
```

### 5. Use it
1. Paste your Groq API key in the sidebar
2. Upload a PDF or .txt file
3. Click **Process Document**
4. Start asking questions!

## 🗂 Project Structure
```
rag_chatbot/
├── app.py              # Main Streamlit app
├── requirements.txt    # Dependencies
└── README.md
```

## 🧠 How It Works
1. **Ingestion** — PDF/text is loaded and split into 500-token chunks
2. **Embedding** — Each chunk is converted to a vector using MiniLM
3. **Storage** — Vectors stored in FAISS (in-memory)
4. **Retrieval** — On query, top 4 most relevant chunks are fetched
5. **Generation** — Context + question sent to LLaMA 3 via Groq API
6. **Response** — Answer displayed with source page references

## 📦 Deploy for Free
- [Streamlit Cloud](https://streamlit.io/cloud) — connect your GitHub repo and deploy in 1 click
- Add your Groq API key as a secret in Streamlit Cloud settings

## 💡 Resume Talking Points
- Built end-to-end RAG pipeline from scratch
- Used vector similarity search (FAISS) for semantic retrieval
- Integrated open-source LLM (LLaMA 3) via Groq API
- Deployed as interactive web app on Streamlit Cloud
- Achieved sub-2s response time using Groq's inference engine
