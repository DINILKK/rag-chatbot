# 🧠 RAG-Based PDF Document Chatbot

> Upload any PDF. Ask questions in natural language. Get precise, grounded answers — with measurable evaluation.

## 🔴 Live Demo
[🚀 Try it live](https://local-chatbot.streamlit.app/)

## What It Does

An end-to-end Retrieval-Augmented Generation (RAG) system that enables natural language Q&A over uploaded PDF documents. Built with a full retrieval pipeline including embedding, cosine similarity search, and cross-encoder reranking — evaluated using the RAGAS framework.

---

## Architecture

```
PDF Upload
    │
    ▼
Document Chunking (300 words, 50-word overlap)
    │
    ▼
Sentence Embeddings (MiniLM-L6-v2, 384-dim)
    │
    ▼
Cosine Similarity Retrieval (top-20 candidates)
    │
    ▼
Cross-Encoder Reranking (ms-marco-MiniLM-L-6-v2, top-5)
    │
    ▼
LLM Generation (LLaMA 3 via Groq API)
    │
    ▼
Answer + Source Pages
    │
    ▼
RAGAS Evaluation (Faithfulness · Answer Relevancy · Context Precision)
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (384-dim) |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | LLaMA 3.3 70B via Groq API |
| Retrieval | Cosine similarity (sklearn) |
| Evaluation | RAGAS (faithfulness, answer relevancy, context precision) |
| UI | Streamlit |

---

## Evaluation (RAGAS)

The app includes a built-in **RAGAS Evaluation Dashboard** that scores the pipeline across all queries in a session.

| Metric | Description |
|---|---|
| **Faithfulness** | Is the answer grounded in the retrieved context? (hallucination detection) |
| **Answer Relevancy** | Does the answer address the question asked? |
| **Context Precision** | Are the retrieved chunks relevant to the question? |

Results are exportable as JSON for reproducibility.

---

## Setup

```bash
git clone https://github.com/DINILKK/rag-pdf-chatbot
cd rag-pdf-chatbot
python -m venv venv && venv\Scripts\activate  # Windows
pip install -r requirements.txt
streamlit run main.py
```

### Requirements

```
streamlit
groq
pypdf
sentence-transformers
scikit-learn
numpy
ragas
langchain-groq
datasets
```

You'll need a free **Groq API key** from [console.groq.com](https://console.groq.com).

---

## Key Design Decisions

**Why cross-encoder reranking?**
Bi-encoder embeddings (used for initial retrieval) optimize for speed over precision. Cross-encoders do full attention over the query-chunk pair, significantly improving retrieval quality at the cost of latency. We retrieve 20 candidates cheaply, then rerank to top 5 precisely.

**Why RAGAS evaluation?**
Most RAG demos have no way to measure correctness. RAGAS provides reference-free evaluation — it judges faithfulness and relevancy using an LLM judge, requiring no human-annotated ground truth.

---

## Project Structure

```
├── main.py              # Full app: chat + evaluation tabs
├── requirements.txt
└── README.md
```

---

## Author

**Dinil K K** — [github.com/DINILKK](https://github.com/DINILKK) · [dinilkk007@gmail.com](mailto:dinilkk007@gmail.com)