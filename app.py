import streamlit as st
import os
import tempfile
import numpy as np
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="RAG Chatbot", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.stApp { background: #0a0a0f; color: #e8e8f0; }
[data-testid="stSidebar"] { background: #0f0f1a !important; border-right: 1px solid #1e1e2e; }
[data-testid="collapsedControl"] { display: flex !important; background: #0f0f1a !important; }
section[data-testid="stSidebar"] { display: block !important; min-width: 250px !important; }
.rag-title { font-family: 'Space Mono', monospace; font-size: 1.6rem; font-weight: 700; color: #7c6af5; margin-bottom: 0.2rem; }
.rag-subtitle { font-size: 0.75rem; color: #555570; font-family: 'Space Mono', monospace; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 2rem; }
.status-badge { display: inline-block; padding: 3px 10px; border-radius: 20px; font-size: 0.7rem; font-family: 'Space Mono', monospace; letter-spacing: 1px; text-transform: uppercase; }
.status-ready   { background: #0d2b1f; color: #2ecc71; border: 1px solid #2ecc71; }
.status-waiting { background: #1a1a2e; color: #7c6af5; border: 1px solid #7c6af5; }
.user-msg { background: #13131f; border: 1px solid #1e1e35; border-radius: 12px 12px 4px 12px; padding: 14px 18px; margin: 8px 0; margin-left: 15%; color: #c8c8e8; font-size: 0.9rem; line-height: 1.6; }
.bot-msg  { background: #0f0f20; border-left: 3px solid #7c6af5; border-radius: 4px 12px 12px 12px; padding: 14px 18px; margin: 8px 0; margin-right: 15%; color: #d8d8f0; font-size: 0.9rem; line-height: 1.6; }
.msg-label { font-family: 'Space Mono', monospace; font-size: 0.65rem; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 6px; opacity: 0.5; }
.source-box { background: #0a0a18; border: 1px dashed #2a2a45; border-radius: 8px; padding: 10px 14px; margin-top: 8px; font-size: 0.75rem; color: #5555aa; font-family: 'Space Mono', monospace; }
.stTextInput > div > div > input { background: #13131f !important; border: 1px solid #1e1e35 !important; border-radius: 8px !important; color: #e8e8f0 !important; font-size: 0.9rem !important; padding: 12px 16px !important; }
.stTextInput > div > div > input:focus { border-color: #7c6af5 !important; box-shadow: 0 0 0 2px #7c6af520 !important; }
.stButton > button { background: #7c6af5 !important; color: white !important; border: none !important; border-radius: 8px !important; font-family: 'Space Mono', monospace !important; font-size: 0.75rem !important; padding: 10px 20px !important; }
.stButton > button:hover { background: #6a58e0 !important; }
hr { border-color: #1e1e2e !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for key, val in [("messages", []), ("chunks", None), ("embeddings", None), ("doc_name", None)]:
    if key not in st.session_state:
        st.session_state[key] = val


@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def load_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def load_pdf(path):
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            pages.append({"text": text.strip(), "page": i + 1})
    return pages


def load_txt(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return [{"text": text, "page": 1}]


def chunk_text(pages, chunk_size=300, overlap=50):
    chunks = []
    for p in pages:
        words = p["text"].split()
        i = 0
        while i < len(words):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append({"text": chunk, "page": p["page"]})
            i += chunk_size - overlap
    return chunks


def retrieve(query, k=20):
    model = load_model()
    q_emb  = model.encode([query])
    scores = cosine_similarity(q_emb, st.session_state.embeddings)[0]
    top_k  = np.argsort(scores)[::-1][:k]
    return [st.session_state.chunks[i] for i in top_k]


def rerank(query, chunks, k=5):
    reranker = load_reranker()
    pairs    = [[query, chunk["text"]] for chunk in chunks]
    scores   = reranker.predict(pairs)
    ranked   = sorted(zip(scores, chunks), reverse=True)
    return [chunk for score, chunk in ranked[:k]]


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="rag-title">🧠 RAG Chat</div>', unsafe_allow_html=True)
    st.markdown('<div class="rag-subtitle">Document Intelligence</div>', unsafe_allow_html=True)

    groq_api_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...", help="Get free key at console.groq.com")
    st.markdown("---")

    model = st.selectbox("Model", ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"])
    st.markdown("---")

    st.markdown("**Upload Document**")
    uploaded_file = st.file_uploader("upload", type=["pdf", "txt"], label_visibility="collapsed")

    if uploaded_file and groq_api_key:
        if st.button("⚡ Process Document"):
            with st.spinner("Embedding document — first run downloads model, please wait..."):
                try:
                    is_pdf = uploaded_file.type == "application/pdf"
                    suffix = ".pdf" if is_pdf else ".txt"

                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name

                    pages  = load_pdf(tmp_path) if is_pdf else load_txt(tmp_path)
                    chunks = chunk_text(pages)
                    texts  = [c["text"] for c in chunks]

                    embedder   = load_model()
                    embeddings = embedder.encode(texts, show_progress_bar=False)

                    st.session_state.chunks     = chunks
                    st.session_state.embeddings = embeddings
                    st.session_state.doc_name   = uploaded_file.name

                    os.unlink(tmp_path)
                    st.success(f"✅ {len(chunks)} chunks embedded!")

                except Exception as e:
                    st.error(f"Error: {str(e)}")

    st.markdown("---")

    if st.session_state.doc_name:
        st.markdown(f'<span class="status-badge status-ready">● Ready — {st.session_state.doc_name}</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge status-waiting">○ No document loaded</span>', unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.markdown('<div style="font-size:0.7rem;color:#333355;font-family:monospace;margin-top:1rem;">Stack: Groq + LLaMA3<br>Embeddings: MiniLM-L6-v2<br>Reranking: CrossEncoder</div>', unsafe_allow_html=True)


# ── Main ──────────────────────────────────────────────────────────────────────
_, col2, _ = st.columns([1, 6, 1])

with col2:
    st.markdown("## Ask anything about your document")
    st.markdown("Upload a PDF or text file in the sidebar, then start chatting.")
    st.markdown("---")

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="user-msg"><div class="msg-label">You</div>{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            src      = msg.get("sources", "")
            src_html = f"<div class='source-box'>📎 {src}</div>" if src else ""
            st.markdown(f'<div class="bot-msg"><div class="msg-label">Assistant</div>{msg["content"]}{src_html}</div>', unsafe_allow_html=True)

    st.markdown("---")

    query = st.text_input("Message", placeholder="Ask a question about your document...", label_visibility="collapsed", key="query_input")
    _, col_btn = st.columns([5, 1])
    with col_btn:
        send = st.button("Send →")

    if send and query:
        if not groq_api_key:
            st.error("Please enter your Groq API key in the sidebar.")
        elif not st.session_state.chunks:
            st.error("Please upload and process a document first.")
        else:
            st.session_state.messages.append({"role": "user", "content": query})
            with st.spinner("Thinking..."):
                try:
                    # Retrieve top 20 candidates
                    candidates = retrieve(query, k=20)

                    # Rerank and get top 5
                    docs = rerank(query, candidates, k=5)

                    context  = "\n\n".join([d["text"] for d in docs])
                    pages    = list(set([str(d["page"]) for d in docs]))
                    src_info = f"Page(s): {', '.join(sorted(pages, key=int))}" if pages else ""

                    system_prompt = """You are a precise and helpful document assistant.
Answer questions using ONLY the provided context from the document.
When asked to summarize, provide a well-structured summary with:
- Main theme or argument
- Key points and ideas
- Important details or examples
If the answer is not in the context, say so clearly.
Be thorough but concise."""

                    client   = Groq(api_key=groq_api_key)
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user",   "content": f"Context from document:\n{context}\n\nQuestion: {query}\n\nAnswer:"}
                        ],
                        temperature=0.3,
                        max_tokens=2048
                    )

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response.choices[0].message.content,
                        "sources": src_info
                    })
                    st.rerun()

                except Exception as e:
                    st.error(f"Error: {str(e)}")