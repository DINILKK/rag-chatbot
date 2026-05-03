import streamlit as st
import os
import tempfile
import numpy as np
import json
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
.metric-card { background: #0f0f20; border: 1px solid #1e1e35; border-radius: 10px; padding: 16px; text-align: center; }
.metric-value { font-family: 'Space Mono', monospace; font-size: 1.8rem; font-weight: 700; color: #7c6af5; }
.metric-label { font-size: 0.7rem; color: #555570; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }
.metric-good  { color: #2ecc71 !important; }
.metric-mid   { color: #f39c12 !important; }
.metric-bad   { color: #e74c3c !important; }
hr { border-color: #1e1e2e !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for key, val in [
    ("messages", []),
    ("chunks", None),
    ("embeddings", None),
    ("doc_name", None),
    ("eval_log", []),        # stores {question, answer, contexts} per query
    ("ragas_results", None), # stores last RAGAS evaluation results
]:
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


def score_color_class(val):
    if val >= 0.8:
        return "metric-good"
    elif val >= 0.5:
        return "metric-mid"
    return "metric-bad"


def run_ragas_evaluation(eval_data, groq_api_key, model_name):
    """
    Runs RAGAS evaluation using Groq as the LLM judge.
    Falls back to lightweight heuristic metrics if RAGAS import fails.
    """
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision
        from ragas.llms import LangchainLLMWrapper
        from langchain_groq import ChatGroq
        from datasets import Dataset

        llm = ChatGroq(api_key=groq_api_key, model_name=model_name, temperature=0)
        wrapped_llm = LangchainLLMWrapper(llm)

        dataset = Dataset.from_dict({
            "question":  [d["question"] for d in eval_data],
            "answer":    [d["answer"]   for d in eval_data],
            "contexts":  [d["contexts"] for d in eval_data],
        })

        results = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision],
            llm=wrapped_llm,
        )

        return {
            "faithfulness":       round(float(results["faithfulness"]), 3),
            "answer_relevancy":   round(float(results["answer_relevancy"]), 3),
            "context_precision":  round(float(results["context_precision"]), 3),
            "method": "ragas",
            "n_samples": len(eval_data),
        }

    except ImportError:
        # ── Heuristic fallback (no extra deps needed) ──────────────────────────
        import re

        def token_overlap(a, b):
            ta = set(re.findall(r'\w+', a.lower()))
            tb = set(re.findall(r'\w+', b.lower()))
            return len(ta & tb) / (len(ta | tb) + 1e-9)

        faithfulness_scores = []
        relevancy_scores    = []
        precision_scores    = []

        for d in eval_data:
            ctx_blob = " ".join(d["contexts"])
            # faithfulness: how much of answer is grounded in context
            faithfulness_scores.append(token_overlap(d["answer"], ctx_blob))
            # answer relevancy: how much answer overlaps with question intent
            relevancy_scores.append(token_overlap(d["answer"], d["question"]))
            # context precision: how much top context overlaps with question
            precision_scores.append(token_overlap(d["contexts"][0] if d["contexts"] else "", d["question"]))

        return {
            "faithfulness":      round(np.mean(faithfulness_scores), 3),
            "answer_relevancy":  round(np.mean(relevancy_scores), 3),
            "context_precision": round(np.mean(precision_scores), 3),
            "method": "heuristic (install ragas for LLM-based eval)",
            "n_samples": len(eval_data),
        }


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
                    st.session_state.eval_log   = []   # reset eval log for new doc
                    st.session_state.ragas_results = None

                    os.unlink(tmp_path)
                    st.success(f"✅ {len(chunks)} chunks embedded!")

                except Exception as e:
                    st.error(f"Error: {str(e)}")

    st.markdown("---")

    if st.session_state.doc_name:
        st.markdown(f'<span class="status-badge status-ready">● Ready — {st.session_state.doc_name}</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge status-waiting">○ No document loaded</span>', unsafe_allow_html=True)

    if st.session_state.eval_log:
        st.markdown(f'<div style="font-size:0.7rem;color:#2ecc71;font-family:monospace;margin-top:8px;">📊 {len(st.session_state.eval_log)} queries logged for eval</div>', unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []
        st.session_state.eval_log = []
        st.session_state.ragas_results = None
        st.rerun()

    st.markdown('<div style="font-size:0.7rem;color:#333355;font-family:monospace;margin-top:1rem;">Stack: Groq + LLaMA3<br>Embeddings: MiniLM-L6-v2<br>Reranking: CrossEncoder<br>Eval: RAGAS</div>', unsafe_allow_html=True)


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_chat, tab_eval = st.tabs(["💬 Chat", "📊 RAGAS Evaluation"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CHAT
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:
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
                        candidates = retrieve(query, k=20)
                        docs       = rerank(query, candidates, k=5)

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

                        answer = response.choices[0].message.content

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": src_info
                        })

                        # ── Log for RAGAS evaluation ───────────────────────
                        st.session_state.eval_log.append({
                            "question": query,
                            "answer":   answer,
                            "contexts": [d["text"] for d in docs],
                        })

                        st.rerun()

                    except Exception as e:
                        st.error(f"Error: {str(e)}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — RAGAS EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
with tab_eval:
    _, col_e, _ = st.columns([1, 6, 1])

    with col_e:
        st.markdown("## 📊 RAG Evaluation Dashboard")
        st.markdown(
            "Evaluates your RAG pipeline using **RAGAS metrics** across all queries made in this session. "
            "Chat with your document first, then run evaluation here."
        )
        st.markdown("---")

        n_logged = len(st.session_state.eval_log)

        if n_logged == 0:
            st.info("No queries logged yet. Ask questions in the Chat tab first, then come back here.")
        else:
            st.markdown(f"**{n_logged} queries ready for evaluation.**")

            col_run, col_export = st.columns([2, 1])
            with col_run:
                run_eval = st.button(f"▶ Run RAGAS Evaluation ({n_logged} samples)", use_container_width=True)
            with col_export:
                if st.session_state.ragas_results:
                    export_data = json.dumps({
                        "results": st.session_state.ragas_results,
                        "eval_log": st.session_state.eval_log,
                    }, indent=2)
                    st.download_button(
                        "⬇ Export JSON",
                        data=export_data,
                        file_name="ragas_evaluation.json",
                        mime="application/json",
                        use_container_width=True,
                    )

            if run_eval:
                if not groq_api_key:
                    st.error("Add your Groq API key in the sidebar first.")
                else:
                    with st.spinner("Running RAGAS evaluation..."):
                        results = run_ragas_evaluation(
                            st.session_state.eval_log,
                            groq_api_key,
                            model,
                        )
                        st.session_state.ragas_results = results
                    st.rerun()

            # ── Display results ────────────────────────────────────────────
            if st.session_state.ragas_results:
                r = st.session_state.ragas_results
                st.markdown("---")
                st.markdown("### Results")

                m1, m2, m3 = st.columns(3)
                metrics = [
                    (m1, "Faithfulness",      r["faithfulness"],      "Answer grounded in context?"),
                    (m2, "Answer Relevancy",  r["answer_relevancy"],  "Answer relevant to question?"),
                    (m3, "Context Precision", r["context_precision"], "Retrieved chunks useful?"),
                ]
                for col, label, val, desc in metrics:
                    cc = score_color_class(val)
                    with col:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value {cc}">{val:.2f}</div>
                            <div class="metric-label">{label}</div>
                            <div style="font-size:0.65rem;color:#333355;margin-top:6px;">{desc}</div>
                        </div>
                        """, unsafe_allow_html=True)

                st.markdown("---")

                # Score interpretation
                avg = np.mean([r["faithfulness"], r["answer_relevancy"], r["context_precision"]])
                if avg >= 0.8:
                    verdict = "🟢 **Excellent** — Production-ready RAG pipeline."
                elif avg >= 0.6:
                    verdict = "🟡 **Good** — Minor improvements possible in retrieval or prompting."
                else:
                    verdict = "🔴 **Needs Work** — Consider better chunking, reranking threshold, or prompt tuning."

                st.markdown(f"**Overall Score: {avg:.2f}** — {verdict}")
                st.markdown(f"*Evaluation method: {r['method']} · {r['n_samples']} samples*")

                st.markdown("---")
                st.markdown("### Per-Query Log")

                for i, entry in enumerate(st.session_state.eval_log):
                    with st.expander(f"Query {i+1}: {entry['question'][:80]}..."):
                        st.markdown(f"**Question:** {entry['question']}")
                        st.markdown(f"**Answer:** {entry['answer']}")
                        st.markdown(f"**Retrieved contexts ({len(entry['contexts'])}):**")
                        for j, ctx in enumerate(entry["contexts"]):
                            st.markdown(f"*Context {j+1}:* {ctx[:300]}...")