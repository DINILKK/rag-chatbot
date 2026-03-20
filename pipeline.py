import numpy as np
import re
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

model = SentenceTransformer("all-MiniLM-L6-v2")

def read_pdf(path):
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text(extraction_mode="layout")
        if not text:
            text = page.extract_text()
        if text and text.strip():
            # Fix missing spaces between words
            import re
            text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
            text = re.sub(r'\s+', ' ', text).strip()
            pages.append({
                "text": text,
                "page": i + 1
            })
    return pages


def chunk_text(pages, chunk_size=300, overlap=50):
    chunks = []
    for page in pages:
        words = page["text"].split()
        i = 0
        while i < len(words):
            chunk_words = words[i:i+chunk_size]
            chunk = " ".join(chunk_words)  # ← space not empty string
            chunks.append({
                "text": chunk,
                "page": page["page"]
            })
            i += chunk_size - overlap

    return chunks  # ← outside the for loop, no indentation




def create_embeddings(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts  = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings



def retrieve(query, chunks, embeddings, k=4):
    model = SentenceTransformer("all-MiniLM-L6-v2")

    query_embedding = model.encode([query])
    scores = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = np.argsort(scores)[::-1][:k]

    results = []
    for i in top_indices:
        results.append(
            {
                "text": chunks[i]["text"],
                "page": chunks[i]["page"],
                "score": round(float(scores[i]), 3)
            }
        )
    return results


def generate_answer(query, retrieved_chunks, api_key):
    context = "\n\n".join([chunk["text"] for chunk in retrieved_chunks])

    system_prompt  = """ you are a helpful document assistant .
    answer questions using only a the provided contextn.
    if the answer is not in the context , say so clearly. 
    be consie and direct """

    user_prompt = f""" context from document : {context}
    question : {query}
    answer : """

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model = "llama-3.3-70b-versatile",
        messages = [
            {"role" : "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ], temperature=0.2,
        max_tokens=1024
    )
    return response.choices[0].message.content