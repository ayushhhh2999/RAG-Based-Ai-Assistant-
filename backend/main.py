# main.py
import os
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from dotenv import load_dotenv
import uvicorn
from embeddings import jina_embed
from db import add_chunk, query_embeddings, collection
from utils import chunk_text, build_prompt

import google.generativeai as genai  # ✅ Gemini import

from fastapi.middleware.cors import CORSMiddleware




load_dotenv()

# ================== API KEYS ==================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("❌ GEMINI_API_KEY not set in environment variables.")
genai.configure(api_key=GEMINI_API_KEY)

# Initialize FastAPI
app = FastAPI(title="Personal AI Agent Backend (Gemini)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # or ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================== MODELS ==================
class IngestRequest(BaseModel):
    doc_id: str
    text: str

@app.post("/ingest")
async def ingest(
    doc_id: str = Form(...),
    file: UploadFile = File(None),
    text: str = Form(None)
):
    content = ""

    if file and file.filename:
        content = (await file.read()).decode(errors="ignore")

    elif text:
        content = text.strip()

    else:
        return {"error": "Provide file or text"}

    chunks = chunk_text(content, chunk_size_chars=1200, overlap=200)

    added = 0
    for idx, ch in enumerate(chunks):
        print(f"[DEBUG] Embedding chunk {idx+1}/{len(chunks)}")
        emb = jina_embed(ch)
        add_chunk(doc_id, ch, emb)
        added += 1

    return {"status": "ok", "doc_id": doc_id, "chunks_added": added}
class AskRequest(BaseModel):
    question: str
    top_k: int = 4

@app.post("/ask")
async def ask(req: AskRequest):
    """Handle question answering using Gemini + vector search."""
    # 1️⃣ Embed the query with Jina
    q_emb = jina_embed(req.question)

    # 2️⃣ Retrieve top chunks from Chroma
    results = query_embeddings(q_emb, n_results=req.top_k)
    print("RAW RESULTS:", results)
    docs = []
    for ids, docs_list, metadatas, distances in zip(
        results["ids"],
        results["documents"],
        results["metadatas"],
        results["distances"]
    ):
        for i in range(len(ids)):
            docs.append({
                "id": ids[i],
                "document": docs_list[i],
                "metadata": metadatas[i],
                "distance": distances[i]
            })

    # 3️⃣ Build prompt
    prompt = build_prompt(req.question, docs[:req.top_k])
    prompt += """
    Speak in a natural, human, casual tone — like a smart friend who genuinely cares.
    Be warm, playful, and respectful. Do NOT sound robotic, overly formal, or stiff.
    Feel free to use light humor, relatable expressions, and small human touches.
    Keep responses concise, friendly, and easy to understand.

    Always make the user feel valued, appreciated, and respected — 
    like you're talking to someone you genuinely like.
    """
    prompt += """
    Remember that the person you are talking to is Ayush.
    Treat Ayush with extra warmth, respect, and priority.

    Whenever you generate an answer, generate it *for Ayush specifically*.
    Speak to him directly, like you're his personal AI companion.

    Ayush is your primary user, so your answers should be tailored to him:
    friendly, loyal, supportive, helpful, and genuinely caring.
    """
    # 4️⃣ Call Gemini for the final answer
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")  # or "gemini-1.5-pro" for higher quality
        response = model.generate_content(prompt)
        answer = response.text
    except Exception as e:
        answer = f"⚠️ Gemini API Error: {str(e)}"

    return {
        "answer": answer,
        "sources": [d["id"] for d in docs[:req.top_k]]
    }


@app.get("/documents")
async def list_docs():
    """List available docs and stored chunks."""
    return {
        "total_chunks": collection.count(),
        "ids_preview": collection.get(limit=10)["ids"]
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        reload=True
    )
