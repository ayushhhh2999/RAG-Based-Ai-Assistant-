# main.py
import os
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from dotenv import load_dotenv
import uvicorn
from embeddings import jina_embed
from db import add_chunk, query_embeddings, collection,clean_messed_up_data,find_corrupted_chunks
from utils import chunk_text, build_prompt, extract_text_from_file
from models import IngestRequest, AskRequest, ChatStoreRequest
import google.generativeai as genai  # ✅ Gemini import

from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
import io


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


from fastapi import UploadFile, File, Form
from pydantic import BaseModel

@app.post("/ingest")
async def ingest(
    doc_id: str = Form(...),
    file: UploadFile = File(None),
    text: str = Form(None),
):
    """
    Ingest file or raw text → extract text → chunk → embed → store in ChromaDB.
    """

    # 1️⃣ Extract content
    if file and file.filename:
        file_bytes = await file.read()
        content = extract_text_from_file(file_bytes, file.filename)

        if not content.strip():
            return {"error": "Unable to extract text from file."}

    elif text:
        content = text.strip()

    else:
        return {"error": "Provide either a file or text"}

    # 2️⃣ Chunk text
    chunks = chunk_text(content, chunk_size_chars=1200, overlap=200)

    # 3️⃣ Store chunks
    added = 0
    for idx, chunk in enumerate(chunks):
        print(f"[DEBUG] Embedding chunk {idx+1}/{len(chunks)}")

        emb = jina_embed(chunk)         # or Ollama embed if switched
        add_chunk(doc_id, chunk, emb)
        added += 1

    return {
        "status": "ok",
        "doc_id": doc_id,
        "chunks_added": added
    }


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


@app.post("/chat")
async def chat_analyze(req: ChatStoreRequest):
    """
    Analyze chat message and decide if it should be stored.
    """

    prompt = """
You are an intelligent classifier that decides if a piece of text should be stored
in a long-term personal memory knowledge base.

User message:
"{chat}"

Your job:
1. Decide if this text is valuable to store.
2. If it's just greetings (hi, hello, ok), or random chat, mark flag = false.
3. If the text contains useful personal info, knowledge, preferences,
   project notes, plans, tasks, or anything that can be helpful later,
   mark flag = true.
4. If flag=true:
    - Extract the core important information only.
    - Create a short title summarizing the memory.
5. Respond ONLY in JSON with keys: flag, title, information.

Example Response:
{
  "flag": true,
  "title": "Rust Project Idea",
  "information": "Ayush wants to build a hate speech detection system in Rust."
}

Now analyze the chat and respond:
""".replace("{chat}", req.chat)
    # <-- This prevents f-string issues

    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        llm_json = response.text
    except Exception as e:
        return {"error": f"Gemini Error: {str(e)}"}
    
    def clean_llm_json(text: str) -> str:
        text = text.strip()
    # Remove triple backticks and language tags
        if text.startswith("```"):
            text = text.replace("```json", "").replace("```", "").strip()
        return text
    import json 
    cleaned = clean_llm_json(llm_json)

    try:
        result = json.loads(cleaned)
    except Exception:
        return {
        "error": "LLM returned invalid JSON",
        "raw": llm_json,
        "cleaned_attempt": cleaned
    }

    if not result.get("flag"):
        return {
            "flag": False,
            "title": None,
            "information": None
        }

    # Auto-ingest memory
    title = result.get("title", "memory")
    information = result.get("information", "")

    chunks = chunk_text(information, chunk_size_chars=1200, overlap=200)

    stored = 0
    for ch in chunks:
        emb = jina_embed(ch)
        add_chunk(title, ch, emb)
        stored += 1

    return {
        "flag": True,
        "title": title,
        "information": information,
        "stored_chunks": stored
    }


@app.delete("/clean-database")
async def clean_db():
    result = clean_messed_up_data()
    return {"status": "ok", "deleted": result["deleted"]}

@app.get("/documents")
async def list_docs():
    """List available docs and stored chunks."""
    return {
        "total_chunks": collection.count(),
        "ids_preview": collection.get(limit=10)["ids"]
    }

@app.get("/find_corrupted-chunks")
def find_corrupted_chunks_endpoint():
    """Find and return corrupted chunks in the database."""
    corrupted_ids = find_corrupted_chunks()
    return {
        "corrupted_count": len(corrupted_ids),
        "corrupted_ids": corrupted_ids
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        reload=True
    )
