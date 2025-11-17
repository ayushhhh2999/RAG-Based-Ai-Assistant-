# db.py
import os
from dotenv import load_dotenv
import chromadb

load_dotenv()

PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")

# NEW correct client (no Settings)
client = chromadb.PersistentClient(path=PERSIST_DIR)

# Create / get collection
collection = client.get_or_create_collection(
    name="personal_docs",
    metadata={"hnsw:space": "cosine"}   # similarity metric
)

def add_chunk(doc_id: str, chunk_text: str, embedding):
    uid = f"{doc_id}-{collection.count()}"

    collection.add(
        ids=[uid],
        documents=[chunk_text],
        embeddings=[embedding.tolist()],
        metadatas=[{"doc_id": doc_id}]
    )

    return uid


def query_embeddings(query_embedding, n_results=4):
    res = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    return res

def clean_messed_up_data():
    """
    Delete ONLY unreadable, binary, garbage chunks.
    Keeps all human-readable text.
    """

    print("🔍 Scanning ChromaDB for corrupt/unreadable chunks...")

    data = collection.get(include=["documents", "metadatas", "embeddings"])

    delete_ids = []

    for doc, meta, emb, item_id in zip(
        data["documents"], data["metadatas"], data["embeddings"], data["ids"]
    ):
        if not is_human_readable(doc):
            delete_ids.append(item_id)

    if not delete_ids:
        print("✅ No corrupted items found. Database is clean.")
        return {"deleted": 0}

    print(f"🧹 Removing {len(delete_ids)} unreadable garbage chunks…")
    collection.delete(ids=delete_ids)

    print("✨ Cleanup Completed!")
    return {"deleted": len(delete_ids)}

import re
import string

def is_human_readable(text: str) -> bool:
    """
    Returns True only if the text looks like genuine
    human-readable language usable by an LLM.
    """

    if not isinstance(text, str):
        return False

    t = text.strip()

    # too short = meaningless
    if len(t) < 10:
        return False

    # PDF / binary markers
    binary_markers = ["endstream", "obj", "xref", "%PDF", "stream"]
    if any(m in t for m in binary_markers):
        return False

    # hex bytecode pattern
    if re.search(r"\\x[0-9A-Fa-f]{2}", t):
        return False

    # lots of weird symbols → likely garbage
    letters = sum(c.isalpha() for c in t)
    symbols = sum(c in string.punctuation for c in t)

    if letters == 0:
        return False

    if symbols > letters * 2:
        return False

    # check actual word structure
    words = t.split()
    real_words = sum(1 for w in words if re.search(r"[A-Za-z]", w))

    if real_words < 3:
        return False

    return True


def find_corrupted_chunks():
    """
    Returns ONLY unreadable/binary/gibberish chunks.
    Does NOT delete anything.
    """

    print("🔍 Scanning chunks for unreadable garbage…")

    data = collection.get(include=["documents", "metadatas", "embeddings"])

    corrupted = []

    for doc, meta, emb, item_id in zip(
        data["documents"], data["metadatas"], data["embeddings"], data["ids"]
    ):
        if not is_human_readable(doc):
            corrupted.append({
                "id": item_id,
                "doc_preview": doc[:200],
                "meta": meta,
            })

    print(f"⚠️ Found {len(corrupted)} unreadable garbage chunks.")
    return corrupted
