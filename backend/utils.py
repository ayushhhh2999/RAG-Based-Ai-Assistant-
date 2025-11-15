# utils.py
import math
import re

def chunk_text(text: str, chunk_size_chars: int = 1000, overlap: int = 200):
    """
    Simple char-based chunker with overlap. Returns list of chunk strings.
    """
    text = re.sub(r"\s+", " ", text).strip()
    start = 0
    chunks = []
    L = len(text)
    while start < L:
        end = start + chunk_size_chars
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
    return chunks

def build_prompt(query: str, retrieved_chunks):
    """
    retrieved_chunks: list of dicts: {"id":..., "document":..., "metadata":..., "distance":...}
    Build a prompt instructing the LLM to use only provided context.
    """
    ctx = "\n\n".join([f"Source [{r.get('id', i)}]: {r['document']}" for i, r in enumerate(retrieved_chunks)])
    prompt = f"""
You are a helpful assistant that answers user questions using ONLY the provided context. If the answer cannot be found in the context, reply "I don't know".

Context:
{ctx}

User question:
{query}

Provide a concise answer and, when possible
"""
    return prompt
