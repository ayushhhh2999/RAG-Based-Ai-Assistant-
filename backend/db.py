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
