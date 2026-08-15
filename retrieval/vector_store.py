import os

import chromadb
from chromadb.api.models.Collection import Collection
from dotenv import load_dotenv

from pipeline.embeddings import embed

_COLLECTION_NAME = "derag_chunks"


def init_store(persist_directory: str | None = None) -> Collection:
    if persist_directory is None:
        load_dotenv()
        persist_directory = os.environ.get("CHROMA_DB_DIR", "./vector_store")
    client = chromadb.PersistentClient(path=persist_directory)
    return client.get_or_create_collection(
        _COLLECTION_NAME,
        configuration={"hnsw": {"space": "cosine"}},
        embedding_function=None,
    )


def add_documents(
    records: list[dict], source_file: str, collection: Collection | None = None
) -> None:
    if not records:
        return
    if collection is None:
        collection = init_store()
    ids = [record["hash"] for record in records]
    embeddings = [record["embedding"] for record in records]
    documents = [record["text"] for record in records]
    metadatas = [
        {
            "source_file": source_file,
            "chunk_id": record["chunk_id"],
            "ipfs_cid": "",  # Empty string instead of None (Chroma doesn't support None in metadata)
            "doc_hash": record["hash"],
        }
        for record in records
    ]
    collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)


def query(query_text: str, k: int = 10, collection: Collection | None = None) -> list[dict]:
    if collection is None:
        collection = init_store()
    query_embedding = embed([query_text])[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    ids = results["ids"][0]
    if not ids:
        return []
    hits = [
        {
            "chroma_id": chroma_id,
            "text": text,
            "similarity": 1.0 - distance,
            "metadata": metadata,
        }
        for chroma_id, text, metadata, distance in zip(
            ids, results["documents"][0], results["metadatas"][0], results["distances"][0]
        )
    ]
    hits.sort(key=lambda hit: hit["similarity"], reverse=True)
    return hits
