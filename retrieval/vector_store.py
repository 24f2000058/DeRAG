import os

import chromadb
from chromadb.api.models.Collection import Collection
from dotenv import load_dotenv

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
