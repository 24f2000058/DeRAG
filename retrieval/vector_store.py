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
