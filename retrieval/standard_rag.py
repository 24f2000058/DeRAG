from chromadb.api.models.Collection import Collection

from retrieval.vector_store import query


def standard_rag_answer_context(
    query_text: str, k: int = 3, collection: Collection | None = None
) -> list[dict]:
    return query(query_text, k=k, collection=collection)
