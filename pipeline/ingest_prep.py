from pipeline.chunking import chunk_text
from pipeline.embeddings import embed
from pipeline.hashing import hash_chunk
from pipeline.normalize import normalize_text


def ingest_prep(raw_text: str) -> list[dict]:
    normalized = normalize_text(raw_text)
    chunks = chunk_text(normalized)
    embeddings = embed(chunks)
    return [
        {
            "chunk_id": i,
            "text": chunk,
            "hash": hash_chunk(chunk),
            "embedding": embedding,
        }
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]
