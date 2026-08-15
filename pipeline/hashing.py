import hashlib


def hash_chunk(chunk: str) -> str:
    return hashlib.sha256(chunk.encode("utf-8")).hexdigest()
