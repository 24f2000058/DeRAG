from retrieval.vector_store import init_store


def test_init_store_creates_cosine_collection(tmp_path):
    collection = init_store(persist_directory=str(tmp_path))
    assert collection.name == "derag_chunks"
    assert collection.configuration_json["hnsw"]["space"] == "cosine"


def test_init_store_is_idempotent_against_same_directory(tmp_path):
    first = init_store(persist_directory=str(tmp_path))
    second = init_store(persist_directory=str(tmp_path))
    assert first.name == second.name == "derag_chunks"
