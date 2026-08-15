from pipeline.ingest_prep import ingest_prep
from retrieval.vector_store import add_documents, init_store


def test_init_store_creates_cosine_collection(tmp_path):
    collection = init_store(persist_directory=str(tmp_path))
    assert collection.name == "derag_chunks"
    assert collection.configuration_json["hnsw"]["space"] == "cosine"


def test_init_store_is_idempotent_against_same_directory(tmp_path):
    first = init_store(persist_directory=str(tmp_path))
    second = init_store(persist_directory=str(tmp_path))
    assert first.name == second.name == "derag_chunks"


def test_add_documents_stores_four_field_metadata(tmp_path):
    collection = init_store(persist_directory=str(tmp_path))
    records = ingest_prep("Solar panels convert sunlight directly into electricity.")

    add_documents(records, source_file="solar.txt", collection=collection)

    stored = collection.get(ids=[records[0]["hash"]], include=["metadatas", "documents"])
    assert stored["ids"] == [records[0]["hash"]]
    metadata = stored["metadatas"][0]
    assert set(metadata.keys()) == {"source_file", "chunk_id", "ipfs_cid", "doc_hash"}
    assert metadata["source_file"] == "solar.txt"
    assert metadata["chunk_id"] == records[0]["chunk_id"]
    # ipfs_cid uses empty string instead of None because Chroma doesn't support None in metadata
    assert metadata["ipfs_cid"] == ""
    assert metadata["doc_hash"] == records[0]["hash"]
    assert stored["documents"][0] == records[0]["text"]


def test_add_documents_reingesting_identical_chunk_does_not_duplicate(tmp_path):
    collection = init_store(persist_directory=str(tmp_path))
    records = ingest_prep("Solar panels convert sunlight directly into electricity.")

    add_documents(records, source_file="solar.txt", collection=collection)
    count_after_first = collection.count()

    add_documents(records, source_file="solar.txt", collection=collection)
    count_after_second = collection.count()

    assert count_after_first == count_after_second == len(records)


def test_add_documents_empty_records_is_noop(tmp_path):
    collection = init_store(persist_directory=str(tmp_path))
    add_documents([], source_file="empty.txt", collection=collection)
    assert collection.count() == 0
