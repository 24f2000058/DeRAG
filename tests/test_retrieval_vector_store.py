from pipeline.ingest_prep import ingest_prep
from retrieval.vector_store import add_documents, init_store, query


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
    # Ruling: Chroma silently drops None-valued metadata keys entirely (not an
    # error) — "" is the placeholder that survives the round-trip
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


def test_add_documents_dedupes_identical_chunks_within_one_call(tmp_path):
    collection = init_store(persist_directory=str(tmp_path))
    # Repeated boilerplate text collapses (after normalize_text's whitespace
    # collapsing) into multiple chunks sharing the same content hash. A naive
    # upsert() call with duplicate IDs in one batch raises Chroma's
    # DuplicateIDError; add_documents() must dedupe by hash before upserting.
    records = ingest_prep(("A" * 400 + " ") * 6)
    assert len(records) > 1
    assert len({record["hash"] for record in records}) == 1

    add_documents(records, source_file="boilerplate.txt", collection=collection)

    assert collection.count() == 1


def test_add_documents_empty_records_is_noop(tmp_path):
    collection = init_store(persist_directory=str(tmp_path))
    add_documents([], source_file="empty.txt", collection=collection)
    assert collection.count() == 0


SAMPLE_TOPICS = {
    "solar.txt": "Solar panels convert sunlight directly into electricity using the photovoltaic effect.",
    "pasta.txt": "To cook pasta al dente, boil salted water and simmer the noodles for eight to ten minutes.",
    "jazz.txt": "Bebop jazz emerged in the 1940s with fast tempos and complex chord improvisation.",
    "volcano.txt": "Shield volcanoes form from slow-flowing basaltic lava that spreads over wide areas.",
}


def _seed_store(collection):
    for source_file, text in SAMPLE_TOPICS.items():
        add_documents(ingest_prep(text), source_file=source_file, collection=collection)


def test_query_top_hit_matches_expected_topic(tmp_path):
    collection = init_store(persist_directory=str(tmp_path))
    _seed_store(collection)

    hits = query("How do photovoltaic cells generate power from the sun?", collection=collection)

    assert len(hits) > 0
    assert hits[0]["metadata"]["source_file"] == "solar.txt"
    assert set(hits[0].keys()) == {"chroma_id", "text", "similarity", "metadata"}


def test_query_results_sorted_by_similarity_descending(tmp_path):
    collection = init_store(persist_directory=str(tmp_path))
    _seed_store(collection)

    hits = query("jazz improvisation and chord changes", collection=collection)

    similarities = [hit["similarity"] for hit in hits]
    assert similarities == sorted(similarities, reverse=True)


def test_query_respects_k(tmp_path):
    collection = init_store(persist_directory=str(tmp_path))
    _seed_store(collection)

    hits = query("volcanic activity", k=2, collection=collection)

    assert len(hits) == 2


def test_query_empty_collection_returns_empty_list(tmp_path):
    collection = init_store(persist_directory=str(tmp_path))

    hits = query("anything at all", collection=collection)

    assert hits == []
