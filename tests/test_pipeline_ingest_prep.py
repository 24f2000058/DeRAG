import hashlib

from pipeline.ingest_prep import ingest_prep
from pipeline.normalize import normalize_text


def test_ingest_prep_record_shape():
    records = ingest_prep("Hello WORLD. " * 60)
    assert len(records) > 0
    for record in records:
        assert set(record.keys()) == {"chunk_id", "text", "hash", "embedding"}
        assert isinstance(record["chunk_id"], int)
        assert isinstance(record["text"], str)
        assert isinstance(record["hash"], str)
        assert isinstance(record["embedding"], list)
        assert len(record["embedding"]) == 384


def test_ingest_prep_chunk_ids_are_sequential():
    records = ingest_prep("Hello WORLD. " * 60)
    assert [r["chunk_id"] for r in records] == list(range(len(records)))


def test_ingest_prep_hash_matches_normalized_chunk_text():
    raw = "Café RÉSUMÉ with MIXED case. " * 40
    records = ingest_prep(raw)
    for record in records:
        expected_hash = hashlib.sha256(record["text"].encode("utf-8")).hexdigest()
        assert record["hash"] == expected_hash


def test_ingest_prep_text_is_normalized():
    raw = "SHOUTING TEXT with​artifacts"
    records = ingest_prep(raw)
    combined = " ".join(r["text"] for r in records)
    assert combined == normalize_text(combined)
