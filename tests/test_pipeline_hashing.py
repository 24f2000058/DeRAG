"""
Milestone 4 dependency: Milestone 4's IPFS round trip fetches stored chunk
text back from IPFS and re-hashes it to verify integrity. That re-hash must
equal the hash computed here at ingest time. hash_chunk() hashes exactly the
string it is given (explicit UTF-8, no implicit encoding, no internal
re-normalization) — so as long as Milestone 4 hashes the fetched chunk text
verbatim (not re-running it through normalize_text), the hashes match by
construction. This test file documents and locks that contract.
"""
import hashlib

from pipeline.hashing import hash_chunk


def test_hash_is_deterministic():
    text = "the quick brown fox"
    assert hash_chunk(text) == hash_chunk(text)


def test_different_inputs_produce_different_hashes():
    assert hash_chunk("the quick brown fox") != hash_chunk("the lazy dog")


def test_hash_matches_explicit_sha256_utf8():
    text = "café résumé"
    expected = hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert hash_chunk(text) == expected


def test_hash_is_stable_across_normalize_then_chunk_pipeline():
    # The exact contract Milestone 4 relies on: hashing a chunk that came
    # out of normalize_text + chunk_text is just hash_chunk() on that
    # string — no special-casing, no re-normalization step.
    from pipeline.chunking import chunk_text
    from pipeline.normalize import normalize_text

    normalized = normalize_text("Café RÉSUMÉ with MIXED case")
    chunks = chunk_text(normalized)
    for chunk in chunks:
        assert hash_chunk(chunk) == hashlib.sha256(chunk.encode("utf-8")).hexdigest()
