from blockchain.registry_client import _doc_hash_to_bytes32


def test_doc_hash_to_bytes32_round_trips_hex_string_without_prefix():
    doc_hash = "a" * 64
    result = _doc_hash_to_bytes32(doc_hash)
    assert isinstance(result, bytes)
    assert len(result) == 32
    assert result.hex() == doc_hash


def test_doc_hash_to_bytes32_accepts_0x_prefixed_string():
    doc_hash = "0x" + "b" * 64
    result = _doc_hash_to_bytes32(doc_hash)
    assert len(result) == 32
    assert result.hex() == "b" * 64
