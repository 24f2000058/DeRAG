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


import os

import pytest

from blockchain.registry_client import (
    get_document,
    register_document,
    update_reputation,
    verify_integrity,
)

# Hardhat's deterministic default account #1 and #2 private keys (printed by
# `npx hardhat node` on startup) — publicly known test-only keys, no real funds.
PUBLISHER_1_KEY = "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d"
PUBLISHER_2_KEY = "0x5de4111afa1a4b94908f83103eb1f1706367c2e68ca870fc3fb9a804cdab365a"


@pytest.mark.slow
def test_register_update_and_read_back_end_to_end():
    doc_hash = "1" * 64
    register_document(doc_hash, ipfs_cid="QmTestCID", signing_key=PUBLISHER_1_KEY)

    integrity = verify_integrity(doc_hash)
    assert integrity["exists"] is True
    assert integrity["reputation"] == 0

    update_reputation(doc_hash, delta=-7)
    integrity_after = verify_integrity(doc_hash)
    assert integrity_after["reputation"] == -7

    doc = get_document(doc_hash)
    assert doc["ipfs_cid"] == "QmTestCID"
    assert doc["reputation_score"] == -7
    assert doc["is_registered"] is True


@pytest.mark.slow
def test_publisher_attribution_is_per_account_not_global():
    hash_a = "2" * 64
    hash_b = "3" * 64

    register_document(hash_a, ipfs_cid="cid-a", signing_key=PUBLISHER_1_KEY)
    register_document(hash_b, ipfs_cid="cid-b", signing_key=PUBLISHER_2_KEY)

    doc_a = get_document(hash_a)
    doc_b = get_document(hash_b)
    assert doc_a["publisher"] != doc_b["publisher"]

    integrity_a = verify_integrity(hash_a)
    assert integrity_a["publisher"] == doc_a["publisher"]

    # Same hash from a second account must revert (dedup rule), not silently
    # reassign the publisher.
    with pytest.raises(Exception):
        register_document(hash_a, ipfs_cid="cid-a-again", signing_key=PUBLISHER_2_KEY)
