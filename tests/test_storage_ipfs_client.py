"""Tests for storage/ipfs_client.py.

The fast tests below (multiaddr parsing) need no external services.

Tests marked @pytest.mark.slow require a real local Kubo IPFS daemon
running and reachable at the multiaddr in IPFS_API_URL (see
.env.example: /ip4/127.0.0.1/tcp/5001), with its public gateway on port
8080 (Kubo's default). Start one before running the slow suite:

    ipfs daemon

Then run: `IPFS_API_URL=/ip4/127.0.0.1/tcp/5001 uv run pytest -m slow tests/test_storage_ipfs_client.py`
"""

import os

import pytest

from storage.ipfs_client import _api_base_url, IPFSError, upload_text, fetch_text


def test_api_base_url_parses_multiaddr(monkeypatch):
    monkeypatch.setenv("IPFS_API_URL", "/ip4/127.0.0.1/tcp/5001")
    assert _api_base_url() == "http://127.0.0.1:5001"


def test_api_base_url_parses_different_host_and_port(monkeypatch):
    monkeypatch.setenv("IPFS_API_URL", "/ip4/10.0.0.5/tcp/5002")
    assert _api_base_url() == "http://10.0.0.5:5002"


@pytest.mark.slow
def test_upload_text_returns_a_cid_string():
    cid = upload_text("hello from DeRAG milestone 4")
    assert isinstance(cid, str)
    assert cid.startswith("Qm")


@pytest.mark.slow
def test_upload_text_raises_ipfs_error_when_daemon_unreachable(monkeypatch):
    # An unused local TCP port — well-formed multiaddr, nothing listening,
    # so requests raises a connection-level error we must catch and wrap.
    monkeypatch.setenv("IPFS_API_URL", "/ip4/127.0.0.1/tcp/1")
    with pytest.raises(IPFSError):
        upload_text("this will never reach a daemon")


@pytest.mark.slow
def test_upload_then_fetch_round_trips_identical_text():
    original = "The quick brown fox jumps over the lazy dog."
    cid = upload_text(original)
    fetched = fetch_text(cid)
    assert fetched == original


@pytest.mark.slow
def test_fetch_text_raises_ipfs_error_for_malformed_cid():
    with pytest.raises(IPFSError):
        fetch_text("not-a-valid-cid")


@pytest.mark.slow
def test_fetch_text_raises_ipfs_error_for_well_formed_but_missing_cid():
    # Valid CIDv0 shape, but content was never added to this node.
    with pytest.raises(IPFSError):
        fetch_text("QmYwAPJzv5CZsnA625s3Xf2nemtYgPpHdWEz79ojWnPbdG")


@pytest.mark.slow
def test_fetch_text_raises_ipfs_error_when_daemon_unreachable(monkeypatch):
    monkeypatch.setenv("IPFS_API_URL", "/ip4/127.0.0.1/tcp/1")
    with pytest.raises(IPFSError):
        fetch_text("QmYwAPJzv5CZsnA625s3Xf2nemtYgPpHdWEz79ojWnPbdG")


from pipeline.hashing import hash_chunk
from storage.ipfs_client import verify_hash


@pytest.mark.slow
def test_verify_hash_true_for_correctly_hashed_content():
    text = "Milestone 4 content for hash verification."
    cid = upload_text(text)
    expected_hash = hash_chunk(text)
    assert verify_hash(cid, expected_hash) is True


@pytest.mark.slow
def test_verify_hash_false_for_tampered_expected_hash():
    text = "Original content that was actually uploaded."
    different_text = "Different content — simulates a tampered hash claim."
    cid = upload_text(text)
    wrong_hash = hash_chunk(different_text)
    assert verify_hash(cid, wrong_hash) is False


@pytest.mark.slow
def test_verbatim_hash_survives_real_ipfs_round_trip():
    """Proves Milestone 1's verbatim-hashing contract end-to-end.

    This is not a generic round-trip check (that's already covered by
    test_upload_then_fetch_round_trips_identical_text above). This test
    proves the specific claim Milestone 1 depends on: for text of the
    kind normalize_text() would already have normalized (mixed case,
    irregular whitespace) BEFORE it ever reaches hash_chunk, the hash
    computed before upload equals the hash computed on the exact same
    content after a real fetch from IPFS — with zero re-normalization
    in between. If fetch_text ever re-normalized or otherwise altered
    the bytes it returns, this test (not the generic round-trip test)
    is what would catch it.
    """
    already_normalized_text = "  The QUICK Brown   Fox\tJumps\n\nOver the LAZY dog  "
    hash_before_upload = hash_chunk(already_normalized_text)

    cid = upload_text(already_normalized_text)
    fetched_text = fetch_text(cid)
    hash_after_fetch = hash_chunk(fetched_text)

    assert hash_after_fetch == hash_before_upload
