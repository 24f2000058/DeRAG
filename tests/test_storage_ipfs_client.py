"""Tests for storage/ipfs_client.py.

The fast tests below (multiaddr parsing) need no external services.

Tests marked @pytest.mark.slow require a real local Kubo IPFS daemon
running and reachable at the multiaddr in IPFS_API_URL (see
.env.example: /ip4/127.0.0.1/tcp/5001), with its public gateway on port
8080 (Kubo's default). Start one before running the slow suite:

    ipfs daemon

Then run: `uv run pytest -m slow tests/test_storage_ipfs_client.py`
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
