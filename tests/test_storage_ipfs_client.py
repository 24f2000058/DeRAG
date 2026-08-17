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

from storage.ipfs_client import _api_base_url


def test_api_base_url_parses_multiaddr(monkeypatch):
    monkeypatch.setenv("IPFS_API_URL", "/ip4/127.0.0.1/tcp/5001")
    assert _api_base_url() == "http://127.0.0.1:5001"


def test_api_base_url_parses_different_host_and_port(monkeypatch):
    monkeypatch.setenv("IPFS_API_URL", "/ip4/10.0.0.5/tcp/5002")
    assert _api_base_url() == "http://10.0.0.5:5002"
