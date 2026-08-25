from unittest.mock import patch

import pytest

from app.ui_helpers import (
    DERAG_MODE,
    STANDARD_RAG_MODE,
    format_audit_row,
    format_hash_short,
    get_context_docs,
)
from blockchain.registry_client import RegistryError
from storage.ipfs_client import IPFSError


def test_format_hash_short_truncates_long_hash():
    doc_hash = "a" * 64
    result = format_hash_short(doc_hash)
    assert result == "aaaaaaaaaa...aaaa"


def test_format_hash_short_returns_short_hash_unchanged():
    short = "abc123"
    assert format_hash_short(short) == short


def test_format_audit_row_success_shape():
    fake_doc = {
        "publisher": "0xPublisher",
        "content_hash": "0x" + "a" * 64,
        "ipfs_cid": "QmTest",
        "reputation_score": 5,
        "timestamp": 12345,
        "is_registered": True,
    }
    with patch("app.ui_helpers.get_document", return_value=fake_doc) as mock_get_document, \
         patch("app.ui_helpers.verify_hash", return_value=True) as mock_verify_hash:
        row = format_audit_row("a" * 64)

    mock_get_document.assert_called_once_with("a" * 64)
    mock_verify_hash.assert_called_once_with("QmTest", "a" * 64)
    assert row == {
        "doc_hash_short": format_hash_short("a" * 64),
        "ipfs_cid": "QmTest",
        "publisher": "0xPublisher",
        "reputation": 5,
        "verified": True,
    }


def test_format_audit_row_unverified_badge_when_hash_mismatch():
    fake_doc = {
        "publisher": "0xPublisher",
        "content_hash": "0x" + "a" * 64,
        "ipfs_cid": "QmTest",
        "reputation_score": 5,
        "timestamp": 12345,
        "is_registered": True,
    }
    with patch("app.ui_helpers.get_document", return_value=fake_doc), \
         patch("app.ui_helpers.verify_hash", return_value=False):
        row = format_audit_row("a" * 64)

    assert row["verified"] is False


def test_format_audit_row_collapses_on_registry_error():
    with patch("app.ui_helpers.get_document", side_effect=RegistryError("blockchain down")) as mock_get_document, \
         patch("app.ui_helpers.verify_hash") as mock_verify_hash:
        row = format_audit_row("a" * 64)

    mock_verify_hash.assert_not_called()
    assert row["doc_hash_short"] == format_hash_short("a" * 64)
    assert "error" in row
    assert "publisher" not in row


def test_format_audit_row_keeps_registry_data_on_ipfs_error():
    fake_doc = {
        "publisher": "0xPublisher",
        "content_hash": "0x" + "a" * 64,
        "ipfs_cid": "QmTest",
        "reputation_score": 5,
        "timestamp": 12345,
        "is_registered": True,
    }
    with patch("app.ui_helpers.get_document", return_value=fake_doc), \
         patch("app.ui_helpers.verify_hash", side_effect=IPFSError("gateway down")):
        row = format_audit_row("a" * 64)

    assert row["publisher"] == "0xPublisher"
    assert row["reputation"] == 5
    assert row["ipfs_cid"] == "QmTest"
    assert row["verified"] is None
    assert "verify_error" in row


def test_get_context_docs_standard_mode_calls_standard_rag_answer_context():
    fake_collection = object()
    with patch("app.ui_helpers.standard_rag_answer_context", return_value=[{"text": "x"}]) as mock_fn:
        result = get_context_docs("query", STANDARD_RAG_MODE, collection=fake_collection)

    mock_fn.assert_called_once_with("query", collection=fake_collection)
    assert result == [{"text": "x"}]


def test_get_context_docs_derag_mode_calls_trust_weighted_rerank_without_collection():
    with patch("app.ui_helpers.trust_weighted_rerank", return_value=[{"text": "y"}]) as mock_fn:
        result = get_context_docs("query", DERAG_MODE, collection=object())

    mock_fn.assert_called_once_with("query")
    assert result == [{"text": "y"}]


def test_get_context_docs_unknown_mode_raises_value_error():
    with pytest.raises(ValueError):
        get_context_docs("query", "not-a-real-mode")
