"""Tests for scripts/ingest.py.

The tests in this file that need no live services (all except the one
marked @pytest.mark.slow) run by default. The slow test requires BOTH:

  - a running Hardhat node on WEB3_PROVIDER_URI (see .env.example),
    deployed via `npx hardhat run scripts/deploy.ts --network localhost`
    from inside contracts/
  - a running `ipfs daemon` reachable at IPFS_API_URL (see .env.example)

Run the fast suite: `uv run pytest tests/test_scripts_ingest.py -v`
Run the slow test:  `uv run pytest -m slow tests/test_scripts_ingest.py -v`
"""

from pathlib import Path
from unittest.mock import patch

import pytest
import requests
from web3.exceptions import ContractLogicError

from scripts.ingest import PUBLISHER_1_KEY, _collect_input_files, _extract_text, _resolve_chunk_cid


def test_extract_text_reads_txt_file_as_utf8(tmp_path):
    txt_path = tmp_path / "sample.txt"
    txt_path.write_text("Solar panels convert sunlight into electricity.", encoding="utf-8")

    assert _extract_text(txt_path) == "Solar panels convert sunlight into electricity."


def test_extract_text_reads_pdf_file(tmp_path):
    pypdf = pytest.importorskip("pypdf")
    pdf_path = tmp_path / "sample.pdf"

    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=200, height=200)
    with open(pdf_path, "wb") as f:
        writer.write(f)

    # A blank page extracts to empty (or near-empty) text — this test only
    # proves the .pdf branch runs pypdf's extraction path without raising,
    # not that extraction produces specific content.
    result = _extract_text(pdf_path)
    assert isinstance(result, str)


def test_collect_input_files_single_file_returns_that_file(tmp_path):
    txt_path = tmp_path / "solo.txt"
    txt_path.write_text("content", encoding="utf-8")

    assert _collect_input_files(txt_path) == [txt_path]


def test_collect_input_files_directory_returns_sorted_txt_and_pdf_only(tmp_path):
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "c.pdf").write_bytes(b"%PDF-1.4 fake")
    (tmp_path / "d.md").write_text("ignored", encoding="utf-8")

    result = _collect_input_files(tmp_path)

    assert [p.name for p in result] == ["a.txt", "b.txt", "c.pdf"]


_RECORD = {"chunk_id": 0, "text": "Solar panels convert sunlight.", "hash": "hash-a", "embedding": [0.1]}


@patch("scripts.ingest.register_document")
@patch("scripts.ingest.verify_hash")
@patch("scripts.ingest.upload_text")
@patch("scripts.ingest.get_document")
def test_resolve_chunk_cid_skip_path_reuses_stored_cid(
    mock_get_document, mock_upload_text, mock_verify_hash, mock_register_document
):
    mock_get_document.return_value = {
        "publisher": "0xabc",
        "content_hash": "0x" + "aa" * 32,
        "ipfs_cid": "QmStoredCID",
        "reputation_score": 0,
        "timestamp": 12345,
        "is_registered": True,
    }

    result = _resolve_chunk_cid(_RECORD, source_file="solar.txt", publisher_key=PUBLISHER_1_KEY)

    assert result == "QmStoredCID"
    mock_upload_text.assert_not_called()
    mock_register_document.assert_not_called()


@patch("scripts.ingest.register_document")
@patch("scripts.ingest.verify_hash")
@patch("scripts.ingest.upload_text")
@patch("scripts.ingest.get_document")
def test_resolve_chunk_cid_fresh_path_calls_in_order(
    mock_get_document, mock_upload_text, mock_verify_hash, mock_register_document
):
    mock_get_document.return_value = {"is_registered": False}
    mock_upload_text.return_value = "QmFreshCID"
    mock_verify_hash.return_value = True
    call_order = []
    mock_upload_text.side_effect = lambda *a, **k: call_order.append("upload_text") or "QmFreshCID"
    mock_verify_hash.side_effect = lambda *a, **k: call_order.append("verify_hash") or True
    mock_register_document.side_effect = lambda *a, **k: call_order.append("register_document")

    result = _resolve_chunk_cid(_RECORD, source_file="solar.txt", publisher_key=PUBLISHER_1_KEY)

    assert result == "QmFreshCID"
    assert call_order == ["upload_text", "verify_hash", "register_document"]
    mock_register_document.assert_called_once_with(
        _RECORD["hash"], "QmFreshCID", signing_key=PUBLISHER_1_KEY
    )


@patch("scripts.ingest.register_document")
@patch("scripts.ingest.verify_hash")
@patch("scripts.ingest.upload_text")
@patch("scripts.ingest.get_document")
def test_resolve_chunk_cid_verify_failure_excludes_chunk_without_registering(
    mock_get_document, mock_upload_text, mock_verify_hash, mock_register_document
):
    mock_get_document.return_value = {"is_registered": False}
    mock_upload_text.return_value = "QmBadCID"
    mock_verify_hash.return_value = False

    result = _resolve_chunk_cid(_RECORD, source_file="solar.txt", publisher_key=PUBLISHER_1_KEY)

    assert result is None
    mock_register_document.assert_not_called()


@patch("scripts.ingest.register_document")
@patch("scripts.ingest.verify_hash")
@patch("scripts.ingest.upload_text")
@patch("scripts.ingest.get_document")
def test_resolve_chunk_cid_get_document_connection_error_is_caught(
    mock_get_document, mock_upload_text, mock_verify_hash, mock_register_document
):
    mock_get_document.side_effect = requests.exceptions.ConnectionError("node unreachable")

    result = _resolve_chunk_cid(_RECORD, source_file="solar.txt", publisher_key=PUBLISHER_1_KEY)

    assert result is None
    mock_upload_text.assert_not_called()
    mock_register_document.assert_not_called()


@patch("scripts.ingest.register_document")
@patch("scripts.ingest.verify_hash")
@patch("scripts.ingest.upload_text")
@patch("scripts.ingest.get_document")
def test_resolve_chunk_cid_register_document_contract_revert_is_caught(
    mock_get_document, mock_upload_text, mock_verify_hash, mock_register_document
):
    mock_get_document.return_value = {"is_registered": False}
    mock_upload_text.return_value = "QmFreshCID"
    mock_verify_hash.return_value = True
    mock_register_document.side_effect = ContractLogicError("execution reverted")

    result = _resolve_chunk_cid(_RECORD, source_file="solar.txt", publisher_key=PUBLISHER_1_KEY)

    assert result is None
