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

from blockchain.registry_client import RegistryError
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
    mock_get_document.side_effect = RegistryError("node unreachable")

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
    mock_register_document.side_effect = RegistryError("execution reverted")

    result = _resolve_chunk_cid(_RECORD, source_file="solar.txt", publisher_key=PUBLISHER_1_KEY)

    assert result is None


@patch("scripts.ingest.register_document")
@patch("scripts.ingest.verify_hash")
@patch("scripts.ingest.upload_text")
@patch("scripts.ingest.get_document")
def test_resolve_chunk_cid_unlisted_exception_propagates_instead_of_being_swallowed(
    mock_get_document, mock_upload_text, mock_verify_hash, mock_register_document
):
    mock_get_document.side_effect = KeyError("boom")

    with pytest.raises(KeyError):
        _resolve_chunk_cid(_RECORD, source_file="solar.txt", publisher_key=PUBLISHER_1_KEY)


from scripts.ingest import run_ingest


@patch("scripts.ingest.add_documents")
@patch("scripts.ingest.register_document")
@patch("scripts.ingest.verify_hash")
@patch("scripts.ingest.upload_text")
@patch("scripts.ingest.get_document")
def test_run_ingest_all_success_returns_zero_and_batches_one_call_per_file(
    mock_get_document, mock_upload_text, mock_verify_hash, mock_register_document, mock_add_documents,
    tmp_path,
):
    (tmp_path / "solar.txt").write_text(
        "Solar panels convert sunlight directly into electricity.", encoding="utf-8"
    )
    mock_get_document.return_value = {"is_registered": False}
    mock_upload_text.return_value = "QmFreshCID"
    mock_verify_hash.return_value = True

    exit_code = run_ingest(str(tmp_path), publisher_key="testkey")

    assert exit_code == 0
    mock_add_documents.assert_called_once()
    _, kwargs = mock_add_documents.call_args
    assert kwargs["source_file"] == "solar.txt"
    assert kwargs["cid_by_hash"]
    assert all(cid == "QmFreshCID" for cid in kwargs["cid_by_hash"].values())


@patch("scripts.ingest.add_documents")
@patch("scripts.ingest.register_document")
@patch("scripts.ingest.verify_hash")
@patch("scripts.ingest.upload_text")
@patch("scripts.ingest.get_document")
def test_run_ingest_verify_failure_excludes_chunk_but_continues_and_returns_nonzero(
    mock_get_document, mock_upload_text, mock_verify_hash, mock_register_document, mock_add_documents,
    tmp_path,
):
    # Long text guarantees multiple chunks (chunk_size=500 in pipeline.chunking).
    long_text = ("Wind turbines convert kinetic energy into electricity. " * 20)
    (tmp_path / "wind.txt").write_text(long_text, encoding="utf-8")
    mock_get_document.return_value = {"is_registered": False}
    mock_upload_text.return_value = "QmSomeCID"
    mock_verify_hash.return_value = False

    exit_code = run_ingest(str(tmp_path), publisher_key="testkey")

    assert exit_code == 1
    mock_register_document.assert_not_called()
    # add_documents is still called for the file, but with zero surviving records
    # since every chunk in this synthetic case failed verify_hash.
    mock_add_documents.assert_called_once()
    _, kwargs = mock_add_documents.call_args
    assert kwargs["cid_by_hash"] == {}


@patch("scripts.ingest.add_documents")
@patch("scripts.ingest.register_document")
@patch("scripts.ingest.verify_hash")
@patch("scripts.ingest.upload_text")
@patch("scripts.ingest.get_document")
def test_run_ingest_second_run_makes_zero_new_register_document_calls(
    mock_get_document, mock_upload_text, mock_verify_hash, mock_register_document, mock_add_documents,
    tmp_path,
):
    (tmp_path / "solar.txt").write_text(
        "Solar panels convert sunlight directly into electricity.", encoding="utf-8"
    )
    mock_upload_text.return_value = "QmFreshCID"
    mock_verify_hash.return_value = True

    # First run: nothing registered yet.
    mock_get_document.return_value = {"is_registered": False}
    run_ingest(str(tmp_path), publisher_key="testkey")
    assert mock_register_document.call_count == 1

    # Second run: registry now reports it as already registered.
    mock_get_document.return_value = {"is_registered": True, "ipfs_cid": "QmFreshCID"}
    exit_code = run_ingest(str(tmp_path), publisher_key="testkey")

    assert exit_code == 0
    assert mock_register_document.call_count == 1  # unchanged — no new calls


@patch("scripts.ingest.add_documents")
@patch("scripts.ingest.register_document")
@patch("scripts.ingest.verify_hash")
@patch("scripts.ingest.upload_text")
@patch("scripts.ingest.get_document")
def test_run_ingest_file_level_extraction_failure_does_not_abort_run(
    mock_get_document, mock_upload_text, mock_verify_hash, mock_register_document, mock_add_documents,
    tmp_path,
):
    # One file with invalid UTF-8 bytes (fails _extract_text) alongside one valid file.
    (tmp_path / "bad.txt").write_bytes(b"\xff\xfe invalid utf-8 bytes \x80\x81")
    (tmp_path / "solar.txt").write_text(
        "Solar panels convert sunlight directly into electricity.", encoding="utf-8"
    )
    mock_get_document.return_value = {"is_registered": False}
    mock_upload_text.return_value = "QmFreshCID"
    mock_verify_hash.return_value = True

    exit_code = run_ingest(str(tmp_path), publisher_key="testkey")

    assert exit_code == 1
    # The valid file's chunks must still be batched despite the sibling failure.
    mock_add_documents.assert_called_once()
    _, kwargs = mock_add_documents.call_args
    assert kwargs["source_file"] == "solar.txt"
    assert kwargs["cid_by_hash"]


import secrets

from retrieval.vector_store import init_store
from storage.ipfs_client import verify_hash as real_verify_hash
from blockchain.registry_client import verify_integrity


@pytest.mark.slow
def test_run_ingest_end_to_end_against_live_services(tmp_path):
    """Requires a running `npx hardhat node` (deployed via
    contracts/scripts/deploy.ts) AND a running `ipfs daemon` simultaneously —
    the first test in this repo needing two live services at once.
    """
    unique = secrets.token_hex(8)
    sample_text = (
        f"DeRAG Milestone 5 integration probe {unique}. "
        "Fusion power research explores confining plasma with magnetic fields. "
        "Battery storage technologies buffer intermittent renewable generation. "
        "Grid interconnects allow regions to share surplus electricity capacity. "
        "Demand response programs shift consumption away from peak load hours. "
        "Offshore wind farms exploit stronger and steadier ocean winds. "
        "Hydroelectric dams convert falling water into dispatchable power. "
        "Geothermal plants tap subterranean heat for baseload generation. "
        "Carbon capture systems trap emissions before they reach the atmosphere. "
        "Smart meters give utilities real time visibility into consumption. "
        "Microgrids let communities island themselves during outages. "
        "Electric vehicle fleets can act as distributed storage reservoirs. "
        "Hydrogen electrolysis converts surplus renewable power into fuel. "
        "Nuclear small modular reactors promise compact reliable baseload output. "
        "Tidal barrages harness predictable lunar driven ocean currents. "
        "Biomass gasification turns organic waste into usable syngas. "
        "Power purchase agreements let corporations fund new renewable capacity. "
        "Transmission line upgrades reduce curtailment of remote wind resources. "
        "Vehicle to grid technology returns stored energy back to the network. "
        "Long duration storage chemistries target multi day discharge cycles."
    )
    # chunk_size in pipeline.chunking is 500 chars; the text above runs well
    # past 1000 chars so it reliably splits into >= 3 overlapping chunks.
    input_file = tmp_path / "energy_future.txt"
    input_file.write_text(sample_text, encoding="utf-8")

    chroma_dir = tmp_path / "chroma"
    collection = init_store(persist_directory=str(chroma_dir))

    exit_code = run_ingest(str(input_file), collection=collection)

    assert exit_code == 0
    stored = collection.get(include=["metadatas"])
    assert len(stored["ids"]) >= 3

    for doc_hash, metadata in zip(stored["ids"], stored["metadatas"]):
        cid = metadata["ipfs_cid"]
        assert cid != ""
        assert real_verify_hash(cid, doc_hash) is True

    first_hash = stored["ids"][0]
    integrity = verify_integrity(first_hash)
    assert integrity["exists"] is True
    assert isinstance(integrity["reputation"], int)
