# Milestone 5 — End-to-End Ingestion Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `scripts/ingest.py`, the real end-to-end ingestion pipeline that wires together `pipeline.ingest_prep`, `storage.ipfs_client`, `blockchain.registry_client`, and `retrieval.vector_store` for the first time, with idempotent re-runs and per-chunk failure isolation.

**Architecture:** A small extension to `retrieval/vector_store.py::add_documents()` (an optional `cid_by_hash` override, default `None`, fully backward compatible) plus a new orchestration module `scripts/ingest.py`. For every chunk, `ingest.py` first asks the registry if the chunk is already registered (`get_document`); if so it reuses the stored CID with zero IPFS/chain writes; if not, it uploads to IPFS, verifies the round-trip hash *before* registering on-chain (registration is irreversible in this MVP), then registers. All registry/IPFS calls for one chunk are wrapped in a single try/except so a flaky live service degrades that one chunk, not the whole run. Successes are batched into one `add_documents()` call per source file.

**Tech Stack:** `pypdf` (new dependency, BSD-licensed, for `.pdf` text extraction), stdlib `argparse` for the CLI, `unittest.mock` for orchestration unit tests, existing `pytest` `slow` marker for the one integration test that needs both a live Hardhat node and a live IPFS daemon simultaneously — the first milestone to require two live services at once.

**Spec:** This plan implements the design confirmed in-session (see chat transcript / `DECISIONS.md` Milestone 5 entry once written, following the Milestone 4 precedent of baking findings directly into tasks rather than a separate spec file). Two exception types were empirically confirmed against a real local Hardhat node during design, not guessed: a duplicate-registration revert in `register_document` raises `web3.exceptions.ContractLogicError`; an unreachable node raises `requests.exceptions.ConnectionError` unwrapped.

## Global Constraints

- `tests/test_scripts_ingest.py` is built up incrementally across Tasks 2, 3, 4, and 5. Each task's "append" step shows the new test functions and their required imports; when implementing, consolidate all imports at the top of the file (standard Python style) rather than leaving them scattered at each append point — the plan shows them near the tests they belong to only so each task is self-contained to read.

- `retrieval/vector_store.py::add_documents()`'s new `cid_by_hash` parameter must default to `None` and produce byte-identical behavior to today when omitted — Milestone 2's existing tests in `tests/test_retrieval_vector_store.py` must pass **unmodified** (do not edit them, only add new tests).
- The per-chunk failure boundary wraps `get_document()` **and** the fresh-upload path together, not just the fresh-upload path — a dead/flaky Hardhat node during the skip-check is exactly as plausible as a dead IPFS daemon and must get identical handling.
- The catch clause names exact types only: `(IPFSError, requests.exceptions.ConnectionError, ContractLogicError)`. Never a bare `except Exception` — a real bug (`KeyError`, `AttributeError`) must still crash the run instead of being logged as a "transient failure".
- `verify_hash()` returns `bool`, it does not raise. Its `False` case must be funneled through the *same* failure-handling helper as the caught-exception path, not a separate ad-hoc branch.
- `register_document()` is only ever called after `verify_hash()` returns `True` — registration happens after verification, never before, because there is no delete function in this MVP.
- A failed chunk is excluded entirely from that file's `add_documents()` call (not written with a placeholder or wrong CID) — next run's `get_document()` check will correctly retry it.
- `run_ingest()` continues past a single chunk/file failure and only reports failure via a nonzero return value at the very end, after every file has been attempted.
- `.pdf` extraction uses `pypdf` (BSD-licensed), not PyMuPDF (AGPL-3.0) — a deliberate licensing choice for this repo.
- Default publisher key is Hardhat's deterministic Account #1 private key, matching `tests/test_blockchain_registry_client.py`'s `PUBLISHER_1_KEY` constant (`0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d`). No automatic round-robin/hash-based publisher assignment — that's Milestone 9's job, calling `ingest.py` per file with a deliberately chosen key.

---

## File Structure

- **Modify `retrieval/vector_store.py`** — add `cid_by_hash: dict[str, str] | None = None` parameter to `add_documents()`.
- **Modify `tests/test_retrieval_vector_store.py`** — add two new regression tests (no-override placeholder path, override path). Existing tests untouched.
- **Create `scripts/ingest.py`** — the orchestration script: text extraction (`.txt`/`.pdf`), per-chunk CID resolution (skip-if-registered / fresh-upload-verify-register), per-file batching into `add_documents()`, CLI entrypoint.
- **Create `tests/test_scripts_ingest.py`** — fast, fully-mocked unit tests for the orchestration logic (skip path, fresh path, verify-fail path, connection-error path, idempotent re-run, exit code) + one `@pytest.mark.slow` integration test needing a live Hardhat node and IPFS daemon.
- **Modify `pyproject.toml`** — add `pypdf` to `[project.dependencies]`.
- **Modify `DECISIONS.md`** — append a "Milestone 5 — Ingestion Pipeline" section.

---

### Task 1: Extend `add_documents()` with `cid_by_hash` override

**Files:**
- Modify: `retrieval/vector_store.py:24-53`
- Test: `tests/test_retrieval_vector_store.py`

**Interfaces:**
- Produces: `add_documents(records: list[dict], source_file: str, collection: Collection | None = None, cid_by_hash: dict[str, str] | None = None) -> None` — later tasks (Task 3) call this with `cid_by_hash` populated from resolved chunk CIDs.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_retrieval_vector_store.py` (append after the existing `test_add_documents_stores_four_field_metadata` test, do not modify anything above it):

```python
def test_add_documents_without_cid_by_hash_still_writes_placeholder(tmp_path):
    collection = init_store(persist_directory=str(tmp_path))
    records = ingest_prep("Wind turbines convert kinetic energy into electricity.")

    add_documents(records, source_file="wind.txt", collection=collection)

    stored = collection.get(ids=[records[0]["hash"]], include=["metadatas"])
    assert stored["metadatas"][0]["ipfs_cid"] == ""


def test_add_documents_cid_by_hash_overrides_placeholder_for_matching_hash_only(tmp_path):
    collection = init_store(persist_directory=str(tmp_path))
    text = (
        "Geothermal energy harnesses heat from the Earth's interior. "
        "Tidal power generates electricity from the rise and fall of ocean tides."
    )
    records = ingest_prep(text)
    assert len(records) >= 2, "test needs at least two chunks to prove selective override"

    overridden_hash = records[0]["hash"]
    untouched_hash = records[1]["hash"]
    cid_by_hash = {overridden_hash: "QmRealCID123"}

    add_documents(records, source_file="energy.txt", collection=collection, cid_by_hash=cid_by_hash)

    stored = collection.get(ids=[overridden_hash, untouched_hash], include=["metadatas"])
    metadata_by_id = dict(zip(stored["ids"], stored["metadatas"]))
    assert metadata_by_id[overridden_hash]["ipfs_cid"] == "QmRealCID123"
    assert metadata_by_id[untouched_hash]["ipfs_cid"] == ""
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_retrieval_vector_store.py -v -k "cid_by_hash or without_cid_by_hash"`
Expected: FAIL — `add_documents() got an unexpected keyword argument 'cid_by_hash'` (second test), first test currently passes already (placeholder is already the only behavior) but must still be added as an explicit regression lock.

- [ ] **Step 3: Implement the minimal change**

In `retrieval/vector_store.py`, change the `add_documents` signature and metadata construction:

```python
def add_documents(
    records: list[dict],
    source_file: str,
    collection: Collection | None = None,
    cid_by_hash: dict[str, str] | None = None,
) -> None:
    if not records:
        return
    if collection is None:
        collection = init_store()
    # Dedupe by hash within this call: Chroma's upsert() requires unique IDs
    # per call (unlike across calls, where re-upserting the same ID just
    # overwrites in place). Two byte-identical chunks in one batch would
    # otherwise raise DuplicateIDError. Keep the first occurrence of each hash.
    seen: dict[str, dict] = {}
    for record in records:
        seen.setdefault(record["hash"], record)
    records = list(seen.values())
    ids = [record["hash"] for record in records]
    embeddings = [record["embedding"] for record in records]
    documents = [record["text"] for record in records]
    metadatas = [
        {
            "source_file": source_file,
            "chunk_id": record["chunk_id"],
            # Ruling: Chroma silently drops None-valued metadata keys entirely
            # (not an error) — "" is the placeholder that survives the round-trip.
            # cid_by_hash overrides the placeholder only for hashes present in it.
            "ipfs_cid": (cid_by_hash or {}).get(record["hash"], ""),
            "doc_hash": record["hash"],
        }
        for record in records
    ]
    collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_retrieval_vector_store.py -v`
Expected: all tests PASS, including the two new ones and every pre-existing test unmodified.

- [ ] **Step 5: Commit**

```bash
git add retrieval/vector_store.py tests/test_retrieval_vector_store.py
git commit -m "feat: add cid_by_hash override to add_documents for real IPFS CIDs"
```

---

### Task 2: `scripts/ingest.py` — text extraction and CLI skeleton

**Files:**
- Create: `scripts/ingest.py`
- Modify: `pyproject.toml`
- Test: `tests/test_scripts_ingest.py`

**Interfaces:**
- Produces: `_extract_text(path: Path) -> str` — reads `.txt` as UTF-8 or extracts `.pdf` text via `pypdf`.
- Produces: `_collect_input_files(input_path: Path) -> list[Path]` — resolves `--input` (file or directory) into a sorted list of `.txt`/`.pdf` file paths. Later tasks (Task 4) call this from `run_ingest()`.

- [ ] **Step 1: Add the `pypdf` dependency**

In `pyproject.toml`, add `"pypdf>=4.0",` to the `dependencies` list (alphabetically near `pandas`/`numpy` is fine, matching the existing loose ordering).

Run: `uv sync`
Expected: `pypdf` installs cleanly, no conflicts.

- [ ] **Step 2: Write the failing tests**

Create `tests/test_scripts_ingest.py`:

```python
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

import pytest

from scripts.ingest import _collect_input_files, _extract_text


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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_scripts_ingest.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.ingest'`

- [ ] **Step 4: Write the minimal implementation**

Create `scripts/ingest.py` with the extraction/collection pieces (the rest of the module is filled in by Tasks 3-4; this step only needs enough for the tests above to pass):

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pypdf

_TEXT_SUFFIXES = {".txt", ".pdf"}


def _extract_text(path: Path) -> str:
    if path.suffix == ".pdf":
        reader = pypdf.PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8")


def _collect_input_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(
        p for p in input_path.iterdir() if p.is_file() and p.suffix in _TEXT_SUFFIXES
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_scripts_ingest.py -v`
Expected: all 4 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/ingest.py tests/test_scripts_ingest.py pyproject.toml uv.lock
git commit -m "feat: add text extraction and input collection to ingest.py"
```

---

### Task 3: Per-chunk CID resolution — skip path, fresh path, failure handling

**Files:**
- Modify: `scripts/ingest.py`
- Test: `tests/test_scripts_ingest.py`

**Interfaces:**
- Consumes: `registry_client.get_document(doc_hash: str) -> dict` (keys: `publisher`, `content_hash`, `ipfs_cid`, `reputation_score`, `timestamp`, `is_registered`), `registry_client.register_document(doc_hash: str, ipfs_cid: str, signing_key: str | None = None) -> dict`, `ipfs_client.upload_text(text: str) -> str`, `ipfs_client.verify_hash(cid: str, expected_hash: str) -> bool`, `ipfs_client.IPFSError`.
- Produces: `PUBLISHER_1_KEY: str` module constant (default publisher key). `_resolve_chunk_cid(record: dict, source_file: str, publisher_key: str) -> str | None` — returns the resolved CID, or `None` if this chunk failed and must be excluded (Task 4's `run_ingest()` filters `None` results out of both `cid_by_hash` and the records passed to `add_documents`).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_scripts_ingest.py`:

```python
from unittest.mock import patch

import requests
from web3.exceptions import ContractLogicError

from scripts.ingest import PUBLISHER_1_KEY, _resolve_chunk_cid

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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_scripts_ingest.py -v -k resolve_chunk_cid`
Expected: FAIL — `ImportError: cannot import name '_resolve_chunk_cid' from 'scripts.ingest'`

- [ ] **Step 3: Write the minimal implementation**

Add to `scripts/ingest.py` (below the imports added in Task 2):

```python
import sys
from pathlib import Path

import requests
from web3.exceptions import ContractLogicError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pypdf

from blockchain.registry_client import get_document, register_document
from storage.ipfs_client import IPFSError, upload_text, verify_hash

# Hardhat's deterministic default Account #1 private key — same constant
# used in tests/test_blockchain_registry_client.py as PUBLISHER_1_KEY.
# Publicly known test-only key, no real funds.
PUBLISHER_1_KEY = "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d"

_TEXT_SUFFIXES = {".txt", ".pdf"}


def _extract_text(path: Path) -> str:
    if path.suffix == ".pdf":
        reader = pypdf.PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8")


def _collect_input_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(
        p for p in input_path.iterdir() if p.is_file() and p.suffix in _TEXT_SUFFIXES
    )


def _log_chunk_failure(source_file: str, chunk_id: int, doc_hash: str, reason: str) -> None:
    print(
        f"[ingest] FAILED chunk source_file={source_file} chunk_id={chunk_id} "
        f"doc_hash={doc_hash}: {reason}",
        file=sys.stderr,
    )


def _resolve_chunk_cid(record: dict, source_file: str, publisher_key: str) -> str | None:
    doc_hash = record["hash"]
    try:
        document = get_document(doc_hash)
        if document["is_registered"]:
            return document["ipfs_cid"]

        cid = upload_text(record["text"])
        verified = verify_hash(cid, doc_hash)
        if not verified:
            _log_chunk_failure(
                source_file, record["chunk_id"], doc_hash,
                f"verify_hash returned False for cid={cid}",
            )
            return None

        register_document(doc_hash, cid, signing_key=publisher_key)
        return cid
    except (IPFSError, requests.exceptions.ConnectionError, ContractLogicError) as exc:
        _log_chunk_failure(source_file, record["chunk_id"], doc_hash, repr(exc))
        return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_scripts_ingest.py -v`
Expected: all tests from Task 2 and Task 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/ingest.py tests/test_scripts_ingest.py
git commit -m "feat: add per-chunk CID resolution with skip/fresh paths to ingest.py"
```

---

### Task 4: `run_ingest()` orchestration, batching, exit codes, and CLI entrypoint

**Files:**
- Modify: `scripts/ingest.py`
- Test: `tests/test_scripts_ingest.py`

**Interfaces:**
- Consumes: `_collect_input_files`, `_extract_text`, `_resolve_chunk_cid`, `PUBLISHER_1_KEY` (Tasks 2-3), `pipeline.ingest_prep.ingest_prep(raw_text: str) -> list[dict]`, `retrieval.vector_store.add_documents(records, source_file, collection=None, cid_by_hash=None) -> None` (Task 1).
- Produces: `run_ingest(input_path: str, publisher_key: str = PUBLISHER_1_KEY, collection=None) -> int` — returns `0` if every chunk across every file succeeded, `1` if any chunk failed. `main() -> None` — CLI entrypoint, `sys.exit(run_ingest(...))`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_scripts_ingest.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_scripts_ingest.py -v -k run_ingest`
Expected: FAIL — `ImportError: cannot import name 'run_ingest' from 'scripts.ingest'`

- [ ] **Step 3: Write the minimal implementation**

Add to `scripts/ingest.py` (below `_resolve_chunk_cid`):

```python
import argparse

from pipeline.ingest_prep import ingest_prep
from retrieval.vector_store import add_documents


def run_ingest(input_path: str, publisher_key: str = PUBLISHER_1_KEY, collection=None) -> int:
    any_failures = False

    for file_path in _collect_input_files(Path(input_path)):
        raw_text = _extract_text(file_path)
        records = ingest_prep(raw_text)

        cid_by_hash: dict[str, str] = {}
        surviving_records = []
        for record in records:
            cid = _resolve_chunk_cid(record, source_file=file_path.name, publisher_key=publisher_key)
            if cid is None:
                any_failures = True
                continue
            cid_by_hash[record["hash"]] = cid
            surviving_records.append(record)

        add_documents(
            surviving_records,
            source_file=file_path.name,
            collection=collection,
            cid_by_hash=cid_by_hash,
        )

    return 1 if any_failures else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest text/PDF documents into DeRAG.")
    parser.add_argument("--input", required=True, help="Path to a file or directory of .txt/.pdf files")
    parser.add_argument(
        "--publisher-key",
        default=PUBLISHER_1_KEY,
        help="Private key used to sign on-chain registrations (defaults to Hardhat Account #1)",
    )
    args = parser.parse_args()
    sys.exit(run_ingest(args.input, publisher_key=args.publisher_key))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_scripts_ingest.py -v`
Expected: all tests from Tasks 2-4 PASS.

- [ ] **Step 5: Run the full fast suite to confirm no regressions**

Run: `uv run pytest -v`
Expected: all tests PASS (slow tests deselected by default `addopts`).

- [ ] **Step 6: Commit**

```bash
git add scripts/ingest.py tests/test_scripts_ingest.py
git commit -m "feat: add run_ingest orchestration and CLI entrypoint to ingest.py"
```

---

### Task 5: Integration test (both live services) and manual verification

**Files:**
- Modify: `tests/test_scripts_ingest.py`

**Interfaces:**
- Consumes: `run_ingest` (Task 4), `retrieval.vector_store.init_store`, `storage.ipfs_client.verify_hash`, `blockchain.registry_client.verify_integrity`.

- [ ] **Step 1: Write the integration test**

Append to `tests/test_scripts_ingest.py`:

```python
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
        "Demand response programs shift consumption away from peak load hours."
    )
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
```

- [ ] **Step 2: Start both live services and run the integration test**

Terminal 1: `cd contracts && npx hardhat node`
Terminal 2 (after Terminal 1 is up): `cd contracts && npx hardhat run scripts/deploy.ts --network localhost`
Terminal 3: `ipfs daemon`
Terminal 4: `uv run pytest -m slow tests/test_scripts_ingest.py -v -k end_to_end`

Expected: PASS. If it fails, use `superpowers:systematic-debugging` rather than patching the test to hide a real failure.

- [ ] **Step 3: Manual verification of the CLI itself (definition-of-done check)**

With both services still running:

```bash
uv run python scripts/ingest.py --input data/sample.txt
```

Expected: exits 0, prints no `[ingest] FAILED` lines. Then run it again:

```bash
uv run python scripts/ingest.py --input data/sample.txt
```

Expected: exits 0, and manually confirm via the Hardhat node's console output (or a quick `verify_integrity` check in a Python shell) that no new `registerDocument` transactions were mined on the second run — the skip path took over for every chunk.

Then test the directory-input path with 2-3 files (e.g. copy `data/sample.txt` plus a couple of other `.txt`/`.pdf` files into a scratch directory):

```bash
uv run python scripts/ingest.py --input /path/to/scratch/multi-file-dir
```

Expected: exits 0, each file's basename appears as `source_file` in Chroma metadata for its chunks.

- [ ] **Step 4: Commit**

```bash
git add tests/test_scripts_ingest.py
git commit -m "test: add live end-to-end integration test for ingest.py"
```

---

### Task 6: `DECISIONS.md` entry

**Files:**
- Modify: `DECISIONS.md`

- [ ] **Step 1: Confirm Milestone 2's tests still pass unmodified**

Run: `uv run pytest tests/test_retrieval_vector_store.py -v`
Expected: all pre-existing tests (from before Task 1's edits) PASS with no changes made to their code — confirms `cid_by_hash`'s `None` default is fully backward compatible. Note the exact command and result in the DECISIONS.md entry below.

- [ ] **Step 2: Append the Milestone 5 section**

Add to the end of `DECISIONS.md`, following the Milestone 4 section's format and voice:

```markdown
## Milestone 5 — Ingestion Pipeline (2026-08-18)

- **`add_documents()` extended with `cid_by_hash: dict[str, str] | None = None`**,
  keyed by `doc_hash` (the same field already used as the Chroma ID). Overrides
  the `""` placeholder only for hashes present in the dict; defaults to `None`
  so every existing caller is unaffected. Confirmed by running
  `uv run pytest tests/test_retrieval_vector_store.py -v` after the change —
  all of Milestone 2's original tests passed with zero edits to their code,
  alongside two new direct regression tests (no-override placeholder path,
  selective-override path) added to the same file.
- **Skip-path trusts the stored `ipfs_cid` without re-running `verify_hash`.**
  `get_document()`'s six-field return already carries `ipfs_cid` — this is
  the reason Milestone 3 built the full-struct getter rather than only
  `verify_integrity()`'s three-tuple. Re-verifying every already-registered
  chunk against the live IPFS daemon on every ingestion run is redundant
  work with no new information; that kind of live audit belongs at
  query/display time (a later milestone), not ingestion time.
- **Safety-net ordering on the fresh path**: `upload_text()` →
  `verify_hash()` → `register_document()`, in that order. A `False` from
  `verify_hash()` aborts that chunk (excluded from the batch, logged to
  stderr) before `register_document()` is ever called — on-chain
  registration is effectively permanent in this MVP (no delete function),
  so the check has to happen before the write.
- **Failure isolation is per-chunk, not per-run.** The per-chunk try/except
  wraps `get_document()` as well as the fresh-upload path — a flaky/dead
  Hardhat node during the skip-check is exactly as plausible a transient
  failure as a dead IPFS daemon and gets identical treatment: logged,
  excluded from that file's batch, and the run continues to the next
  chunk/file. Only at the very end, after every file has been attempted,
  does `run_ingest()` return a nonzero exit code if anything failed — a
  corpus of 50-100 chunks (Milestone 9's scale) shouldn't die entirely over
  one transient hiccup, and a failed chunk is simply retried automatically
  on the next run since it was never registered.
- **Exception types confirmed live against a real Hardhat node, not
  guessed**: an unreachable node raises `requests.exceptions.ConnectionError`
  unwrapped (`registry_client.py` does no wrapping of its own — same
  underlying exception class Milestone 4's `IPFSError` wraps for IPFS
  connection failures); a duplicate-registration revert raises
  `web3.exceptions.ContractLogicError` (confirmed via a live
  `register_document()` call against an already-registered hash — the
  existing test in `tests/test_blockchain_registry_client.py` had used a
  bare `pytest.raises(Exception)` for this case, which this milestone's
  design work replaced with the precise type). `scripts/ingest.py`'s catch
  clause names exactly `(IPFSError, requests.exceptions.ConnectionError,
  ContractLogicError)` — never a bare `except Exception` — so a real bug
  still crashes the run instead of being silently logged as a data problem.
- **`publisher_key` defaults to a single fixed account** (Hardhat Account
  #1, matching `tests/test_blockchain_registry_client.py`'s
  `PUBLISHER_1_KEY`), overridable per invocation. No automatic round-robin
  or hash-based assignment across the spec's Accounts #1-#5 publisher
  roles — that decision belongs to Milestone 9, which will call
  `ingest.py` once per source file with deliberately curated
  `publisher_key` values once it's actually simulating which document
  comes from which source, not to this milestone's plumbing.
- **PDF extraction: `pypdf`, not PyMuPDF.** PyMuPDF is faster and more
  accurate but AGPL-3.0-licensed (commercial/closed-source use requires a
  paid Artifex license); `pypdf` is BSD-licensed and sufficient for the
  plain-prose PDFs in Milestone 9's demo corpus, where extraction speed at
  a handful of files doesn't matter. Confirmed current via a web search
  during design (2026-08-18) rather than assumed from stale knowledge —
  PyMuPDF's AGPL status and pypdf's continued active maintenance under the
  py-pdf org were both still current as of that search.
```

- [ ] **Step 3: Commit**

```bash
git add DECISIONS.md
git commit -m "docs: record Milestone 5 ingestion pipeline decisions"
```
