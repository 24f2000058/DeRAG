# Milestone 8 — Streamlit Audit UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the DeRAG Streamlit page (`app/main.py`) with a query box, a Standard RAG / DeRAG mode toggle, a streamed grounded answer, and an `st.sidebar` audit trail with live provenance verification per source — and close the exception-wrapping gap in `blockchain/registry_client.py` and `generation/llm_client.py`, the two remaining service clients that let raw exceptions escape.

**Architecture:** Three small, independently-testable layers underneath one untested Streamlit page. Layer 1 (this milestone's core fix): two new exception classes matching `storage/ipfs_client.py`'s `IPFSError` pattern exactly, applied at each client's HTTP/RPC boundary. Layer 2: `app/ui_helpers.py`, pure functions with no Streamlit dependency, unit-tested by mocking the three retrieval/blockchain/storage modules they call. Layer 3: `app/main.py`, which wires cached resources, the toggle, streaming output, and the sidebar together — verified manually per the milestone's definition of done, since Streamlit UI code isn't unit-testable.

**Tech Stack:** Python 3.11, pytest, `unittest.mock`, Streamlit 1.59.2 (`st.cache_resource` confirmed present), web3.py 6.20.3, chromadb 1.5.9.

**Spec:** `docs/superpowers/specs/2026-08-25-milestone-8-audit-ui-design.md`

## Global Constraints

- `RegistryError` and `GenerationError` must each be a bare `class XError(Exception): pass`, matching `storage/ipfs_client.py`'s `IPFSError` exactly — no custom `__init__`, no extra fields.
- `RegistryError` wraps: `requests.exceptions.RequestException`, `FileNotFoundError`, `web3.exceptions.ContractLogicError`, `ValueError` (raw JSON-RPC revert payloads sometimes surface as `ValueError` in web3.py 6.x). It must **not** wrap `KeyError` from a missing `WEB3_PROVIDER_URI`/`PRIVATE_KEY` env var — that stays a raw, loud failure.
- `GenerationError` wraps only `requests.exceptions.RequestException` around the `requests.post(...)` call in both `generate_blocking` and `generate_streaming`.
- Every wrap site uses `raise XError(f"...") from exc` so `.__cause__` is preserved — regression tests must assert this, not just the exception type.
- `retrieval/rerank.py`'s `trust_weighted_rerank()` signature is locked (Milestone 6) — do not add a `collection` param to it. `app/ui_helpers.py`'s `get_context_docs()` passes `collection` through to `standard_rag_answer_context()` only.
- `app/main.py` is not unit-tested; verification is manual, per the Phase 1 definition of done (all four services up → query, toggle, streamed answer, sidebar badges all work; stop any one service → clean in-UI error in the relevant section, not a crash).
- Real field/function names confirmed by source inspection (not memory): `trust_weighted_rerank()`'s computed score key is literally `"score"`. `get_document()` returns `{"publisher", "content_hash", "ipfs_cid", "reputation_score", "timestamp", "is_registered"}`. `verify_integrity()` returns `{"exists", "reputation", "publisher"}`.

---

## File Structure

- Modify: `blockchain/registry_client.py` — add `RegistryError`, wrap `connect()` and the RPC call sites in `_send_transaction`, `verify_integrity`, `get_document`.
- Modify: `tests/test_blockchain_registry_client.py` — add regression tests (extends the existing file; the existing fast `_doc_hash_to_bytes32` tests and `@pytest.mark.slow` integration tests stay as-is).
- Modify: `generation/llm_client.py` — add `GenerationError`, wrap the `requests.post` call in `generate_blocking` and `generate_streaming`.
- Modify: `tests/test_generation_llm_client.py` — add regression tests (extends the existing file).
- Create: `app/__init__.py` — empty, matching the sibling-package convention (`blockchain/`, `generation/`, `retrieval/`, `storage/`, `pipeline/` all have one; `app/` currently doesn't).
- Create: `app/ui_helpers.py` — `format_hash_short`, `format_audit_row`, `get_context_docs`, mode constants.
- Create: `tests/test_app_ui_helpers.py` — unit tests for the above, mocking `registry_client`/`ipfs_client`/`standard_rag`/`rerank`.
- Modify: `app/main.py` — the Streamlit page itself.

---

### Task 1: RegistryError in blockchain/registry_client.py

**Files:**
- Modify: `blockchain/registry_client.py`
- Test: `tests/test_blockchain_registry_client.py`
- Modify: `scripts/ingest.py` — its `_resolve_chunk_cid()` catches raw `requests.exceptions.ConnectionError`/`ContractLogicError` around `get_document`/`register_document` today; once those raise `RegistryError` instead, that except clause stops catching anything real and chunk-level failures start propagating uncaught. Update it in the same task, since it's a direct, mechanical consequence of this task's own exception-wrapping change, not new work.
- Modify: `tests/test_scripts_ingest.py` — the two tests that mock `get_document`/`register_document` to raise the raw exception types must mock `RegistryError` instead, so they keep testing a scenario the real client can actually produce.

**Interfaces:**
- Produces: `RegistryError(Exception)`, importable as `from blockchain.registry_client import RegistryError`. `register_document`, `update_reputation`, `verify_integrity`, `get_document` keep their existing signatures and return shapes; they now raise `RegistryError` instead of raw `requests`/`web3`/`FileNotFoundError` exceptions. `connect()` keeps its `-> tuple[Web3, Contract]` signature.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_blockchain_registry_client.py` (append at the end of the file; keep the existing imports and tests untouched):

```python
from unittest.mock import MagicMock, patch

import requests
from web3.exceptions import ContractLogicError

from blockchain.registry_client import RegistryError, _send_transaction, connect

TEST_SIGNING_KEY = "0x" + "1" * 64


def test_connect_wraps_missing_deployment_file_in_registry_error(monkeypatch):
    monkeypatch.setenv("WEB3_PROVIDER_URI", "http://127.0.0.1:8545")
    with patch("builtins.open", side_effect=FileNotFoundError("no such file")):
        with pytest.raises(RegistryError) as exc_info:
            connect()
    assert isinstance(exc_info.value.__cause__, FileNotFoundError)


def test_connect_does_not_wrap_missing_env_var():
    original_value = os.environ.pop("WEB3_PROVIDER_URI", None)
    try:
        with pytest.raises(KeyError):
            connect()
    finally:
        if original_value is not None:
            os.environ["WEB3_PROVIDER_URI"] = original_value


def test_send_transaction_wraps_connection_error_in_registry_error():
    mock_w3 = MagicMock()
    mock_function_call = MagicMock()
    mock_function_call.build_transaction.side_effect = requests.exceptions.ConnectionError("down")

    with pytest.raises(RegistryError) as exc_info:
        _send_transaction(mock_w3, mock_function_call, TEST_SIGNING_KEY)
    assert isinstance(exc_info.value.__cause__, requests.exceptions.ConnectionError)


def test_send_transaction_wraps_contract_logic_error_in_registry_error():
    mock_w3 = MagicMock()
    mock_function_call = MagicMock()
    mock_function_call.build_transaction.side_effect = ContractLogicError("execution reverted: duplicate hash")

    with pytest.raises(RegistryError) as exc_info:
        _send_transaction(mock_w3, mock_function_call, TEST_SIGNING_KEY)
    assert isinstance(exc_info.value.__cause__, ContractLogicError)


def test_verify_integrity_wraps_connection_error_in_registry_error():
    mock_contract = MagicMock()
    mock_contract.functions.verifyIntegrity.return_value.call.side_effect = (
        requests.exceptions.ConnectionError("down")
    )
    with patch("blockchain.registry_client.connect", return_value=(MagicMock(), mock_contract)):
        with pytest.raises(RegistryError) as exc_info:
            verify_integrity("a" * 64)
    assert isinstance(exc_info.value.__cause__, requests.exceptions.ConnectionError)


def test_get_document_wraps_contract_logic_error_in_registry_error():
    mock_contract = MagicMock()
    mock_contract.functions.documentLedger.return_value.call.side_effect = ContractLogicError(
        "execution reverted"
    )
    with patch("blockchain.registry_client.connect", return_value=(MagicMock(), mock_contract)):
        with pytest.raises(RegistryError) as exc_info:
            get_document("a" * 64)
    assert isinstance(exc_info.value.__cause__, ContractLogicError)
```

Note: `os` and `pytest` are already imported partway through the existing file (in the block that also imports `get_document`, `register_document`, `update_reputation`, `verify_integrity`), and `get_document`/`verify_integrity` are reused as-is by the new tests below — no need to re-import either.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_blockchain_registry_client.py -v`
Expected: the 5 new tests FAIL — `ImportError: cannot import name 'RegistryError'` (and `_send_transaction` import will also fail until the module is touched, but since `_send_transaction` already exists, only `RegistryError` is the missing name).

- [ ] **Step 3: Write the implementation**

Replace the full contents of `blockchain/registry_client.py` with:

```python
import json
import os
from pathlib import Path

import requests
from eth_account import Account
from web3 import Web3
from web3.contract import Contract
from web3.exceptions import ContractLogicError

DEPLOYMENT_PATH = (
    Path(__file__).resolve().parent.parent / "contracts" / "deployments" / "localhost.json"
)


class RegistryError(Exception):
    pass


def _doc_hash_to_bytes32(doc_hash: str) -> bytes:
    hexstr = doc_hash if doc_hash.startswith("0x") else "0x" + doc_hash
    return Web3.to_bytes(hexstr=hexstr)


def connect() -> tuple[Web3, Contract]:
    provider_uri = os.environ["WEB3_PROVIDER_URI"]
    try:
        w3 = Web3(Web3.HTTPProvider(provider_uri))
        with open(DEPLOYMENT_PATH) as f:
            deployment = json.load(f)
        contract = w3.eth.contract(address=deployment["address"], abi=deployment["abi"])
    except (requests.exceptions.RequestException, FileNotFoundError, ValueError) as exc:
        raise RegistryError(f"could not connect to blockchain node or read deployment: {exc}") from exc
    return w3, contract


def _send_transaction(w3: Web3, function_call, signing_key: str) -> dict:
    try:
        account = Account.from_key(signing_key)
        tx = function_call.build_transaction(
            {
                "from": account.address,
                "nonce": w3.eth.get_transaction_count(account.address),
                "gasPrice": w3.eth.gas_price,
                "chainId": w3.eth.chain_id,
            }
        )
        signed = account.sign_transaction(tx)
        tx_hash = w3.eth.send_raw_transaction(signed.rawTransaction)
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    except (requests.exceptions.RequestException, ContractLogicError, ValueError) as exc:
        raise RegistryError(f"blockchain transaction failed: {exc}") from exc
    return dict(receipt)


def register_document(doc_hash: str, ipfs_cid: str, signing_key: str | None = None) -> dict:
    w3, contract = connect()
    key = signing_key or os.environ["PRIVATE_KEY"]
    hash_bytes = _doc_hash_to_bytes32(doc_hash)
    function_call = contract.functions.registerDocument(hash_bytes, ipfs_cid)
    return _send_transaction(w3, function_call, key)


def update_reputation(doc_hash: str, delta: int, signing_key: str | None = None) -> dict:
    w3, contract = connect()
    key = signing_key or os.environ["PRIVATE_KEY"]
    hash_bytes = _doc_hash_to_bytes32(doc_hash)
    function_call = contract.functions.updateReputation(hash_bytes, delta)
    return _send_transaction(w3, function_call, key)


def verify_integrity(doc_hash: str) -> dict:
    w3, contract = connect()
    hash_bytes = _doc_hash_to_bytes32(doc_hash)
    try:
        exists, reputation, publisher = contract.functions.verifyIntegrity(hash_bytes).call()
    except (requests.exceptions.RequestException, ContractLogicError, ValueError) as exc:
        raise RegistryError(f"could not read document registry: {exc}") from exc
    return {"exists": exists, "reputation": reputation, "publisher": publisher}


def get_document(doc_hash: str) -> dict:
    w3, contract = connect()
    hash_bytes = _doc_hash_to_bytes32(doc_hash)
    try:
        publisher, content_hash, ipfs_cid, reputation_score, timestamp, is_registered = (
            contract.functions.documentLedger(hash_bytes).call()
        )
    except (requests.exceptions.RequestException, ContractLogicError, ValueError) as exc:
        raise RegistryError(f"could not read document registry: {exc}") from exc
    return {
        "publisher": publisher,
        "content_hash": "0x" + content_hash.hex(),
        "ipfs_cid": ipfs_cid,
        "reputation_score": reputation_score,
        "timestamp": timestamp,
        "is_registered": is_registered,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_blockchain_registry_client.py -v`
Expected: all fast tests PASS (the `@pytest.mark.slow` tests remain skipped by default per `addopts = ["-m", "not slow"]`).

- [ ] **Step 5: Update scripts/ingest.py and its tests to catch RegistryError instead of the now-obsolete raw exception types**

`scripts/ingest.py`'s `_resolve_chunk_cid()` currently catches `(IPFSError, requests.exceptions.ConnectionError, ContractLogicError)` around `get_document()`/`register_document()`. After Step 3, those functions raise `RegistryError` instead — the old except clause would silently stop catching anything real. Update the import line and the except clause. In `scripts/ingest.py`, change:

```python
import requests
from web3.exceptions import ContractLogicError
```

(both lines — remove them; neither name is used anywhere else in the file) and change:

```python
from blockchain.registry_client import get_document, register_document
```

to:

```python
from blockchain.registry_client import RegistryError, get_document, register_document
```

and change:

```python
    except (IPFSError, requests.exceptions.ConnectionError, ContractLogicError) as exc:
```

to:

```python
    except (IPFSError, RegistryError) as exc:
```

In `tests/test_scripts_ingest.py`, remove the now-unused imports (neither name is used anywhere else in the file):

```python
import requests
from web3.exceptions import ContractLogicError
```

and add `RegistryError` to the existing import from `blockchain.registry_client`, or add a new import line if none exists yet — check the file's current imports first, since it currently has no `blockchain.registry_client` import at all (it patches `scripts.ingest.get_document`/`scripts.ingest.register_document` instead):

```python
from blockchain.registry_client import RegistryError
```

Then update the two tests whose mocked side effects are now stale:

```python
def test_resolve_chunk_cid_get_document_connection_error_is_caught(
    mock_get_document, mock_upload_text, mock_verify_hash, mock_register_document
):
    mock_get_document.side_effect = RegistryError("node unreachable")

    result = _resolve_chunk_cid(_RECORD, source_file="solar.txt", publisher_key=PUBLISHER_1_KEY)

    assert result is None
    mock_upload_text.assert_not_called()
    mock_register_document.assert_not_called()
```

```python
def test_resolve_chunk_cid_register_document_contract_revert_is_caught(
    mock_get_document, mock_upload_text, mock_verify_hash, mock_register_document
):
    mock_get_document.return_value = {"is_registered": False}
    mock_upload_text.return_value = "QmFreshCID"
    mock_verify_hash.return_value = True
    mock_register_document.side_effect = RegistryError("execution reverted")

    result = _resolve_chunk_cid(_RECORD, source_file="solar.txt", publisher_key=PUBLISHER_1_KEY)

    assert result is None
```

Only the `@patch` decorators, function bodies, and imports change — leave the two functions' names, decorators, and every other test in the file untouched.

- [ ] **Step 6: Run the full fast suite to verify no regressions**

Run: `uv run pytest -v`
Expected: all previously-passing tests still PASS, including `tests/test_scripts_ingest.py`'s full file (this exercises `_resolve_chunk_cid`'s other existing tests too, e.g. the `IPFSError` and unlisted-`KeyError`-propagation cases, which must be unaffected).

- [ ] **Step 7: Commit**

```bash
git add blockchain/registry_client.py tests/test_blockchain_registry_client.py scripts/ingest.py tests/test_scripts_ingest.py
git commit -m "feat: add RegistryError wrapping to blockchain/registry_client.py"
```

---

### Task 2: GenerationError in generation/llm_client.py

**Files:**
- Modify: `generation/llm_client.py`
- Test: `tests/test_generation_llm_client.py`

**Interfaces:**
- Consumes: nothing new from Task 1.
- Produces: `GenerationError(Exception)`, importable as `from generation.llm_client import GenerationError`. `generate_blocking(prompt: str) -> str` and `generate_streaming(prompt: str) -> Iterator[str]` keep their signatures; they now raise `GenerationError` instead of raw `requests.exceptions.RequestException`. **Important for Task 4:** `generate_streaming` is a generator — the `requests.post` call (and therefore the `try/except`) only executes on the first `next()`/iteration, not at call time. `app/main.py` must wrap the *consumption* of the generator (the loop or `st.write_stream(...)` call) in `try/except GenerationError`, not just the call to `generate_streaming(prompt)`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_generation_llm_client.py`, right after the existing imports:

```python
import requests
```

(the file currently doesn't import `requests` directly). Then update the existing import line:

```python
from generation.llm_client import GenerationError, build_prompt, generate_blocking, generate_streaming
```

Append these tests at the end of the file, before the `@pytest.mark.slow` test:

```python
def test_generate_blocking_wraps_connection_error_in_generation_error():
    with patch(
        "generation.llm_client.requests.post",
        side_effect=requests.exceptions.ConnectionError("could not connect"),
    ):
        with pytest.raises(GenerationError) as exc_info:
            generate_blocking("hi")
    assert isinstance(exc_info.value.__cause__, requests.exceptions.ConnectionError)


def test_generate_streaming_wraps_connection_error_in_generation_error():
    with patch(
        "generation.llm_client.requests.post",
        side_effect=requests.exceptions.ConnectionError("could not connect"),
    ):
        with pytest.raises(GenerationError) as exc_info:
            list(generate_streaming("hi"))
    assert isinstance(exc_info.value.__cause__, requests.exceptions.ConnectionError)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_generation_llm_client.py -v`
Expected: FAIL — `ImportError: cannot import name 'GenerationError'`.

- [ ] **Step 3: Write the implementation**

In `generation/llm_client.py`, add the exception class after the constants:

```python
MODEL = "gemma4:12b"
TEMPERATURE = 0.1
MAX_NEW_TOKENS = 512


class GenerationError(Exception):
    pass
```

Replace `generate_blocking` and `generate_streaming` with:

```python
def generate_blocking(prompt: str) -> str:
    url = f"{_ollama_host()}/api/chat"
    try:
        response = requests.post(url, json=_request_body(prompt, stream=False))
    except requests.exceptions.RequestException as exc:
        raise GenerationError(f"could not reach Ollama at {url}: {exc}") from exc
    return response.json()["message"]["content"]


def generate_streaming(prompt: str) -> Iterator[str]:
    url = f"{_ollama_host()}/api/chat"
    try:
        response = requests.post(url, json=_request_body(prompt, stream=True), stream=True)
    except requests.exceptions.RequestException as exc:
        raise GenerationError(f"could not reach Ollama at {url}: {exc}") from exc
    for line in response.iter_lines():
        if not line:
            continue
        chunk = json.loads(line)
        content = chunk["message"]["content"]
        if content:
            yield content
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_generation_llm_client.py -v`
Expected: all fast tests PASS.

- [ ] **Step 5: Commit**

```bash
git add generation/llm_client.py tests/test_generation_llm_client.py
git commit -m "feat: add GenerationError wrapping to generation/llm_client.py"
```

---

### Task 3: app/ui_helpers.py

**Files:**
- Create: `app/__init__.py` (empty)
- Create: `app/ui_helpers.py`
- Test: `tests/test_app_ui_helpers.py`

**Interfaces:**
- Consumes: `RegistryError`/`get_document` from `blockchain.registry_client` (Task 1), `IPFSError`/`verify_hash` from `storage.ipfs_client` (existing), `standard_rag_answer_context` from `retrieval.standard_rag` (existing, `-> list[dict]`, accepts `collection=`), `trust_weighted_rerank` from `retrieval.rerank` (existing, `-> list[dict]`, no `collection` param).
- Produces:
  - `STANDARD_RAG_MODE = "Standard RAG"`, `DERAG_MODE = "DeRAG"` — the exact toggle label strings, consumed by `app/main.py` in Task 4.
  - `format_hash_short(h: str, n: int = 10) -> str`
  - `format_audit_row(doc_hash: str) -> dict` — success shape `{"doc_hash_short", "ipfs_cid", "publisher", "reputation", "verified": bool}`; IPFS-only failure shape `{"doc_hash_short", "ipfs_cid", "publisher", "reputation", "verified": None, "verify_error": str}`; registry failure shape `{"doc_hash_short", "error": str}`.
  - `get_context_docs(query_text: str, mode: str, collection=None) -> list[dict]`

- [ ] **Step 1: Write the failing tests**

Create `app/__init__.py`:

```python
```

(empty file, matching the sibling packages' convention)

Create `tests/test_app_ui_helpers.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_app_ui_helpers.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.ui_helpers'`.

- [ ] **Step 3: Write the implementation**

Create `app/ui_helpers.py`:

```python
from blockchain.registry_client import RegistryError, get_document
from retrieval.rerank import trust_weighted_rerank
from retrieval.standard_rag import standard_rag_answer_context
from storage.ipfs_client import IPFSError, verify_hash

STANDARD_RAG_MODE = "Standard RAG"
DERAG_MODE = "DeRAG"


def format_hash_short(h: str, n: int = 10) -> str:
    if len(h) <= n + 4:
        return h
    return f"{h[:n]}...{h[-4:]}"


def format_audit_row(doc_hash: str) -> dict:
    try:
        doc = get_document(doc_hash)
    except RegistryError as exc:
        return {
            "doc_hash_short": format_hash_short(doc_hash),
            "error": f"registry unreachable: {exc}",
        }

    row = {
        "doc_hash_short": format_hash_short(doc_hash),
        "ipfs_cid": doc["ipfs_cid"],
        "publisher": doc["publisher"],
        "reputation": doc["reputation_score"],
    }
    try:
        row["verified"] = verify_hash(doc["ipfs_cid"], doc_hash)
    except IPFSError as exc:
        row["verified"] = None
        row["verify_error"] = f"IPFS unreachable: {exc}"
    return row


def get_context_docs(query_text: str, mode: str, collection=None) -> list[dict]:
    if mode == STANDARD_RAG_MODE:
        return standard_rag_answer_context(query_text, collection=collection)
    if mode == DERAG_MODE:
        return trust_weighted_rerank(query_text)
    raise ValueError(f"unknown mode: {mode!r}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_app_ui_helpers.py -v`
Expected: all tests PASS.

- [ ] **Step 5: Run the full fast suite to check for regressions**

Run: `uv run pytest -v`
Expected: all fast tests PASS (no `-m slow`), no regressions in previously-passing modules.

- [ ] **Step 6: Commit**

```bash
git add app/__init__.py app/ui_helpers.py tests/test_app_ui_helpers.py
git commit -m "feat: add app/ui_helpers.py with audit row formatting and mode dispatch"
```

---

### Task 4: app/main.py — the Streamlit page

**Files:**
- Modify: `app/main.py`

**Interfaces:**
- Consumes: `RegistryError` (Task 1), `GenerationError` (Task 2), `STANDARD_RAG_MODE`/`DERAG_MODE`/`format_audit_row`/`get_context_docs` (Task 3), `init_store` from `retrieval.vector_store` (existing), `build_prompt`/`generate_streaming` from `generation.llm_client` (existing).
- Produces: the running Streamlit page. No other module imports from `app/main.py`.

This task is not TDD — Streamlit apps aren't unit-testable, and all the logic worth unit-testing already lives in `app/ui_helpers.py` (Task 3). Verification is manual, per the Global Constraints' definition of done.

- [ ] **Step 1: Write app/main.py**

Replace the full contents of `app/main.py` with:

```python
import chromadb.errors
import huggingface_hub.errors
import streamlit as st

from app.ui_helpers import DERAG_MODE, STANDARD_RAG_MODE, format_audit_row, get_context_docs
from generation.llm_client import GenerationError, build_prompt, generate_streaming
from retrieval.vector_store import init_store

st.set_page_config(page_title="DeRAG", page_icon="🔗")

st.title("DeRAG")
st.caption("Decentralized, Verifiable RAG with Trust-Weighted Ranking")


@st.cache_resource
def get_collection():
    return init_store()


query_text = st.text_input("Ask a question")
mode = st.radio("Retrieval mode", [STANDARD_RAG_MODE, DERAG_MODE])

if query_text:
    try:
        collection = get_collection()
        context_docs = get_context_docs(query_text, mode, collection=collection)
    except (chromadb.errors.ChromaError, huggingface_hub.errors.HfHubHTTPError, OSError) as exc:
        st.error(f"Vector store unavailable: {exc}")
        context_docs = None

    if context_docs is not None:
        if not context_docs:
            st.warning("No matching context found for this query.")
        else:
            prompt = build_prompt(query_text, context_docs)
            try:
                st.write_stream(generate_streaming(prompt))
            except GenerationError as exc:
                st.error(f"LLM unreachable — is Ollama running? ({exc})")

            with st.sidebar:
                st.subheader("Audit trail")
                for doc in context_docs[:3]:
                    doc_hash = doc["metadata"]["doc_hash"]
                    row = format_audit_row(doc_hash)
                    st.markdown(f"**{row['doc_hash_short']}**")
                    if "error" in row:
                        st.error(row["error"])
                    else:
                        st.write(f"IPFS CID: {row['ipfs_cid']}")
                        st.write(f"Publisher: {row['publisher']}")
                        st.write(f"Reputation: {row['reputation']}")
                        if row["verified"] is True:
                            st.success("✅ verified")
                        elif row["verified"] is False:
                            st.warning("⚠️ unverified")
                        else:
                            st.info(f"❔ verification unknown: {row.get('verify_error', '')}")
                    st.divider()
```

- [ ] **Step 2: Manual verification — happy path**

Start all four services (Hardhat node + deployed contract, IPFS daemon, `ollama serve` with `gemma4:12b` pulled, and ChromaDB is local-file so needs no separate start), with data already ingested via Milestone 5's `scripts/ingest.py`. Then:

Run: `uv run streamlit run app/main.py`

In the browser: type a query, confirm the "Retrieval mode" toggle shows "Standard RAG" and "DeRAG", confirm the answer streams into the main area, confirm the sidebar shows up to 3 audit rows each with a truncated hash, IPFS CID, publisher, reputation, and a ✅/⚠️/❔ badge. Toggle to the other mode and re-submit the same query; confirm the sidebar updates.

- [ ] **Step 3: Manual verification — degraded paths**

With the app still running, stop the Hardhat node (Ctrl+C the `npx hardhat node` process) and submit a query: confirm the sidebar shows a registry-unreachable error per row (not a crash, not a raw traceback) while the main answer area still streams normally. Restart Hardhat. Stop the IPFS daemon and submit a query: confirm the sidebar rows still show publisher/reputation but the badge shows "❔ verification unknown". Restart IPFS. Stop `ollama serve` and submit a query: confirm the main area shows "LLM unreachable — is Ollama running?" instead of crashing, while the sidebar (which doesn't depend on Ollama) still renders normally.

- [ ] **Step 4: Commit**

```bash
git add app/main.py
git commit -m "feat: build DeRAG Streamlit page with mode toggle and sidebar audit trail"
```

---

## Definition of Done (Phase 1)

- `uv run pytest -v` (fast suite) passes with no regressions, including the new tests in Tasks 1-3.
- With all four services running and data ingested, `streamlit run app/main.py` supports: typing a query, toggling modes, seeing a streamed grounded answer, and seeing a sidebar audit trail with working live verification badges (Step 2 above).
- Stopping any one service produces a clean in-UI error in the relevant section, not a crash (Step 3 above).
