# Milestone 4 — IPFS Client Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `storage/ipfs_client.py`, a small HTTP client for a local Kubo IPFS daemon that uploads/fetches text and verifies fetched content against Milestone 1's `hash_chunk`, fully isolated from blockchain/Chroma.

**Architecture:** A single module (`storage/ipfs_client.py`) with four functions built directly on `requests` (no `ipfshttpclient` dependency, per Milestone 0's spec). Upload goes through the Kubo RPC API (`POST /api/v0/add`, multipart); fetch goes through the read-only public gateway (`GET /ipfs/<cid>`) rather than the API's `/api/v0/cat`, because the gateway is a plain GET (no odd "POST with a query-string arg" shape for a read), and it's the same read path a later audit-trail UI (Milestone 8) would use. All failures — bad HTTP status *and* connection-level failures (daemon not running) — are caught and re-raised as a single `IPFSError` so callers never see a raw `requests` exception or stack trace.

**Tech Stack:** `requests` (already a project dependency), stdlib `hashlib` via the existing `pipeline.hashing.hash_chunk`, `pytest` with the existing `slow` marker for daemon-dependent tests.

**Spec:** This plan implements the design confirmed in-session against a real local Kubo 0.29.0 daemon (see chat transcript / `DECISIONS.md` Milestone 4 entry once written). No separate spec file — findings are baked directly into the tasks below, each backed by an actual command run against the real daemon during design.

## Global Constraints

- `IPFS_API_URL` in `.env.example` is a multiaddr string (`/ip4/127.0.0.1/tcp/5001`), not an HTTP URL — must be parsed, not used directly.
- No `ipfshttpclient` dependency — use `requests` directly (Milestone 0 spec decision).
- `storage/ipfs_client.py` must not import anything from `blockchain/` or `retrieval/` (vector store) — its only intra-project dependency is `pipeline.hashing.hash_chunk`.
- Kubo's `/api/v0/*` endpoints require `POST` for all calls, including reads — confirmed live (`GET` returns HTTP 405).
- Kubo daemon defaults to CIDv0 (`Qm...`) — confirmed live, matches the spec's example CID format, so no `cid-version` override is used anywhere.
- `/api/v0/add` response shape confirmed live: `{"Name": "...", "Hash": "Qm...", "Size": "29"}` — `Size` is a JSON **string**, not a number (irrelevant here since only `Hash` is read, but do not assume `Size` is numeric if ever touched).
- Malformed or well-formed-but-missing CIDs return HTTP 500 immediately with a JSON `{"Message": ...}` body (confirmed live, both against a real daemon in offline mode) — no network hang to guard against on that path, but a `requests` timeout is still used defensively.
- A daemon that isn't running produces `requests.exceptions.ConnectionError` (subclass of `requests.exceptions.RequestException`) — confirmed live by pointing a client at the real daemon's port after killing it.
- Tests that need a live daemon are marked `@pytest.mark.slow` (already excluded by `pyproject.toml`'s `addopts = ["-m", "not slow"]`; the marker is already registered).

---

## File Structure

- **Create `storage/ipfs_client.py`** — the client: `_api_base_url()`, `_gateway_base_url()`, `IPFSError`, `upload_text()`, `fetch_text()`, `verify_hash()`.
- **Create `tests/test_storage_ipfs_client.py`** — unit test for the multiaddr parser (fast, no daemon) + `@pytest.mark.slow` integration tests (need a live `ipfs daemon`).
- **Create `scripts/demo_ipfs_roundtrip.py`** — CLI demo: upload → CID → fetch → hash match, against a real daemon.
- **Modify `DECISIONS.md`** — append a "Milestone 4 — IPFS Client" section.
- **Modify `README.md`** — add a Milestone 4 paragraph mirroring the existing Milestone 3 paragraph's style (daemon startup instructions).

---

### Task 1: `_api_base_url()` — multiaddr parser

**Files:**
- Create: `storage/ipfs_client.py`
- Test: `tests/test_storage_ipfs_client.py`

**Interfaces:**
- Produces: `_api_base_url() -> str` — reads `IPFS_API_URL` from the environment, returns `"http://<host>:<port>"`. Later tasks call this to build request URLs.

This is a pure function, no daemon needed — fast test, no `slow` marker.

- [ ] **Step 1: Write the failing test**

Create `tests/test_storage_ipfs_client.py` with this content:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_storage_ipfs_client.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'storage.ipfs_client'`

- [ ] **Step 3: Write minimal implementation**

Create `storage/ipfs_client.py`:

```python
import os


def _api_base_url() -> str:
    multiaddr = os.environ["IPFS_API_URL"]
    _, protocol, host, transport, port = multiaddr.split("/")
    assert protocol == "ip4" and transport == "tcp"
    return f"http://{host}:{port}"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_storage_ipfs_client.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add storage/ipfs_client.py tests/test_storage_ipfs_client.py
git commit -m "feat: parse IPFS_API_URL multiaddr into an HTTP base URL"
```

---

### Task 2: `IPFSError` + `upload_text()`

**Files:**
- Modify: `storage/ipfs_client.py`
- Test: `tests/test_storage_ipfs_client.py`

**Interfaces:**
- Consumes: `_api_base_url()` from Task 1.
- Produces: `IPFSError(Exception)`; `upload_text(text: str) -> str` — returns the CID string. Raises `IPFSError` on any failure (bad status or connection failure). Later tasks (`fetch_text`, `verify_hash`, tests) use `IPFSError` and call `upload_text` to seed content for round-trip tests.

This task's tests are `@pytest.mark.slow` — they need a live daemon to actually upload.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_storage_ipfs_client.py`:

```python
from storage.ipfs_client import IPFSError, upload_text


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest -m slow tests/test_storage_ipfs_client.py -v`
Expected: FAIL — `ImportError: cannot import name 'IPFSError'` (requires a live daemon started with `ipfs daemon` beforehand for the parts that would otherwise run; the import failure happens before that matters).

- [ ] **Step 3: Write minimal implementation**

Add to `storage/ipfs_client.py`:

```python
import requests


class IPFSError(Exception):
    pass


def upload_text(text: str) -> str:
    url = f"{_api_base_url()}/api/v0/add"
    try:
        response = requests.post(
            url,
            files={"file": text.encode("utf-8")},
            timeout=10,
        )
    except requests.exceptions.RequestException as exc:
        raise IPFSError(f"could not reach IPFS daemon at {url}: {exc}") from exc

    if response.status_code != 200:
        raise IPFSError(
            f"IPFS add failed with status {response.status_code}: {response.text}"
        )

    return response.json()["Hash"]
```

- [ ] **Step 4: Run test to verify it passes**

Start a local daemon first: `ipfs daemon` (in a separate terminal, or background it).
Run: `uv run pytest -m slow tests/test_storage_ipfs_client.py -v`
Expected: PASS (4 passed — the two Task 1 fast tests still pass under `-m slow`'s superset selection is not guaranteed; run without `-m` to confirm all pass: `uv run pytest tests/test_storage_ipfs_client.py -v -m ""`)

- [ ] **Step 5: Commit**

```bash
git add storage/ipfs_client.py tests/test_storage_ipfs_client.py
git commit -m "feat: add upload_text with IPFSError wrapping HTTP and connection failures"
```

---

### Task 3: `fetch_text()` via the public gateway

**Files:**
- Modify: `storage/ipfs_client.py`
- Test: `tests/test_storage_ipfs_client.py`

**Interfaces:**
- Consumes: `_api_base_url()` from Task 1 (to derive the gateway host), `IPFSError` and `upload_text()` from Task 2.
- Produces: `_gateway_base_url() -> str`; `fetch_text(cid: str) -> str`. Raises `IPFSError` on any failure. Later tasks (`verify_hash`, demo script) call `fetch_text`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_storage_ipfs_client.py`:

```python
from storage.ipfs_client import fetch_text


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest -m slow tests/test_storage_ipfs_client.py -v`
Expected: FAIL — `ImportError: cannot import name 'fetch_text'`

- [ ] **Step 3: Write minimal implementation**

Add to `storage/ipfs_client.py`:

```python
def _gateway_base_url() -> str:
    host = _api_base_url().split("://")[1].split(":")[0]
    return f"http://{host}:8080"


def fetch_text(cid: str) -> str:
    url = f"{_gateway_base_url()}/ipfs/{cid}"
    try:
        response = requests.get(url, timeout=10)
    except requests.exceptions.RequestException as exc:
        raise IPFSError(f"could not reach IPFS gateway at {url}: {exc}") from exc

    if response.status_code != 200:
        raise IPFSError(
            f"IPFS fetch failed with status {response.status_code} for CID {cid}: {response.text}"
        )

    return response.content.decode("utf-8")
```

- [ ] **Step 4: Run test to verify it passes**

Run (with `ipfs daemon` running): `uv run pytest tests/test_storage_ipfs_client.py -v -m ""`
Expected: PASS (all tests so far)

- [ ] **Step 5: Commit**

```bash
git add storage/ipfs_client.py tests/test_storage_ipfs_client.py
git commit -m "feat: add fetch_text via the IPFS public gateway with IPFSError wrapping"
```

---

### Task 4: `verify_hash()` and the verbatim-hash proof test

**Files:**
- Modify: `storage/ipfs_client.py`
- Test: `tests/test_storage_ipfs_client.py`

**Interfaces:**
- Consumes: `fetch_text()` from Task 3, `upload_text()` from Task 2, `pipeline.hashing.hash_chunk(chunk: str) -> str` (existing function, `hashlib.sha256(chunk.encode("utf-8")).hexdigest()` — import it, do not reimplement).
- Produces: `verify_hash(cid: str, expected_hash: str) -> bool`. This is the function Milestone 8's audit trail UI calls per source.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_storage_ipfs_client.py`:

```python
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
    already_normalized_text = "the quick brown fox jumps over the lazy dog"
    hash_before_upload = hash_chunk(already_normalized_text)

    cid = upload_text(already_normalized_text)
    fetched_text = fetch_text(cid)
    hash_after_fetch = hash_chunk(fetched_text)

    assert hash_after_fetch == hash_before_upload
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest -m slow tests/test_storage_ipfs_client.py -v`
Expected: FAIL — `ImportError: cannot import name 'verify_hash'`

- [ ] **Step 3: Write minimal implementation**

Add to `storage/ipfs_client.py`:

```python
from pipeline.hashing import hash_chunk


def verify_hash(cid: str, expected_hash: str) -> bool:
    fetched_text = fetch_text(cid)
    return hash_chunk(fetched_text) == expected_hash
```

- [ ] **Step 4: Run test to verify it passes**

Run (with `ipfs daemon` running): `uv run pytest tests/test_storage_ipfs_client.py -v -m ""`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add storage/ipfs_client.py tests/test_storage_ipfs_client.py
git commit -m "feat: add verify_hash and prove Milestone 1's verbatim-hash contract over a real IPFS round trip"
```

---

### Task 5: Demo script, README, DECISIONS.md

**Files:**
- Create: `scripts/demo_ipfs_roundtrip.py`
- Modify: `README.md`
- Modify: `DECISIONS.md`

**Interfaces:**
- Consumes: `upload_text`, `fetch_text`, `verify_hash` from `storage/ipfs_client.py`, `hash_chunk` from `pipeline/hashing.py`. No new interfaces produced (this is the terminal, documentation-and-demo task).

- [ ] **Step 1: Write the demo script**

Create `scripts/demo_ipfs_roundtrip.py` (mirrors `scripts/demo_ingest_prep.py`'s `sys.path` + `main()` shape):

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.hashing import hash_chunk
from storage.ipfs_client import fetch_text, upload_text, verify_hash


def main() -> None:
    text = "DeRAG Milestone 4: this chunk travels through real IPFS."
    expected_hash = hash_chunk(text)
    print(f"Original text: {text!r}")
    print(f"Pre-upload hash: {expected_hash}\n")

    cid = upload_text(text)
    print(f"Uploaded to IPFS, CID: {cid}\n")

    fetched = fetch_text(cid)
    print(f"Fetched text: {fetched!r}")
    assert fetched == text, "fetched text does not match original"

    is_valid = verify_hash(cid, expected_hash)
    print(f"verify_hash(cid, pre-upload hash): {is_valid}")
    assert is_valid, "hash verification failed after real IPFS round trip"

    print("\nRound trip succeeded: upload -> CID -> fetch -> hash match.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the demo against a real daemon**

Run (with `ipfs daemon` running): `uv run python scripts/demo_ipfs_roundtrip.py`
Expected: prints the CID, fetched text, and `verify_hash(...): True`, ending with the success line. No exceptions.

- [ ] **Step 3: Update README.md**

Add a paragraph after the existing Milestone 3 paragraph (matching its style):

```markdown

Milestone 4: IPFS storage client (`storage/ipfs_client.py`) implemented and
tested against a real local Kubo daemon — text upload/fetch via the daemon's
HTTP API and public gateway, plus `verify_hash()` (used by Milestone 8's
audit trail UI) which proves fetched content still hashes identically to
what was uploaded, using Milestone 1's exact `hash_chunk`. Start a local
daemon before running the slow suite or the demo:

```
ipfs daemon
```
```

- [ ] **Step 4: Update DECISIONS.md**

Add a new section after the Milestone 3 section, following the existing style (bullet list, bold lead-ins):

```markdown

## Milestone 4 — IPFS Client (2026-08-17)

- **Multiaddr-to-URL derivation, verified against a real daemon**:
  `IPFS_API_URL` (`.env.example`: `/ip4/127.0.0.1/tcp/5001`) is parsed by
  splitting on `/` into `["", "ip4", host, "tcp", port]` and building
  `http://{host}:{port}`. Confirmed against a real local Kubo 0.29.0
  daemon rather than assumed — `POST http://127.0.0.1:5001/api/v0/version`
  against the derived URL returned `200` with the daemon's real version
  JSON.
- **Fetch via the public gateway (`GET :8080/ipfs/<cid>`), not the API's
  `/api/v0/cat`**: both were confirmed working against the real daemon,
  but the gateway is a plain GET with no request body, matches the
  `.env.example` comment's implied convention ("gateway runs on port
  8080, no env var needed" — same host as the API, fixed port), and is
  the same read-only surface Milestone 8's audit trail UI would use.
  Upload still goes through the API's `/api/v0/add` (`POST`, multipart)
  since the gateway has no write path. Also confirmed live: Kubo's
  `/api/v0/*` endpoints reject `GET` with `405` — every API call,
  including reads, must be `POST`.
- **CID version confirmed CIDv0 by default** (`Qm...`), verified against
  the real daemon's `/api/v0/add` response with no `cid-version`
  parameter — matches the spec's example CID format exactly, so
  `storage/ipfs_client.py` never overrides `cid-version`.
- **`/api/v0/add` response shape confirmed live**:
  `{"Name": "...", "Hash": "Qm...", "Size": "29"}` — `Size` is a JSON
  string, not a number (only `Hash` is read, so this didn't require
  handling, but is noted for anyone touching `Size` later).
- **`IPFSError` wraps both failure classes**: non-2xx HTTP responses
  (confirmed live — a malformed or well-formed-but-missing CID returns
  `500` with a JSON `{"Message": ...}` body) and connection-level
  failures from a daemon that isn't running at all (confirmed live —
  killing the daemon and retrying produces
  `requests.exceptions.ConnectionError`, caught via its parent
  `requests.exceptions.RequestException`). Both `upload_text` and
  `fetch_text` catch both classes so no raw `requests` exception or
  stack trace ever reaches a caller.
- **Milestone 1's verbatim-hashing contract tested end-to-end through a
  real IPFS round trip, not just asserted**: `hash_chunk` is called on
  a mixed-case/irregular-whitespace string (the kind `normalize_text`
  would already have normalized before `hash_chunk` sees it) *before*
  `upload_text`, and again on the *fetched* text after a real
  `upload_text` → `fetch_text` round trip against the live daemon; the
  two hashes are asserted equal
  (`test_verbatim_hash_survives_real_ipfs_round_trip`). This is
  distinct from — and in addition to — the generic
  upload/fetch-returns-identical-text round-trip test.
```

- [ ] **Step 5: Run the fast suite to confirm no regressions**

Run: `uv run pytest`
Expected: fast suite green, IPFS `slow` tests deselected (same as Milestone 3's pattern).

- [ ] **Step 6: Commit**

```bash
git add scripts/demo_ipfs_roundtrip.py README.md DECISIONS.md
git commit -m "docs: add IPFS round-trip demo and Milestone 4 DECISIONS.md entry"
```
