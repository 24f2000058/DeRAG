# Milestone 8 — Streamlit Audit UI, Design

**Goal:** Build the DeRAG Streamlit page (`app/main.py`) that lets a user
query, toggle between Standard RAG and DeRAG (trust-weighted) retrieval,
see a streamed grounded LLM answer, and see a sidebar audit trail proving
provenance for each source used. Along the way, close the last gap in the
project's exception-wrapping pattern: `blockchain/registry_client.py` and
`generation/llm_client.py` are the two remaining external-service clients
that let raw `requests`/`web3` exceptions escape into caller code, and
this is the first milestone where UI code calls either of them directly.

## Verified upstream contracts (re-confirmed by source inspection, not memory)

- `retrieval/standard_rag.py`: `standard_rag_answer_context(query_text, k=3, collection=None) -> list[dict]`. Thin wrapper on `vector_store.query()`; returns `{chroma_id, text, similarity, metadata}` where `metadata` has `source_file, chunk_id, ipfs_cid, doc_hash`.
- `retrieval/rerank.py`: `trust_weighted_rerank(query_text, k_retrieve=10, k_return=3, alpha=0.7) -> list[dict]`. **No `collection` param.** Each result is the base hit plus `exists, reputation, normalized_reputation, score` — the literal key is `"score"` (the `hybrid_score()` helper is the function name, not the dict key).
- `blockchain/registry_client.py`: `verify_integrity(doc_hash) -> {"exists", "reputation", "publisher"}`; `get_document(doc_hash) -> {"publisher", "content_hash", "ipfs_cid", "reputation_score", "timestamp", "is_registered"}`. Currently wraps nothing — every public function can raise raw `requests.exceptions.RequestException`, `FileNotFoundError` (missing `contracts/deployments/localhost.json`), `KeyError` (missing env var), or web3 `ContractLogicError`/`ValueError` (revert).
- `storage/ipfs_client.py`: `IPFSError(Exception)`, the pattern to match. `verify_hash(cid, expected_hash) -> bool` (raises `IPFSError` via `fetch_text()` if the gateway is unreachable or returns non-200).
- `generation/llm_client.py`: `generate_streaming(prompt) -> Iterator[str]`, `generate_blocking(prompt) -> str`, `build_prompt(query, context_docs) -> str`. Currently wraps nothing — both generate functions can raise raw `requests.exceptions.RequestException`. `think: False` is already set in `_request_body()`, so the UI never needs to handle a separate reasoning channel.
- Streamlit 1.59.2 is installed; `st.cache_resource` and `st.cache_data` both exist and are the current caching API.
- `pipeline/embeddings.py`'s `_get_model()` is already `@lru_cache(maxsize=1)` at module level — process-wide, not per-Streamlit-session, so no further caching needed there.

## 1. Exception wrapping — three clients, one pattern

Add two new exception classes, each matching `IPFSError`'s shape exactly (`class XError(Exception): pass`):

- **`RegistryError`** in `blockchain/registry_client.py`. Wrap `connect()` (covers the `open(DEPLOYMENT_PATH)` `FileNotFoundError` and `Web3(...)` construction) and both call-site classes — `_send_transaction`'s `w3.eth.*` calls and the `.call()` invocations in `verify_integrity`/`get_document` — in `try/except (requests.exceptions.RequestException, FileNotFoundError, ContractLogicError, ValueError) as exc: raise RegistryError(f"...") from exc`. `KeyError` from a missing env var (`WEB3_PROVIDER_URI`, `PRIVATE_KEY`) is deliberately **not** wrapped — that's a setup mistake to fail loudly on, not a live-service condition to degrade around. Since `register_document`, `update_reputation`, `verify_integrity`, and `get_document` all route through `connect()` plus one call site, wrapping those two locations covers all four public functions.

- **`GenerationError`** in `generation/llm_client.py`. Wrap the `requests.post(...)` call in both `generate_blocking` and `generate_streaming` in `try/except requests.exceptions.RequestException as exc: raise GenerationError(f"could not reach Ollama at {url}: {exc}") from exc`. `app/main.py` catches `GenerationError`, not the raw `requests` exception — so a future internal refactor of `llm_client.py` (different HTTP library, added retry logic) can't silently break the UI's error handling with no test noticing.

- Regression tests extend (not replace) `tests/test_blockchain_registry_client.py` and `tests/test_generation_llm_client.py`: mock the underlying raw calls to raise each exception type, assert the wrapped type is raised instead, and assert `.__cause__` is the original exception.

## 2. app/ui_helpers.py — pure functions, unit tested

- `format_hash_short(h: str, n: int = 10) -> str` — truncates for display, e.g. `"a3f9c21b8e...d41f"`.
- `format_audit_row(doc_hash: str) -> dict` — the sidebar's per-source data shape. Calls `registry_client.get_document(doc_hash)` and, only if that succeeds, `ipfs_client.verify_hash(cid, doc_hash)`, kept as **separate** try/excepts so a registry failure and an IPFS-only failure degrade independently:
  - `get_document()` raises `RegistryError` → row is `{"doc_hash_short": ..., "error": "registry unreachable: ..."}` (nothing else to show).
  - `get_document()` succeeds but `verify_hash()` raises `IPFSError` → row is the full success shape with `"verified": None` and `"verify_error": "..."` instead of a `True`/`False` badge — publisher/reputation/cid are still shown since we have them.
  - Both succeed → `{"doc_hash_short", "ipfs_cid", "publisher", "reputation", "verified": bool}`.
- Mode dispatch: `get_context_docs(query_text: str, mode: str, collection=None) -> list[dict]`, mapping `"Standard RAG"` → `standard_rag_answer_context(query_text, collection=collection)` and `"DeRAG"` → `trust_weighted_rerank(query_text)` (no `collection` param exists on this one — accepted, see below).

All of this is unit-testable by mocking `registry_client`/`ipfs_client`/`standard_rag`/`rerank` — no live services required.

## 3. app/main.py — the Streamlit page

- `st.set_page_config` (already present), title/caption (already present).
- `@st.cache_resource`-wrapped `get_collection()` → `init_store()`, called once per process. Used for Standard RAG mode only, passed as `collection=` into `standard_rag_answer_context`. DeRAG mode's `trust_weighted_rerank()` has no `collection` param in its current (locked, Milestone 6) signature, so it always reopens Chroma internally on every call — accepted as-is: it's a local SQLite/disk hit, not a network round-trip, so the caching payoff is marginal and not worth touching a locked interface for.
- Query input (`st.text_input`) + mode toggle (`st.radio(["Standard RAG", "DeRAG"])`, matching the spec's exact wording).
- On submit: `get_context_docs()` wrapped in `try/except (chromadb.errors.ChromaError, huggingface_hub.errors.HfHubHTTPError, OSError) as exc` → inline error in the main answer area (e.g. "Vector store unavailable"). No new wrapper class here, deliberately: Chroma (local persistent directory) and the embedding model (in-process, cached after first load via `pipeline/embeddings.py`'s `@lru_cache`) aren't external services with an independent up/down lifecycle the way IPFS/Hardhat/Ollama are, so a fourth `XError` wrapper doesn't fit the pattern — but the exception types are still named explicitly rather than a bare `except Exception`, per this project's established discipline. Verified live against the installed versions: a bad Chroma path raises a `chromadb.errors.ChromaError` subclass (confirmed `InternalError` on a permission-denied path), and a missing/unreachable embedding model raises `huggingface_hub.errors` types (confirmed `RepositoryNotFoundError` for an unknown model) or a plain `OSError` for local cache/disk issues. Then `build_prompt()`, then `st.write_stream(generate_streaming(prompt))` wrapped in `try/except GenerationError` → "LLM unreachable — is Ollama running?" in the answer area, not a crash.
- `st.sidebar`: for each of the (up to 3) `context_docs`, call `format_audit_row(doc["metadata"]["doc_hash"])` and render doc_hash (short), ipfs_cid, publisher, reputation, and a badge — ✅ verified / ⚠️ unverified (verify_hash returned False) / ❔ unknown (`verified is None`, IPFS down) / ❌ error (registry down, row collapsed). Each row's failure is independent of the others and of the main answer area.

## Testing

- TDD for everything in `ui_helpers.py` and the new error classes (pure functions, mockable).
- `app/main.py` itself is not unit-tested (Streamlit); manual verification per the Phase 1 definition of done — run all four services, exercise both toggle modes, then stop each service one at a time and confirm a clean in-UI error in the relevant section rather than a crash.

## Out of scope for this milestone

- Changing `trust_weighted_rerank()`'s signature to accept a `collection` param (would touch a locked Milestone 6 interface for marginal caching gain).
- Any design/visual polish — that's Phase 2 (`impeccable`), after this is functionally verified.
