# Milestone 2 — Vector Store & Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up a local-persistent ChromaDB vector store and a thin "Standard RAG" retrieval layer on top of Milestone 1's `ingest_prep()` output, with no blockchain/IPFS involvement (those get real values in Milestones 3-5).

**Architecture:** Two new modules under `retrieval/`. `vector_store.py` owns all Chroma interaction (client creation, upsert, similarity query); `standard_rag.py` is a one-function wrapper that calls `vector_store.query()` with a fixed `k=3` and returns its result verbatim — no re-ranking (Milestone 6 adds a re-ranking sibling later against the same `vector_store.query()`). Chroma document IDs are Milestone 1's `record["hash"]` (sha256 of the normalized chunk text) used directly, not a new ID scheme, because it's already unique per content and turns re-ingestion of identical content into a natural upsert no-op.

**Tech Stack:** `chromadb==1.5.9` (local `PersistentClient`), `pipeline/embeddings.py` (Milestone 1, `embed()`), `python-dotenv` (already a dependency, currently unused — first consumer here), `pytest` + `tmp_path` for isolated test storage.

**Spec:** No separate design doc — this plan was authorized directly via `superpowers:brainstorming` (bounded path) in this conversation. The locked contracts it depends on: Milestone 1's `ingest_prep()` record shape `{"chunk_id": int, "text": str, "hash": str, "embedding": list[float]}` (`DECISIONS.md`, "Milestone 1" section) and the four-field Chroma metadata schema `source_file`/`chunk_id`/`ipfs_cid`/`doc_hash` agreed in this conversation.

## Global Constraints

- Chroma document ID = `record["hash"]` directly. Never invent a separate ID scheme.
- Chroma collection distance space = cosine, set via `configuration={"hnsw": {"space": "cosine"}}` at `get_or_create_collection()` time — verified against the real `chromadb==1.5.9` install (not assumed from training data): `collection.configuration_json` echoes `{"hnsw": {"space": "cosine", ...}}` back, and two orthogonal unit vectors returned `distance=1.0` (cosine), not `2.0` (squared-L2).
- `embedding_function=None` on every `get_or_create_collection()` call — verified that Chroma's default ONNX auto-embed is genuinely disabled by this (calling `add()` with `documents=` but no `embeddings=` raises `ValueError: You must provide an embedding function...`), not silently substituted.
- Chroma metadata per record is exactly four keys: `source_file` (str), `chunk_id` (int, Milestone 1's per-call value, unchanged), `ipfs_cid` (`""` empty-string placeholder — Task 2 discovered Chroma silently drops `None`-valued metadata entries from storage entirely, so `""` is the ruling: it's the only "no value yet" placeholder that survives a raw Chroma round-trip, including for future code that reads the collection directly rather than through `vector_store.query()`), `doc_hash` (`record["hash"]`). No other fields (no author/publisher — that's on-chain in Milestone 3).
- Use `collection.upsert(...)`, never `collection.add(...)`, for storing records — `add()` raises on a duplicate ID, `upsert()` overwrites, which is required for the doc_hash-as-ID idempotency property.
- All new tests use an isolated Chroma persistent directory per test (pytest `tmp_path` fixture) — never touch the real `./vector_store` directory or leak state between tests.
- Query similarity = `1.0 - cosine_distance`. Chroma's empty-collection query returns `{"ids": [[]], ...}` (verified — does not raise), and requesting `n_results` greater than the stored count is clipped silently by Chroma (verified) — `query()` must rely on both of these rather than adding its own defensive branching for them.

---

## File Structure

- Create: `retrieval/vector_store.py` — `init_store()`, `add_documents()`, `query()`. All Chroma interaction lives here.
- Create: `retrieval/standard_rag.py` — `standard_rag_answer_context()`. Depends only on `retrieval/vector_store.query()`.
- Create: `tests/test_retrieval_vector_store.py`
- Create: `tests/test_retrieval_standard_rag.py`
- Create: `scripts/demo_standard_rag.py` — end-to-end demo, no blockchain/IPFS.
- Modify: `DECISIONS.md` — append a "Milestone 2" section (Task 5).

`retrieval/__init__.py` already exists (empty, from Milestone 0 scaffolding) — no change needed there.

---

### Task 1: `init_store()` — Chroma client + collection factory

**Files:**
- Create: `retrieval/vector_store.py`
- Test: `tests/test_retrieval_vector_store.py`

**Interfaces:**
- Produces: `init_store(persist_directory: str | None = None) -> chromadb.api.models.Collection.Collection`. When `persist_directory` is `None`, loads `.env` via `python-dotenv` and reads `CHROMA_DB_DIR` (falling back to `"./vector_store"` if unset — matches `.env.example`). Collection name is the module constant `derag_chunks`. Always configured with `configuration={"hnsw": {"space": "cosine"}}` and `embedding_function=None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_retrieval_vector_store.py
from retrieval.vector_store import init_store


def test_init_store_creates_cosine_collection(tmp_path):
    collection = init_store(persist_directory=str(tmp_path))
    assert collection.name == "derag_chunks"
    assert collection.configuration_json["hnsw"]["space"] == "cosine"


def test_init_store_is_idempotent_against_same_directory(tmp_path):
    first = init_store(persist_directory=str(tmp_path))
    second = init_store(persist_directory=str(tmp_path))
    assert first.name == second.name == "derag_chunks"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_retrieval_vector_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'retrieval.vector_store'`

- [ ] **Step 3: Write minimal implementation**

```python
# retrieval/vector_store.py
import os

import chromadb
from chromadb.api.models.Collection import Collection
from dotenv import load_dotenv

_COLLECTION_NAME = "derag_chunks"


def init_store(persist_directory: str | None = None) -> Collection:
    if persist_directory is None:
        load_dotenv()
        persist_directory = os.environ.get("CHROMA_DB_DIR", "./vector_store")
    client = chromadb.PersistentClient(path=persist_directory)
    return client.get_or_create_collection(
        _COLLECTION_NAME,
        configuration={"hnsw": {"space": "cosine"}},
        embedding_function=None,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_retrieval_vector_store.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add retrieval/vector_store.py tests/test_retrieval_vector_store.py
git commit -m "feat: add init_store() for persistent Chroma collection with cosine space"
```

---

### Task 2: `add_documents()` — upsert Milestone 1 records with the four-field metadata

**Files:**
- Modify: `retrieval/vector_store.py`
- Test: `tests/test_retrieval_vector_store.py`

**Interfaces:**
- Consumes: `init_store()` from Task 1. `pipeline.ingest_prep.ingest_prep(raw_text: str) -> list[dict]` (Milestone 1, record shape `{"chunk_id": int, "text": str, "hash": str, "embedding": list[float]}`).
- Produces: `add_documents(records: list[dict], source_file: str, collection: Collection | None = None) -> None`. When `collection` is `None`, calls `init_store()` (default `./vector_store` / `CHROMA_DB_DIR`) — tests always pass an explicit `collection` obtained from `init_store(persist_directory=str(tmp_path))` to stay isolated. No-ops on an empty `records` list.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_retrieval_vector_store.py
from pipeline.ingest_prep import ingest_prep
from retrieval.vector_store import add_documents, init_store


def test_add_documents_stores_four_field_metadata(tmp_path):
    collection = init_store(persist_directory=str(tmp_path))
    records = ingest_prep("Solar panels convert sunlight directly into electricity.")

    add_documents(records, source_file="solar.txt", collection=collection)

    stored = collection.get(ids=[records[0]["hash"]], include=["metadatas", "documents"])
    assert stored["ids"] == [records[0]["hash"]]
    metadata = stored["metadatas"][0]
    assert set(metadata.keys()) == {"source_file", "chunk_id", "ipfs_cid", "doc_hash"}
    assert metadata["source_file"] == "solar.txt"
    assert metadata["chunk_id"] == records[0]["chunk_id"]
    assert metadata["ipfs_cid"] == ""  # Ruling (Task 2): Chroma drops None-valued metadata; "" is the placeholder
    assert metadata["doc_hash"] == records[0]["hash"]
    assert stored["documents"][0] == records[0]["text"]


def test_add_documents_reingesting_identical_chunk_does_not_duplicate(tmp_path):
    collection = init_store(persist_directory=str(tmp_path))
    records = ingest_prep("Solar panels convert sunlight directly into electricity.")

    add_documents(records, source_file="solar.txt", collection=collection)
    count_after_first = collection.count()

    add_documents(records, source_file="solar.txt", collection=collection)
    count_after_second = collection.count()

    assert count_after_first == count_after_second == len(records)


def test_add_documents_empty_records_is_noop(tmp_path):
    collection = init_store(persist_directory=str(tmp_path))
    add_documents([], source_file="empty.txt", collection=collection)
    assert collection.count() == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_retrieval_vector_store.py -v`
Expected: FAIL with `ImportError: cannot import name 'add_documents'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to retrieval/vector_store.py

def add_documents(
    records: list[dict], source_file: str, collection: Collection | None = None
) -> None:
    if not records:
        return
    if collection is None:
        collection = init_store()
    ids = [record["hash"] for record in records]
    embeddings = [record["embedding"] for record in records]
    documents = [record["text"] for record in records]
    metadatas = [
        {
            "source_file": source_file,
            "chunk_id": record["chunk_id"],
            "ipfs_cid": "",  # Ruling (Task 2): Chroma drops None-valued metadata entries; "" survives the round-trip
            "doc_hash": record["hash"],
        }
        for record in records
    ]
    collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_retrieval_vector_store.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add retrieval/vector_store.py tests/test_retrieval_vector_store.py
git commit -m "feat: add add_documents() with doc_hash-as-id upsert and four-field metadata"
```

---

### Task 3: `query()` — embed, search, convert distance to similarity, sort

**Files:**
- Modify: `retrieval/vector_store.py`
- Test: `tests/test_retrieval_vector_store.py`

**Interfaces:**
- Consumes: `pipeline.embeddings.embed(texts: list[str]) -> list[list[float]]` (Milestone 1). `add_documents()` and `init_store()` from Tasks 1-2.
- Produces: `query(query_text: str, k: int = 10, collection: Collection | None = None) -> list[dict]`, each dict `{"chroma_id": str, "text": str, "similarity": float, "metadata": dict}`, sorted by `similarity` descending. Returns `[]` on an empty collection. This is the exact interface `retrieval/standard_rag.py` (Task 4) calls.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_retrieval_vector_store.py
from retrieval.vector_store import query


SAMPLE_TOPICS = {
    "solar.txt": "Solar panels convert sunlight directly into electricity using the photovoltaic effect.",
    "pasta.txt": "To cook pasta al dente, boil salted water and simmer the noodles for eight to ten minutes.",
    "jazz.txt": "Bebop jazz emerged in the 1940s with fast tempos and complex chord improvisation.",
    "volcano.txt": "Shield volcanoes form from slow-flowing basaltic lava that spreads over wide areas.",
}


def _seed_store(collection):
    for source_file, text in SAMPLE_TOPICS.items():
        add_documents(ingest_prep(text), source_file=source_file, collection=collection)


def test_query_top_hit_matches_expected_topic(tmp_path):
    collection = init_store(persist_directory=str(tmp_path))
    _seed_store(collection)

    hits = query("How do photovoltaic cells generate power from the sun?", collection=collection)

    assert len(hits) > 0
    assert hits[0]["metadata"]["source_file"] == "solar.txt"
    assert set(hits[0].keys()) == {"chroma_id", "text", "similarity", "metadata"}


def test_query_results_sorted_by_similarity_descending(tmp_path):
    collection = init_store(persist_directory=str(tmp_path))
    _seed_store(collection)

    hits = query("jazz improvisation and chord changes", collection=collection)

    similarities = [hit["similarity"] for hit in hits]
    assert similarities == sorted(similarities, reverse=True)


def test_query_respects_k(tmp_path):
    collection = init_store(persist_directory=str(tmp_path))
    _seed_store(collection)

    hits = query("volcanic activity", k=2, collection=collection)

    assert len(hits) == 2


def test_query_empty_collection_returns_empty_list(tmp_path):
    collection = init_store(persist_directory=str(tmp_path))

    hits = query("anything at all", collection=collection)

    assert hits == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_retrieval_vector_store.py -v`
Expected: FAIL with `ImportError: cannot import name 'query'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to retrieval/vector_store.py
from pipeline.embeddings import embed


def query(query_text: str, k: int = 10, collection: Collection | None = None) -> list[dict]:
    if collection is None:
        collection = init_store()
    query_embedding = embed([query_text])[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    ids = results["ids"][0]
    if not ids:
        return []
    hits = [
        {
            "chroma_id": chroma_id,
            "text": text,
            "similarity": 1.0 - distance,
            "metadata": metadata,
        }
        for chroma_id, text, metadata, distance in zip(
            ids, results["documents"][0], results["metadatas"][0], results["distances"][0]
        )
    ]
    hits.sort(key=lambda hit: hit["similarity"], reverse=True)
    return hits
```

(`embed` import goes at the top of the file alongside the existing imports — not inline.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_retrieval_vector_store.py -v`
Expected: PASS (9 tests)

- [ ] **Step 5: Commit**

```bash
git add retrieval/vector_store.py tests/test_retrieval_vector_store.py
git commit -m "feat: add query() with cosine-distance-to-similarity conversion"
```

---

### Task 4: `standard_rag_answer_context()` — thin top-3 wrapper

**Files:**
- Create: `retrieval/standard_rag.py`
- Test: `tests/test_retrieval_standard_rag.py`

**Interfaces:**
- Consumes: `retrieval.vector_store.query(query_text: str, k: int = 10, collection: Collection | None = None) -> list[dict]` (Task 3), `retrieval.vector_store.init_store` / `add_documents` for test setup, `pipeline.ingest_prep.ingest_prep`.
- Produces: `standard_rag_answer_context(query: str, k: int = 3, collection: Collection | None = None) -> list[dict]` — same dict shape as `vector_store.query()`. Pure delegation: `return vector_store.query(query, k=k, collection=collection)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_retrieval_standard_rag.py
from pipeline.ingest_prep import ingest_prep
from retrieval.standard_rag import standard_rag_answer_context
from retrieval.vector_store import add_documents, init_store

SAMPLE_TOPICS = {
    "solar.txt": "Solar panels convert sunlight directly into electricity using the photovoltaic effect.",
    "pasta.txt": "To cook pasta al dente, boil salted water and simmer the noodles for eight to ten minutes.",
    "jazz.txt": "Bebop jazz emerged in the 1940s with fast tempos and complex chord improvisation.",
    "volcano.txt": "Shield volcanoes form from slow-flowing basaltic lava that spreads over wide areas.",
}


def _seed_store(collection):
    for source_file, text in SAMPLE_TOPICS.items():
        add_documents(ingest_prep(text), source_file=source_file, collection=collection)


def test_standard_rag_returns_top_3_by_default(tmp_path):
    collection = init_store(persist_directory=str(tmp_path))
    _seed_store(collection)

    hits = standard_rag_answer_context("solar power from sunlight", collection=collection)

    assert len(hits) == 3
    assert hits[0]["metadata"]["source_file"] == "solar.txt"


def test_standard_rag_respects_custom_k(tmp_path):
    collection = init_store(persist_directory=str(tmp_path))
    _seed_store(collection)

    hits = standard_rag_answer_context("jazz music", k=1, collection=collection)

    assert len(hits) == 1
    assert hits[0]["metadata"]["source_file"] == "jazz.txt"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_retrieval_standard_rag.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'retrieval.standard_rag'`

- [ ] **Step 3: Write minimal implementation**

```python
# retrieval/standard_rag.py
from chromadb.api.models.Collection import Collection

from retrieval.vector_store import query


def standard_rag_answer_context(
    query_text: str, k: int = 3, collection: Collection | None = None
) -> list[dict]:
    return query(query_text, k=k, collection=collection)
```

Note: the parameter is named `query_text` internally (matching `vector_store.query`'s parameter name) to avoid shadowing the imported `query` function; callers still invoke it positionally as `standard_rag_answer_context(query, k=3)` per the spec.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_retrieval_standard_rag.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add retrieval/standard_rag.py tests/test_retrieval_standard_rag.py
git commit -m "feat: add standard_rag_answer_context() top-3 retrieval wrapper"
```

---

### Task 5: Demo script + full suite verification + DECISIONS.md entry

**Files:**
- Create: `scripts/demo_standard_rag.py`
- Modify: `DECISIONS.md`

**Interfaces:**
- Consumes: `pipeline.ingest_prep.ingest_prep`, `retrieval.vector_store.init_store`/`add_documents`, `retrieval.standard_rag.standard_rag_answer_context` — all from Tasks 1-4. Reads `data/sample.txt` (already exists from Milestone 1's demo).

- [ ] **Step 1: Write the demo script**

```python
# scripts/demo_standard_rag.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.ingest_prep import ingest_prep
from retrieval.standard_rag import standard_rag_answer_context
from retrieval.vector_store import add_documents, init_store


def main() -> None:
    sample_path = Path(__file__).resolve().parent.parent / "data" / "sample.txt"
    raw_text = sample_path.read_text(encoding="utf-8")

    collection = init_store()
    records = ingest_prep(raw_text)
    add_documents(records, source_file=str(sample_path.name), collection=collection)
    print(f"Stored {len(records)} chunk(s) from {sample_path} into ChromaDB\n")

    query_text = "How does DeRAG defend against data poisoning in retrieval?"
    hits = standard_rag_answer_context(query_text, collection=collection)

    print(f"Query: {query_text}\n")
    for rank, hit in enumerate(hits, start=1):
        print(f"#{rank} similarity={hit['similarity']:.4f} source={hit['metadata']['source_file']}")
        print(f"    {hit['text'][:120]}...")
        print()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the demo and confirm plausible ranked output**

Run: `uv run python scripts/demo_standard_rag.py`
Expected: prints "Stored N chunk(s)..." then a query followed by up to 3 ranked hits with descending similarity scores, all `source=sample.txt`, no errors, no blockchain/IPFS references anywhere in the output.

- [ ] **Step 3: Run the full test suite**

Run: `uv run pytest`
Expected: all tests pass, including the pre-existing Milestone 0/1 suites plus the two new Milestone 2 test files (13 new tests total: 9 in `test_retrieval_vector_store.py`, 2 in `test_retrieval_standard_rag.py`, plus whatever Milestone 1 already had).

- [ ] **Step 4: Append the DECISIONS.md entry**

Add this section to `DECISIONS.md`, after the existing "Milestone 1" section:

```markdown
## Milestone 2 — Vector Store & Retrieval (2026-08-15)

- **Chroma document ID = `record["hash"]` (Milestone 1's sha256 of the normalized
  chunk text), used directly** instead of inventing a new ID scheme. It's already
  unique per content, so no extra bookkeeping is needed, and it sidesteps
  Milestone 1's flagged caveat that `chunk_id` is only unique within a single
  `ingest_prep()` call, not globally. Storage uses `collection.upsert()` (never
  `add()`), so re-ingesting byte-identical content overwrites in place instead of
  raising a duplicate-ID error or creating a duplicate entry. **This idempotency
  is convenient but coincidental** — it falls out of Chroma's upsert semantics,
  not a designed guarantee. Milestone 5's blockchain-registration idempotency
  needs its own explicit dedup check against on-chain state; don't assume this
  Chroma behavior extends there.
- **Collection distance space: cosine**, set via `configuration={"hnsw": {"space":
  "cosine"}}` on `get_or_create_collection()`. Verified against the real
  `chromadb==1.5.9` install rather than assumed: `collection.configuration_json`
  echoes the setting back, and a manual check with two orthogonal unit vectors
  returned `distance=1.0` (cosine) rather than `2.0` (squared-L2), confirming the
  space is actually applied, not silently defaulted. Embeddings are L2-normalized
  (Milestone 1), so cosine and L2 rank identically here regardless — this
  configuration matters for matching the spec's stated metric and for readers of
  the code, not for the actual ranking output.
- **`embedding_function=None` on every collection open**, since this project
  always supplies its own vectors via `pipeline/embeddings.py`. Verified this
  actually disables Chroma's default ONNX auto-embed (not just declines to set
  one): calling `add()` with `documents=` but no `embeddings=` raises
  `ValueError: You must provide an embedding function...`.
- **Metadata schema is locked to exactly four fields**: `source_file` (str),
  `chunk_id` (int, Milestone 1's per-call value, unchanged — NOT the Chroma ID),
  `ipfs_cid` (`""` empty-string placeholder until Milestone 4/5 wire in real values), `doc_hash` (the
  same string as the Chroma ID, `record["hash"]`). No author/publisher field —
  that identity lives on-chain in Milestone 3's `Document` struct, looked up via
  `doc_hash`. **Milestones 3-5 must populate `ipfs_cid` using this exact field
  name** when real IPFS CIDs become available.
```

- [ ] **Step 5: Commit**

```bash
git add scripts/demo_standard_rag.py DECISIONS.md
git commit -m "docs: add standard RAG demo script and Milestone 2 decisions record"
```
