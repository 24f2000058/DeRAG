# Milestone 1: DeRAG Preprocessing Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `/pipeline` as five reusable, independently testable modules implementing the fixed preprocessing order (normalize → chunk → hash) plus embedding, with zero blockchain/IPFS/ChromaDB integration — those come in later milestones.

**Architecture:** Five single-responsibility modules under `pipeline/`, each with one public function (plus `embeddings.py`'s two), composed by `pipeline/ingest_prep.py` into one pipeline call. Every module is directly unit-testable with no external service dependency except `embeddings.py`, which downloads and caches `all-MiniLM-L6-v2` from the HuggingFace Hub on first run (network required once; fully offline after that).

**Tech Stack:** Python 3.11, `uv`, `pytest`, `langchain-text-splitters` 1.1.x (`RecursiveCharacterTextSplitter`), `sentence-transformers` 5.7.0 (`all-MiniLM-L6-v2`), `numpy`, stdlib `hashlib`/`unicodedata`/`re`.

**Spec:** DeRAG project spec (external, user-provided — not yet checked into `docs/superpowers/specs/`); this milestone's two open design points (normalization algorithm, `langchain-text-splitters` import/dependency) were resolved via in-chat brainstorm on 2026-08-15 and are recorded in this plan's Global Constraints and in `DECISIONS.md`. Builds on `docs/superpowers/specs/2026-08-14-milestone-0-scaffolding-design.md` (repo layout, `uv`/pytest conventions).

## Global Constraints

- Preprocessing order is fixed: **normalize → chunk → hash**. Do not reorder, do not fold steps together.
- Chunking: `RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)` — characters, not tokens. Do not reopen these values.
- `pipeline/chunking.py` operates only on already-normalized text; it must never call `normalize_text` itself.
- Hashing is SHA-256 hex digest of the exact string passed in, encoded explicit UTF-8 (`chunk.encode("utf-8")`) — no implicit encoding, no re-normalization inside `hash_chunk`.
- Normalization contract (hard dependency for Milestone 4's IPFS re-hash round trip): NFKC → strip Unicode categories `Cc`/`Cf` except `\t`/`\n`/`\r` → collapse whitespace runs to single space + strip → lowercase. Must be idempotent: `normalize_text(normalize_text(x)) == normalize_text(x)`.
- Embedding model is `all-MiniLM-L6-v2` via `sentence-transformers`, loaded once and cached (module-level `lru_cache`), not reloaded per call. Output vectors are 384-dimensional. Model's `max_seq_length` must be 256 and pooling mode must be mean pooling — asserted directly in tests, not assumed.
- `ingest_prep()`'s output record shape is a hard dependency for Milestone 5's ingestion script — field names (`chunk_id`, `text`, `hash`, `embedding`) must not change once written.
- All Python tests live in the single top-level `/tests/` directory, named `tests/test_<module>_<thing>.py` — no per-module `tests/` subfolders (locked in Milestone 0).
- No blockchain, IPFS, or ChromaDB code anywhere in this milestone.

---

## File Structure

```
DeRAG/
├── pipeline/
│   ├── __init__.py          # exists, untouched
│   ├── normalize.py         # new
│   ├── chunking.py          # new
│   ├── hashing.py           # new
│   ├── embeddings.py        # new
│   └── ingest_prep.py       # new
├── tests/
│   ├── test_pipeline_normalize.py     # new
│   ├── test_pipeline_chunking.py      # new
│   ├── test_pipeline_hashing.py       # new
│   ├── test_pipeline_embeddings.py    # new
│   └── test_pipeline_ingest_prep.py   # new
├── data/
│   └── sample.txt           # new
├── scripts/
│   └── demo_ingest_prep.py  # new
├── pyproject.toml           # modified: add langchain-text-splitters
└── DECISIONS.md             # modified: append Milestone 1 entry
```

---

## Task 1: `pipeline/normalize.py`

**Files:**
- Create: `pipeline/normalize.py`
- Test: `tests/test_pipeline_normalize.py`

**Interfaces:**
- Produces: `normalize_text(raw: str) -> str`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_pipeline_normalize.py
from pipeline.normalize import normalize_text


def test_normalize_removes_unicode_artifacts_and_lowercases():
    raw = "Café​ RÉSUMÉ  with\tMixed\nCASE   and   spaces"
    result = normalize_text(raw)
    assert result == "café résumé with mixed case and spaces"


def test_normalize_is_idempotent():
    raw = "Ｆüllwidth​Tëst  Ｓtring  with   ARTIFACTS\n\n"
    once = normalize_text(raw)
    twice = normalize_text(once)
    assert once == twice
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_pipeline_normalize.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.normalize'`

- [ ] **Step 3: Write the implementation**

```python
# pipeline/normalize.py
import re
import unicodedata

_KEEP_WHITESPACE = "\t\n\r"

_REMOVE_TABLE = {
    codepoint: None
    for codepoint in range(0x110000)
    if unicodedata.category(chr(codepoint)) in ("Cc", "Cf")
    and chr(codepoint) not in _KEEP_WHITESPACE
}

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(raw: str) -> str:
    text = unicodedata.normalize("NFKC", raw)
    text = text.translate(_REMOVE_TABLE)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text.lower()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_pipeline_normalize.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add pipeline/normalize.py tests/test_pipeline_normalize.py
git commit -m "feat(pipeline): add normalize_text with NFKC + control-char strip + whitespace collapse + lowercase"
```

---

## Task 2: `langchain-text-splitters` explicit dependency + `pipeline/chunking.py`

**Context:** `RecursiveCharacterTextSplitter` currently resolves only because `langchain-community` pulls in `langchain-text-splitters` transitively. `langchain-community` prints a sunset/deprecation warning on import. Add the real dependency directly so this import doesn't silently break on a future `langchain-community` removal.

**Files:**
- Modify: `pyproject.toml`
- Create: `pipeline/chunking.py`
- Test: `tests/test_pipeline_chunking.py`

**Interfaces:**
- Consumes: nothing from Task 1 (chunking must not call `normalize_text`)
- Produces: `chunk_text(normalized_text: str) -> list[str]`

- [ ] **Step 1: Add the explicit dependency**

In `pyproject.toml`, in the `dependencies` list, add a line after `"langchain-community>=0.3",`:

```toml
    "langchain-text-splitters>=1.1",
```

Then run: `uv sync`
Expected: resolves without changing the already-locked `langchain-text-splitters` version (1.1.2 is already in `uv.lock` as a transitive dep, so this should be a no-op re-lock that just adds the direct edge).

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_pipeline_chunking.py
from pipeline.chunking import chunk_text

SAMPLE = " ".join(f"word{i:04d}" for i in range(200))  # 1799 chars, no punctuation


def test_chunk_count_and_sizes():
    chunks = chunk_text(SAMPLE)
    assert len(chunks) == 4
    for chunk in chunks:
        assert len(chunk) <= 500


def test_chunk_overlap_shares_boundary_words():
    chunks = chunk_text(SAMPLE)
    assert chunks[0].endswith(
        "word0050 word0051 word0052 word0053 word0054"
    )
    assert chunks[1].startswith(
        "word0050 word0051 word0052 word0053 word0054"
    )
    assert chunks[1].endswith(
        "word0100 word0101 word0102 word0103 word0104"
    )
    assert chunks[2].startswith(
        "word0100 word0101 word0102 word0103 word0104"
    )


def test_chunking_does_not_renormalize():
    # Mixed-case, unnormalized input passed straight through untouched
    # except for the splitter's own boundary logic. Confirms chunking is
    # a pure split, not a normalize+split.
    raw = "ALREADY Mixed CASE text " * 30
    chunks = chunk_text(raw)
    assert any(c != c.lower() for c in chunks)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_pipeline_chunking.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.chunking'`

- [ ] **Step 4: Write the implementation**

```python
# pipeline/chunking.py
from langchain_text_splitters import RecursiveCharacterTextSplitter

_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)


def chunk_text(normalized_text: str) -> list[str]:
    return _SPLITTER.split_text(normalized_text)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_pipeline_chunking.py -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock pipeline/chunking.py tests/test_pipeline_chunking.py
git commit -m "feat(pipeline): add chunk_text via RecursiveCharacterTextSplitter, declare langchain-text-splitters directly"
```

---

## Task 3: `pipeline/hashing.py`

**Files:**
- Create: `pipeline/hashing.py`
- Test: `tests/test_pipeline_hashing.py`

**Interfaces:**
- Consumes: nothing (operates on any string)
- Produces: `hash_chunk(chunk: str) -> str`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_pipeline_hashing.py
"""
Milestone 4 dependency: Milestone 4's IPFS round trip fetches stored chunk
text back from IPFS and re-hashes it to verify integrity. That re-hash must
equal the hash computed here at ingest time. hash_chunk() hashes exactly the
string it is given (explicit UTF-8, no implicit encoding, no internal
re-normalization) — so as long as Milestone 4 hashes the fetched chunk text
verbatim (not re-running it through normalize_text), the hashes match by
construction. This test file documents and locks that contract.
"""
import hashlib

from pipeline.hashing import hash_chunk


def test_hash_is_deterministic():
    text = "the quick brown fox"
    assert hash_chunk(text) == hash_chunk(text)


def test_different_inputs_produce_different_hashes():
    assert hash_chunk("the quick brown fox") != hash_chunk("the lazy dog")


def test_hash_matches_explicit_sha256_utf8():
    text = "café résumé"
    expected = hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert hash_chunk(text) == expected


def test_hash_is_stable_across_normalize_then_chunk_pipeline():
    # The exact contract Milestone 4 relies on: hashing a chunk that came
    # out of normalize_text + chunk_text is just hash_chunk() on that
    # string — no special-casing, no re-normalization step.
    from pipeline.chunking import chunk_text
    from pipeline.normalize import normalize_text

    normalized = normalize_text("Café RÉSUMÉ with MIXED case")
    chunks = chunk_text(normalized)
    for chunk in chunks:
        assert hash_chunk(chunk) == hashlib.sha256(chunk.encode("utf-8")).hexdigest()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_pipeline_hashing.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.hashing'`

- [ ] **Step 3: Write the implementation**

```python
# pipeline/hashing.py
import hashlib


def hash_chunk(chunk: str) -> str:
    return hashlib.sha256(chunk.encode("utf-8")).hexdigest()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_pipeline_hashing.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add pipeline/hashing.py tests/test_pipeline_hashing.py
git commit -m "feat(pipeline): add hash_chunk, document Milestone 4 hash-contract in test docstring"
```

---

## Task 4: `pipeline/embeddings.py`

**Context:** First run downloads `all-MiniLM-L6-v2` (~90MB) from the HuggingFace Hub into `~/.cache/huggingface`; subsequent runs (including CI, if any) are offline. This is a one-time network dependency for this task only — confirm network access before running these tests if working in a fresh sandbox.

**Files:**
- Create: `pipeline/embeddings.py`
- Test: `tests/test_pipeline_embeddings.py`

**Interfaces:**
- Consumes: nothing from Tasks 1–3
- Produces: `embed(texts: list[str]) -> list[list[float]]`, `cosine_similarity(a: list[float], b: list[float]) -> float`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_pipeline_embeddings.py
from pipeline.embeddings import cosine_similarity, embed, _get_model


def test_embed_returns_384_dim_vectors():
    vectors = embed(["hello world", "another sentence"])
    assert len(vectors) == 2
    for v in vectors:
        assert len(v) == 384


def test_semantic_similarity_ranks_paraphrase_above_unrelated():
    anchor = "The cat sat on the mat."
    paraphrase = "A cat was sitting on the mat."
    unrelated = "The stock market fell sharply today."

    vecs = embed([anchor, paraphrase, unrelated])
    sim_paraphrase = cosine_similarity(vecs[0], vecs[1])
    sim_unrelated = cosine_similarity(vecs[0], vecs[2])

    assert sim_paraphrase > sim_unrelated


def test_model_max_seq_length_is_256():
    model = _get_model()
    assert model.max_seq_length == 256


def test_model_pooling_mode_is_mean():
    model = _get_model()
    pooling_module = model[1]
    assert pooling_module.pooling_mode == "mean"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_pipeline_embeddings.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.embeddings'`

- [ ] **Step 3: Write the implementation**

```python
# pipeline/embeddings.py
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

_MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    return SentenceTransformer(_MODEL_NAME)


def embed(texts: list[str]) -> list[list[float]]:
    model = _get_model()
    vectors = model.encode(texts, convert_to_numpy=True)
    return vectors.tolist()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a_arr = np.array(a)
    b_arr = np.array(b)
    return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_pipeline_embeddings.py -v`
Expected: PASS (4 tests). First run will pause to download the model — this is expected, not a failure.

- [ ] **Step 5: Commit**

```bash
git add pipeline/embeddings.py tests/test_pipeline_embeddings.py
git commit -m "feat(pipeline): add embed/cosine_similarity via cached all-MiniLM-L6-v2, assert seq_length/pooling defaults"
```

---

## Task 5: `pipeline/ingest_prep.py` + demo script + sample data

**Files:**
- Create: `pipeline/ingest_prep.py`
- Create: `data/sample.txt`
- Create: `scripts/demo_ingest_prep.py`
- Test: `tests/test_pipeline_ingest_prep.py`

**Interfaces:**
- Consumes: `normalize_text` (Task 1), `chunk_text` (Task 2), `hash_chunk` (Task 3), `embed` (Task 4)
- Produces: `ingest_prep(raw_text: str) -> list[dict]`, each dict shaped `{"chunk_id": int, "text": str, "hash": str, "embedding": list[float]}` — this exact shape is what Milestone 5's ingestion script consumes.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_pipeline_ingest_prep.py
import hashlib

from pipeline.ingest_prep import ingest_prep
from pipeline.normalize import normalize_text


def test_ingest_prep_record_shape():
    records = ingest_prep("Hello WORLD. " * 60)
    assert len(records) > 0
    for record in records:
        assert set(record.keys()) == {"chunk_id", "text", "hash", "embedding"}
        assert isinstance(record["chunk_id"], int)
        assert isinstance(record["text"], str)
        assert isinstance(record["hash"], str)
        assert isinstance(record["embedding"], list)
        assert len(record["embedding"]) == 384


def test_ingest_prep_chunk_ids_are_sequential():
    records = ingest_prep("Hello WORLD. " * 60)
    assert [r["chunk_id"] for r in records] == list(range(len(records)))


def test_ingest_prep_hash_matches_normalized_chunk_text():
    raw = "Café RÉSUMÉ with MIXED case. " * 40
    records = ingest_prep(raw)
    for record in records:
        expected_hash = hashlib.sha256(record["text"].encode("utf-8")).hexdigest()
        assert record["hash"] == expected_hash


def test_ingest_prep_text_is_normalized():
    raw = "SHOUTING TEXT with​artifacts"
    records = ingest_prep(raw)
    combined = " ".join(r["text"] for r in records)
    assert combined == normalize_text(combined)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_pipeline_ingest_prep.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.ingest_prep'`

- [ ] **Step 3: Write the implementation**

```python
# pipeline/ingest_prep.py
from pipeline.chunking import chunk_text
from pipeline.embeddings import embed
from pipeline.hashing import hash_chunk
from pipeline.normalize import normalize_text


def ingest_prep(raw_text: str) -> list[dict]:
    normalized = normalize_text(raw_text)
    chunks = chunk_text(normalized)
    embeddings = embed(chunks)
    return [
        {
            "chunk_id": i,
            "text": chunk,
            "hash": hash_chunk(chunk),
            "embedding": embedding,
        }
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_pipeline_ingest_prep.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Create the sample data file**

```
# data/sample.txt
DeRAG is a decentralized, verifiable retrieval-augmented generation system.
It combines local embedding models, an immutable blockchain reputation
ledger, and content-addressed storage to defend against data poisoning
attacks on retrieval corpora.  Unlike conventional RAG pipelines, which
trust semantic similarity alone, DeRAG cross-references every retrieved
document against a crowd-sourced trust score before it ever reaches the
language model's context window.  Low-reputation or unverified sources are
down-weighted instead of blindly trusted, and every chunk is content-hashed
so that later fetches from decentralized storage can be checked byte-for-byte
against what was originally ingested.  This preprocessing stage — normalize,
then chunk, then hash — is the foundation every later stage of the system
depends on: retrieval ranks chunks by embedding similarity, the blockchain
layer stores each chunk's hash and reputation on-chain, and the generation
stage assembles a final answer only from chunks that clear the trust
threshold.
```

- [ ] **Step 6: Write the demo script**

```python
# scripts/demo_ingest_prep.py
from pathlib import Path

from pipeline.ingest_prep import ingest_prep


def main() -> None:
    sample_path = Path(__file__).resolve().parent.parent / "data" / "sample.txt"
    raw_text = sample_path.read_text(encoding="utf-8")

    records = ingest_prep(raw_text)

    print(f"Ingested {len(records)} chunk(s) from {sample_path}\n")
    for record in records:
        print(f"chunk_id={record['chunk_id']}")
        print(f"  text ({len(record['text'])} chars): {record['text'][:80]}...")
        print(f"  hash: {record['hash']}")
        print(f"  embedding dims: {len(record['embedding'])}")
        print()


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Run the demo script and confirm output looks correct**

Run: `uv run python scripts/demo_ingest_prep.py`
Expected: prints 1+ chunk records, each with a 64-char hex hash and `embedding dims: 384`.

- [ ] **Step 8: Commit**

```bash
git add pipeline/ingest_prep.py tests/test_pipeline_ingest_prep.py data/sample.txt scripts/demo_ingest_prep.py
git commit -m "feat(pipeline): add ingest_prep combining normalize+chunk+hash+embed, demo script, sample data"
```

---

## Task 6: Full suite verification + DECISIONS.md entry

**Files:**
- Modify: `DECISIONS.md`

- [ ] **Step 1: Run the full test suite**

Run: `uv run pytest -v`
Expected: all tests pass — Milestone 0's `tests/test_smoke.py` plus all five new `tests/test_pipeline_*.py` files.

- [ ] **Step 2: Append the Milestone 1 entry to `DECISIONS.md`**

```markdown

## Milestone 1 — Preprocessing Pipeline (2026-08-15)

- **`normalize_text()` algorithm (hard contract for Milestone 4's IPFS
  re-hash round trip):** `unicodedata.normalize("NFKC", raw)` → strip
  Unicode categories `Cc`/`Cf` except `\t`/`\n`/`\r` (via a precomputed
  `str.translate` removal table) → collapse whitespace runs to a single
  space and strip ends (`re.sub(r"\s+", " ", text).strip()`) → `.lower()`.
  NFKC runs first so it folds exotic whitespace (e.g. NBSP) to regular
  space before the control-char strip and whitespace collapse run, and so
  it folds compatibility/full-width variants before lowercasing. Verified
  idempotent: `normalize_text(normalize_text(x)) == normalize_text(x)`.
  Milestone 4 must hash fetched chunk text **verbatim** (never re-run it
  through `normalize_text`) — `hash_chunk` only ever hashes the exact
  string it's given, so the two hashes match by construction as long as
  that invariant holds.
- **`langchain-text-splitters` is now a direct dependency**
  (`pyproject.toml`, `>=1.1`, resolved to 1.1.2 in `uv.lock`), not just a
  transitive pull via `langchain-community` (which prints a
  sunset/deprecation warning on import as of 0.4.2). Import path:
  `from langchain_text_splitters import RecursiveCharacterTextSplitter`.
- **`all-MiniLM-L6-v2` downloads on first use**, cached at
  `~/.cache/huggingface` afterward. One-time network dependency for this
  milestone only (and any future fresh environment); everything after the
  first load is fully offline, consistent with the project's "no cloud
  APIs" runtime stance.
- **`ingest_prep()` record shape is locked**: `{"chunk_id": int, "text":
  str, "hash": str, "embedding": list[float]}`, `chunk_id` is a 0-based
  sequential index per call. Milestone 5's ingestion script consumes this
  shape directly — do not rename these fields without updating Milestone 5.
```

- [ ] **Step 3: Commit**

```bash
git add DECISIONS.md
git commit -m "docs: record Milestone 1 normalization contract and langchain-text-splitters resolution"
```
