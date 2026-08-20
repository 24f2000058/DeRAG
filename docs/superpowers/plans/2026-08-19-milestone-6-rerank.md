# Milestone 6: Trust-Weighted Re-Ranking Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `retrieval/rerank.py`, the trust-weighted re-ranking engine that combines vector similarity with on-chain reputation to demote high-similarity, low/negative-reputation ("poisoned") chunks below lower-similarity, well-reputed ones.

**Architecture:** Four pure/near-pure functions layered bottom-up: a thin reputation-read wrapper on `blockchain.registry_client.verify_integrity()`, a batch normalizer that maps raw reputation (including the `exists=False` case) onto `[0, 1]` via a fixed sigmoid curve, a hybrid scorer that linearly combines similarity and normalized reputation, and a top-level orchestrator that pulls candidates from `retrieval.vector_store.query()`, enriches them, and returns the top-K by score. All four are independently unit-testable with mocked inputs — no live Hardhat/IPFS/Chroma required for the core suite.

**Tech Stack:** Python 3.11, `math.exp` (stdlib, no new dependency), `pytest` + `unittest.mock.patch` (existing project convention, see `tests/test_scripts_ingest.py`).

**Spec:** No separate spec doc — this is a bounded task; the approved design is captured in this plan (see Global Constraints) and will be written to `DECISIONS.md` as the final task.

## Global Constraints

- **Reputation read function: `verify_integrity()`, never `get_document()`.** `verify_integrity()` returns the cheap 3-tuple `{exists, reputation, publisher}` via a single contract view call — the only fields this milestone needs. `get_document()` pulls all six struct fields (including `ipfs_cid`, `timestamp`) and stays reserved for Milestone 5's skip-check and Milestone 8's audit UI.
- **`exists=False` handling: a per-batch hard floor, not a fixed reputation=0.** A chunk in Chroma with no on-chain record at all is a strictly worse signal than a registered-but-neutral (`reputation=0`) chunk, because `registerDocument()` already assigns `reputation=0` to every legitimately-published, not-yet-reviewed document — 0 is the neutral midpoint of an unbounded (can-go-negative) range, not a penalty value. `exists=False` candidates get `floor = baseline - FLOOR_PENALTY`, where `baseline` is `min(reputation for c in batch if c["exists"])`, or `0` if no candidate in the batch is registered. `FLOOR_PENALTY = 50` is chosen relative to `SCALE = 10` (5 sigmoid half-widths) so the floor's normalized value saturates near 0 regardless of the batch's actual minimum — this is what makes the "unregistered chunk ranks below every registered chunk, even reputation exactly 0" invariant hold in practice, not just in the common case. (A margin of `FLOOR_PENALTY = 1` was the first idea during design but was rejected after computing that it produces only a ~0.025 normalized-reputation gap — too small to survive a realistic similarity-score gap between candidates.)
- **Normalization method: sigmoid, not min-max.** `normalized = 1 / (1 + exp(-reputation / SCALE))`, `SCALE = 10` — a fixed global curve so a document's normalized trust score is reproducible across different queries and candidate sets (needed for Milestone 10's MRR/Precision@K evaluation), and has no degenerate case (min-max's `(rep-min)/(max-min)` divides by ~0 when all K candidates share similar reputation). `SCALE` is a named module-level constant, not an inline literal.
- **`hybrid_score`: `S(q,d) = alpha*similarity + (1-alpha)*normalized_reputation`, `alpha=0.7` default.** Named, overridable parameter — Milestone 9 will tune it empirically against the real poisoned corpus.
- **No live services required for the core test suite.** All reputation and similarity inputs are mocked via `unittest.mock.patch`, following the pattern in `tests/test_scripts_ingest.py` (`@patch("module.function_name")`, patching the name as imported into `retrieval/rerank.py`, not the origin module).

---

## Task 1: Reputation layer — `get_candidate_reputation` and `normalize_reputation`

**Files:**
- Create: `retrieval/rerank.py`
- Test: `tests/test_retrieval_rerank.py`

**Interfaces:**
- Consumes: `blockchain.registry_client.verify_integrity(doc_hash: str) -> dict` (returns `{"exists": bool, "reputation": int, "publisher": str}`, already implemented).
- Produces:
  - `SCALE: int = 10` (module constant)
  - `FLOOR_PENALTY: int = 50` (module constant)
  - `get_candidate_reputation(doc_hash: str) -> tuple[bool, int]`
  - `normalize_reputation(candidates: list[dict]) -> list[dict]` — each input dict must have `"exists": bool` and `"reputation": int` keys; returns a new list of dicts (original keys preserved, `"normalized_reputation": float` added).

- [ ] **Step 1: Write the failing tests**

```python
import math

import pytest
from unittest.mock import patch

from retrieval.rerank import FLOOR_PENALTY, SCALE, get_candidate_reputation, normalize_reputation


def test_get_candidate_reputation_returns_exists_and_reputation_tuple():
    with patch(
        "retrieval.rerank.verify_integrity",
        return_value={"exists": True, "reputation": 7, "publisher": "0xabc"},
    ):
        result = get_candidate_reputation("somehash")

    assert result == (True, 7)


def test_get_candidate_reputation_passes_through_exists_false():
    with patch(
        "retrieval.rerank.verify_integrity",
        return_value={"exists": False, "reputation": 0, "publisher": "0x" + "0" * 40},
    ):
        result = get_candidate_reputation("unregistered")

    assert result == (False, 0)


def test_normalize_reputation_applies_sigmoid_to_existing_candidates():
    candidates = [
        {"exists": True, "reputation": 0},
        {"exists": True, "reputation": 10},
    ]

    result = normalize_reputation(candidates)

    assert result[0]["normalized_reputation"] == pytest.approx(0.5)
    assert result[1]["normalized_reputation"] == pytest.approx(1 / (1 + math.exp(-1)))


def test_normalize_reputation_preserves_original_keys():
    candidates = [{"exists": True, "reputation": 3, "doc_hash": "abc"}]

    result = normalize_reputation(candidates)

    assert result[0]["doc_hash"] == "abc"
    assert result[0]["reputation"] == 3


def test_normalize_reputation_floors_exists_false_below_batch_minimum():
    candidates = [
        {"exists": True, "reputation": -5},
        {"exists": False, "reputation": 0},
    ]

    result = normalize_reputation(candidates)

    expected_floor = -5 - FLOOR_PENALTY
    assert result[1]["normalized_reputation"] == pytest.approx(
        1 / (1 + math.exp(-expected_floor / SCALE))
    )
    assert result[1]["normalized_reputation"] < result[0]["normalized_reputation"]


def test_normalize_reputation_all_exists_false_falls_back_to_zero_baseline():
    candidates = [
        {"exists": False, "reputation": 0},
        {"exists": False, "reputation": 0},
    ]

    result = normalize_reputation(candidates)

    expected = 1 / (1 + math.exp(-(0 - FLOOR_PENALTY) / SCALE))
    assert result[0]["normalized_reputation"] == pytest.approx(expected)
    assert result[1]["normalized_reputation"] == pytest.approx(expected)


def test_normalize_reputation_identical_reputations_all_candidates_produce_same_value():
    candidates = [
        {"exists": True, "reputation": 5},
        {"exists": True, "reputation": 5},
        {"exists": True, "reputation": 5},
    ]

    result = normalize_reputation(candidates)

    values = {c["normalized_reputation"] for c in result}
    assert len(values) == 1


def test_normalize_reputation_single_candidate_does_not_crash():
    candidates = [{"exists": True, "reputation": 3}]

    result = normalize_reputation(candidates)

    assert len(result) == 1
    assert 0.0 < result[0]["normalized_reputation"] < 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_retrieval_rerank.py -v`
Expected: FAIL/ERROR — `retrieval.rerank` module does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
import math

from blockchain.registry_client import verify_integrity

SCALE = 10
FLOOR_PENALTY = 50


def get_candidate_reputation(doc_hash: str) -> tuple[bool, int]:
    result = verify_integrity(doc_hash)
    return result["exists"], result["reputation"]


def normalize_reputation(candidates: list[dict]) -> list[dict]:
    existing_reputations = [c["reputation"] for c in candidates if c["exists"]]
    baseline = min(existing_reputations) if existing_reputations else 0
    floor = baseline - FLOOR_PENALTY

    result = []
    for candidate in candidates:
        effective_reputation = candidate["reputation"] if candidate["exists"] else floor
        normalized = 1 / (1 + math.exp(-effective_reputation / SCALE))
        result.append({**candidate, "normalized_reputation": normalized})
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_retrieval_rerank.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add retrieval/rerank.py tests/test_retrieval_rerank.py
git commit -m "feat: add reputation read and sigmoid normalization for trust-weighted rerank"
```

---

## Task 2: Scoring — `hybrid_score`

**Files:**
- Modify: `retrieval/rerank.py`
- Test: `tests/test_retrieval_rerank.py`

**Interfaces:**
- Consumes: nothing new (pure function of its own arguments).
- Produces: `hybrid_score(similarity: float, normalized_reputation: float, alpha: float = 0.7) -> float`

- [ ] **Step 1: Write the failing tests**

```python
from retrieval.rerank import hybrid_score


def test_hybrid_score_default_alpha_weights_similarity_at_seventy_percent():
    score = hybrid_score(similarity=1.0, normalized_reputation=0.0, alpha=0.7)
    assert score == pytest.approx(0.7)


def test_hybrid_score_default_alpha_weights_reputation_at_thirty_percent():
    score = hybrid_score(similarity=0.0, normalized_reputation=1.0, alpha=0.7)
    assert score == pytest.approx(0.3)


def test_hybrid_score_respects_custom_alpha():
    score = hybrid_score(similarity=0.5, normalized_reputation=0.5, alpha=0.3)
    assert score == pytest.approx(0.5)


def test_hybrid_score_uses_default_alpha_when_not_specified():
    score = hybrid_score(similarity=1.0, normalized_reputation=0.0)
    assert score == pytest.approx(0.7)
```

(Add `import pytest` at the top of `tests/test_retrieval_rerank.py` if not already present from Task 1.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_retrieval_rerank.py -v -k hybrid_score`
Expected: FAIL — `hybrid_score` not defined.

- [ ] **Step 3: Write minimal implementation**

Add to `retrieval/rerank.py`:

```python
def hybrid_score(similarity: float, normalized_reputation: float, alpha: float = 0.7) -> float:
    return alpha * similarity + (1 - alpha) * normalized_reputation
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_retrieval_rerank.py -v -k hybrid_score`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add retrieval/rerank.py tests/test_retrieval_rerank.py
git commit -m "feat: add hybrid_score linear combination of similarity and reputation"
```

---

## Task 3: Orchestration — `trust_weighted_rerank` and the core thesis tests

**Files:**
- Modify: `retrieval/rerank.py`
- Test: `tests/test_retrieval_rerank.py`

**Interfaces:**
- Consumes:
  - `retrieval.vector_store.query(query_text: str, k: int = 10, collection=None) -> list[dict]` (already implemented; each hit has `"chroma_id"`, `"text"`, `"similarity"`, `"metadata"` where `metadata["doc_hash"]` is present).
  - `get_candidate_reputation`, `normalize_reputation`, `hybrid_score` from Tasks 1-2.
- Produces: `trust_weighted_rerank(query: str, k_retrieve: int = 10, k_return: int = 3, alpha: float = 0.7) -> list[dict]` — each returned dict is a candidate's original hit dict plus `"exists"`, `"reputation"`, `"normalized_reputation"`, `"score"` keys, sorted by `"score"` descending, truncated to `k_return`.

- [ ] **Step 1: Write the failing tests**

```python
from retrieval.rerank import trust_weighted_rerank


def _poisoning_scenario_hits():
    return [
        {
            "chroma_id": "poisoned_hash",
            "text": "poisoned chunk with strong keyword overlap",
            "similarity": 0.95,
            "metadata": {"doc_hash": "poisoned_hash", "source_file": "poison.txt"},
        },
        {
            "chroma_id": "factual_hash",
            "text": "factual chunk with slightly lower overlap",
            "similarity": 0.90,
            "metadata": {"doc_hash": "factual_hash", "source_file": "factual.txt"},
        },
    ]


def _poisoning_scenario_reputations(doc_hash):
    reputations = {
        "poisoned_hash": {"exists": True, "reputation": -50, "publisher": "0xabc"},
        "factual_hash": {"exists": True, "reputation": 10, "publisher": "0xdef"},
    }
    return reputations[doc_hash]


def test_trust_weighted_rerank_demotes_high_similarity_negative_reputation_doc():
    # Prove the pure-cosine baseline (Standard RAG) would rank the poisoned
    # doc #1: it has the higher similarity score.
    standard_rag_order = sorted(
        _poisoning_scenario_hits(), key=lambda hit: hit["similarity"], reverse=True
    )
    assert standard_rag_order[0]["metadata"]["doc_hash"] == "poisoned_hash"

    with (
        patch("retrieval.rerank.query", return_value=_poisoning_scenario_hits()),
        patch("retrieval.rerank.verify_integrity", side_effect=_poisoning_scenario_reputations),
    ):
        results = trust_weighted_rerank("test query", k_retrieve=2, k_return=2)

    assert results[0]["metadata"]["doc_hash"] == "factual_hash"
    assert results[1]["metadata"]["doc_hash"] == "poisoned_hash"


def _unregistered_scenario_hits():
    return [
        {
            "chroma_id": "unregistered_hash",
            "text": "highest similarity but never registered on-chain",
            "similarity": 0.95,
            "metadata": {"doc_hash": "unregistered_hash", "source_file": "unregistered.txt"},
        },
        {
            "chroma_id": "neutral_hash",
            "text": "lower similarity but registered with neutral reputation",
            "similarity": 0.80,
            "metadata": {"doc_hash": "neutral_hash", "source_file": "neutral.txt"},
        },
    ]


def _unregistered_scenario_reputations(doc_hash):
    reputations = {
        "unregistered_hash": {"exists": False, "reputation": 0, "publisher": "0x" + "0" * 40},
        "neutral_hash": {"exists": True, "reputation": 0, "publisher": "0xabc"},
    }
    return reputations[doc_hash]


def test_trust_weighted_rerank_ranks_unregistered_candidate_below_neutral_registered_one():
    hits = _unregistered_scenario_hits()
    assert hits[0]["metadata"]["doc_hash"] == "unregistered_hash"
    assert hits[0]["similarity"] > hits[1]["similarity"]

    with (
        patch("retrieval.rerank.query", return_value=hits),
        patch("retrieval.rerank.verify_integrity", side_effect=_unregistered_scenario_reputations),
    ):
        results = trust_weighted_rerank("test query", k_retrieve=2, k_return=2)

    assert results[0]["metadata"]["doc_hash"] == "neutral_hash"
    assert results[1]["metadata"]["doc_hash"] == "unregistered_hash"


def test_trust_weighted_rerank_respects_k_return():
    hits = _poisoning_scenario_hits()

    with (
        patch("retrieval.rerank.query", return_value=hits),
        patch("retrieval.rerank.verify_integrity", side_effect=_poisoning_scenario_reputations),
    ):
        results = trust_weighted_rerank("test query", k_retrieve=2, k_return=1)

    assert len(results) == 1
```

(Add `from unittest.mock import patch` at the top of `tests/test_retrieval_rerank.py` if not already present.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_retrieval_rerank.py -v -k trust_weighted_rerank`
Expected: FAIL — `trust_weighted_rerank` not defined.

- [ ] **Step 3: Write minimal implementation**

Add import and function to `retrieval/rerank.py`. The function parameter is
named `query_text`, not `query` — `query` is already the name of the
imported function from `vector_store`, and a same-named parameter would
shadow it inside the function body:

```python
from retrieval.vector_store import query
```

```python
def trust_weighted_rerank(
    query_text: str, k_retrieve: int = 10, k_return: int = 3, alpha: float = 0.7
) -> list[dict]:
    hits = query(query_text, k=k_retrieve)

    candidates = []
    for hit in hits:
        doc_hash = hit["metadata"]["doc_hash"]
        exists, reputation = get_candidate_reputation(doc_hash)
        candidates.append({**hit, "exists": exists, "reputation": reputation})

    normalized_candidates = normalize_reputation(candidates)

    scored_candidates = [
        {
            **candidate,
            "score": hybrid_score(candidate["similarity"], candidate["normalized_reputation"], alpha),
        }
        for candidate in normalized_candidates
    ]
    scored_candidates.sort(key=lambda candidate: candidate["score"], reverse=True)
    return scored_candidates[:k_return]
```

No changes to the tests from Step 1 are needed: they already call
`trust_weighted_rerank("test query", ...)` positionally, and the mock
patch target `"retrieval.rerank.query"` already matches the import above.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_retrieval_rerank.py -v`
Expected: PASS (all tests from Tasks 1-3)

- [ ] **Step 5: Run the full project test suite to confirm no regressions**

Run: `uv run pytest`
Expected: PASS, same pass count as before plus the new `test_retrieval_rerank.py` tests, no live-service tests run (the `slow` marker is excluded by default per `pyproject.toml`).

- [ ] **Step 6: Commit**

```bash
git add retrieval/rerank.py tests/test_retrieval_rerank.py
git commit -m "feat: add trust_weighted_rerank orchestrator with poisoning-demotion and exists=False tests"
```

---

## Task 4: Demo script and DECISIONS.md entry

**Files:**
- Create: `scripts/demo_rerank.py`
- Modify: `DECISIONS.md`

**Interfaces:**
- Consumes: `retrieval.rerank.trust_weighted_rerank`, `retrieval.standard_rag.standard_rag_answer_context` (existing, `retrieval/standard_rag.py`).
- Produces: nothing consumed by later tasks — this is the milestone's terminal task.

- [ ] **Step 1: Write the demo script**

```python
"""Manual side-by-side comparison of standard (pure-cosine) RAG retrieval
versus trust-weighted re-ranking, for a query against real ingested data.

Requires a live Hardhat node (WEB3_PROVIDER_URI) and a populated Chroma
store — run `scripts/ingest.py` first if this is a fresh environment, or
if the Hardhat node has been restarted since data was last registered
(Hardhat's chain state is in-memory only and does not persist across
restarts, so previously-registered documents will fail verify_integrity()
after a restart until re-ingested).

Usage:
    uv run python scripts/demo_rerank.py "<query text>"
"""

import sys

from retrieval.rerank import trust_weighted_rerank
from retrieval.standard_rag import standard_rag_answer_context


def _print_ranking(title: str, hits: list[dict]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for rank, hit in enumerate(hits, start=1):
        source = hit["metadata"].get("source_file", "?")
        similarity = hit.get("similarity", float("nan"))
        score = hit.get("score")
        score_part = f", score={score:.4f}" if score is not None else ""
        print(f"{rank}. {source} (similarity={similarity:.4f}{score_part})")
        print(f"   {hit['text'][:120]}")


def main() -> None:
    query_text = sys.argv[1] if len(sys.argv) > 1 else "renewable energy sources"

    standard_hits = standard_rag_answer_context(query_text, k=3)
    trust_hits = trust_weighted_rerank(query_text, k_retrieve=10, k_return=3)

    _print_ranking(f"Standard RAG (pure cosine similarity) — query: {query_text!r}", standard_hits)
    _print_ranking(f"Trust-Weighted Rerank — query: {query_text!r}", trust_hits)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test the script's imports (no live services needed for this check)**

Run: `uv run python -c "import scripts.demo_rerank"`
Expected: No import errors (this only validates imports resolve; running `main()` requires live Hardhat + populated Chroma and is a manual/optional check, not part of this task's definition of done).

- [ ] **Step 3: Write the DECISIONS.md entry**

Append to `DECISIONS.md`:

```markdown
- **Milestone 6 reputation read: `verify_integrity()`, confirmed over
  `get_document()`.** Re-ranking only ever needs `exists` and `reputation`
  per candidate — never `ipfs_cid` or `timestamp` — and runs on every
  query against every retrieved candidate, so the cheaper 3-tuple contract
  call is the correct choice on both grounds. `get_document()`'s six-field
  getter remains reserved for Milestone 5's skip-check and Milestone 8's
  audit UI, per the Milestone 3 design.
- **`exists=False` no longer treated as `reputation=0`.** The original
  framing called `reputation=0` the "worst case" for an unregistered
  chunk. That stopped holding once `updateReputation()`'s delta was
  confirmed unbounded (Milestone 3): `registerDocument()` assigns
  `reputation=0` to every freshly-published, not-yet-reviewed document, so
  0 is the neutral midpoint of the reputation range, not a floor. A chunk
  present in Chroma with zero on-chain record at all is a strictly weaker
  provenance signal than a registered-but-neutral chunk, so it must
  normalize below every registered candidate in its batch, including one
  sitting at exactly `reputation=0`. `normalize_reputation()` implements
  this as a hard floor: `floor = min(reputation for existing candidates in
  the batch) - FLOOR_PENALTY`, or `0 - FLOOR_PENALTY` if no candidate in
  the batch is registered. `FLOOR_PENALTY = 50` was chosen relative to the
  sigmoid's `SCALE = 10` (5 half-widths) specifically so the floor's
  normalized value saturates near 0 regardless of the batch's actual
  minimum — an earlier `FLOOR_PENALTY = 1` was rejected during design
  because it produced only a ~0.025 normalized-reputation gap, too small
  to reliably survive a realistic similarity-score gap between candidates.
  Exclusion from top-K entirely was considered and rejected: a hard floor
  keeps the scoring function total (no filtering side-channel) and keeps
  unregistered chunks visible-but-demoted, which serves both Milestone 8's
  audit UI and Milestone 10's evaluation better than silently dropping
  them.
- **Normalization: sigmoid, not min-max.** `normalized_reputation =
  1 / (1 + exp(-reputation / SCALE))`, a fixed global curve rather than a
  per-batch min-max scale, for two reasons. First, Milestone 10's
  MRR/Precision@K evaluation needs a document's trust contribution to be
  reproducible across queries — min-max would make the same document
  normalize differently depending on which unrelated candidates happened
  to land in the same top-K for a given query, conflating the document's
  own reputation with the incidental composition of that query's
  candidate set. Second, min-max degenerates (divide-by-~0) when all K
  candidates share similar or identical reputation, which needs a
  special-cased fallback that sigmoid does not. `SCALE = 10` is a named
  module-level constant in `retrieval/rerank.py`, tunable alongside
  `alpha` once Milestone 9's real poisoned corpus is available.
- **`alpha = 0.7` confirmed as the default**, implemented as a named,
  overridable parameter on `hybrid_score()` and `trust_weighted_rerank()`
  rather than an inline literal, per spec — Milestone 9 will tune it
  empirically.
```

- [ ] **Step 4: Commit**

```bash
git add scripts/demo_rerank.py DECISIONS.md
git commit -m "docs: add trust-weighted rerank demo script and Milestone 6 DECISIONS.md entry"
```
