# Milestone 10: Evaluation, Final Documentation & Viva Prep — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `source_doc_id` to the demo-corpus manifest, build a quantitative retrieval evaluation (MRR/P@K for both retrieval modes + alpha sweep) and a qualitative hallucination comparison, then bring `README.md`, `DEMO.md`, `DECISIONS.md` up to date and mutually consistent and write a decision-grounded `VIVA.md`.

**Architecture:** Two new non-test scripts under `scripts/`. `scripts/evaluate.py` is pure metric functions (`reciprocal_rank`, `precision_at_k`, `mean_reciprocal_rank`, ground-truth resolution from the manifest, markdown rendering) plus a thin live-retrieval `main()`; it needs Hardhat + IPFS + a seeded Chroma store but **not** Ollama. `scripts/eval_hallucination.py` is a thin generator that reuses `scripts/evaluate.py`'s manifest helpers and needs Ollama on top. All numeric/prose logic is unit-tested offline against synthetic data; the two `eval/*.md` artifacts are produced once in a single live-services window and committed as captured output, the same pattern `DEMO.md` already uses.

**Tech Stack:** Python 3.11, pytest, existing project modules (`retrieval.standard_rag`, `retrieval.rerank`, `generation.llm_client`, `pipeline.embeddings`). No new dependencies — matplotlib is deliberately **not** added; the alpha-sweep chart is dependency-free ASCII.

**Spec:** `docs/superpowers/specs/2026-08-27-milestone-10-evaluation-design.md`

## Global Constraints

- **This is the last milestone in the DeRAG roadmap.** The closing `DECISIONS.md` entry must say so explicitly.
- **No new dependencies.** `pyproject.toml` is not modified. No matplotlib.
- **`manifest.json` must still regenerate byte-identically** (modulo the intended new field) — Milestone 9 DECISIONS requirement.
- **`source_doc_id` semantics:** the SciFact document a chunk's *text* came from. `str(doc_id)` for `scifact_*` chunks; JSON `null` for `FABRICATED_MISINFO_*` chunks. Never overload it with "which topic does this poison attack" — `ground_truth_chunk_id` already carries that.
- **The product/UI retrieval default `k_return=3` is not changed.** The evaluation uses a `k_return=10` override for full-ranking metrics, evaluation-only.
- **Result framing is "two positives, one honest negative":** TMAO (wide ~0.198 margin) and H. pylori (thin ~0.046 margin) both show trust-weighting correctly overriding a similarity-based fooling attempt; antibiotics is a deliberate non-fooling contrast presented as a **no-regression check**, not a failure. This exact framing — including the prose reasoning, not just the tables — must appear in `eval/results.md` and `eval/hallucination_qualitative.md` and be asserted by tests.
- **`VIVA.md` answers must be grounded in decisions actually on record** (DECISIONS.md / claude-mem), reconstructed before writing — not plausible-sounding invention.
- Run tests with `uv run pytest` from the `DeRAG/` directory. Fast suite: `uv run pytest`. Slow/live: `uv run pytest -m slow`.
- Commit after every green step. Conventional-commit prefixes (`feat:`, `test:`, `docs:`), `(m10)` scope.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `data/demo_corpus/manifest.json` | **Modify** — add `source_doc_id` to all 53 entries. |
| `scripts/build_demo_corpus.py` | **Modify** — `build_manifest()` emits `source_doc_id`. |
| `tests/test_scripts_build_demo_corpus.py` | **Modify** — assert `build_manifest()` output carries correct `source_doc_id`. |
| `tests/test_demo_corpus_fixtures.py` | **Modify** — schema assertion on the committed manifest. |
| `scripts/evaluate.py` | **Create** — metric functions, ground-truth resolution, alpha sweep, `render_results_md`, live `main()`. |
| `tests/test_scripts_evaluate.py` | **Create** — offline unit tests for every pure function incl. framing-language assertions. |
| `scripts/eval_hallucination.py` | **Create** — Standard-RAG-vs-DeRAG answer generation, `render_md`. |
| `tests/test_scripts_eval_hallucination.py` | **Create** — offline `render_md` / commentary-language test. |
| `eval/results.md` | **Create** (live run) — committed captured evaluation output. |
| `eval/hallucination_qualitative.md` | **Create** (live run) — committed captured side-by-side answers. |
| `eval/.gitkeep` | **Delete** once `results.md` exists (optional tidy). |
| `README.md` | **Modify** — fill in "Services to start before running anything"; add M10 status entry. |
| `VIVA.md` | **Create** — anticipated viva Q&A grouped by DL / ISR / Blockchain. |
| `DEMO.md` | **Modify** — re-verify against live numbers; explain the antibiotics similarity-pair discrepancy. |
| `DECISIONS.md` | **Modify** — closing Milestone 10 entry. |

---

## Task 1: `source_doc_id` in the manifest and the builder

**Files:**
- Modify: `scripts/build_demo_corpus.py` (`build_manifest`, ~lines 145-187)
- Modify: `data/demo_corpus/manifest.json` (all 53 entries)
- Test: `tests/test_scripts_build_demo_corpus.py`, `tests/test_demo_corpus_fixtures.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: every `manifest.json` chunk entry has key `source_doc_id` positioned immediately after `chunk_id`; value is `str(doc_id)` (digits) for `scifact_*` chunks, `None`/`null` for `FABRICATED_MISINFO_*` chunks. Task 2 relies on this.

- [ ] **Step 1: Write the failing builder test**

Add to `tests/test_scripts_build_demo_corpus.py`:

```python
def test_build_manifest_sets_source_doc_id():
    filler = [
        {"doc_id": 111, "title": "Doc 111", "slug": "doc_111", "text": "t"},
        {"doc_id": 222, "title": "Doc 222", "slug": "doc_222", "text": "t"},
    ]
    chunks = build_manifest(filler)["chunks"]
    by_id = {c["chunk_id"]: c for c in chunks}

    # filler + targeted legit -> stringified SciFact doc id, taken from chunk_id
    for c in chunks:
        if c["chunk_id"].startswith("scifact_"):
            expected = c["chunk_id"].split("_")[1]
            assert c["source_doc_id"] == expected
            assert c["source_doc_id"].isdigit()

    # poison -> None
    for c in chunks:
        if c["is_poisoned"]:
            assert c["source_doc_id"] is None

    # key order: source_doc_id immediately follows chunk_id
    sample = by_id["scifact_111_doc_111"]
    keys = list(sample.keys())
    assert keys[keys.index("chunk_id") + 1] == "source_doc_id"
```

- [ ] **Step 2: Run it, expect failure**

Run: `uv run pytest tests/test_scripts_build_demo_corpus.py::test_build_manifest_sets_source_doc_id -v`
Expected: FAIL — `KeyError: 'source_doc_id'`.

- [ ] **Step 3: Implement in `build_manifest`**

In `scripts/build_demo_corpus.py`, add `source_doc_id` right after `chunk_id` in each of the three append blocks:

```python
    for f in filler:
        chunk_id = f"scifact_{f['doc_id']}_{f['slug']}"
        chunks.append(
            {
                "filename": f"{chunk_id}.txt",
                "chunk_id": chunk_id,
                "source_doc_id": str(f["doc_id"]),
                "is_poisoned": False,
                "publisher_key": "PUBLISHER_1_KEY",
                "target_reputation": 0,
                "target_queries": [],
                "ground_truth_chunk_id": None,
            }
        )
    for doc_id, meta in TARGETS.items():
        chunk_id = f"scifact_{doc_id}_{meta['slug']}"
        chunks.append(
            {
                "filename": f"{chunk_id}.txt",
                "chunk_id": chunk_id,
                "source_doc_id": str(doc_id),
                "is_poisoned": False,
                "publisher_key": "PUBLISHER_2_KEY",
                "target_reputation": 10,
                "target_queries": [meta["query"]],
                "ground_truth_chunk_id": chunk_id,
            }
        )
    for chunk_id, pmeta in POISON.items():
        target = TARGETS[pmeta["target_doc_id"]]
        gt_chunk_id = f"scifact_{pmeta['target_doc_id']}_{target['slug']}"
        chunks.append(
            {
                "filename": pmeta["filename"],
                "chunk_id": chunk_id,
                "source_doc_id": None,
                "is_poisoned": True,
                "publisher_key": "PUBLISHER_5_KEY",
                "target_reputation": -50,
                "target_queries": [target["query"]],
                "ground_truth_chunk_id": gt_chunk_id,
            }
        )
```

- [ ] **Step 4: Run it, expect pass**

Run: `uv run pytest tests/test_scripts_build_demo_corpus.py::test_build_manifest_sets_source_doc_id -v`
Expected: PASS.

- [ ] **Step 5: Apply the same field to the committed `manifest.json` deterministically**

The SciFact archive is not cached locally and a full rebuild needs network. Transform the committed file in place instead — same formatting the builder uses (`json.dumps(..., indent=2, ensure_ascii=False) + "\n"`), new key right after `chunk_id`:

```bash
uv run python - <<'PY'
import json, re
from pathlib import Path

p = Path("data/demo_corpus/manifest.json")
manifest = json.loads(p.read_text(encoding="utf-8"))
for c in manifest["chunks"]:
    if c["chunk_id"].startswith("scifact_"):
        src = c["chunk_id"].split("_")[1]
        assert src.isdigit(), c["chunk_id"]
        source_doc_id = src
    else:
        source_doc_id = None
    # rebuild the dict so source_doc_id lands right after chunk_id
    rebuilt = {}
    for k, v in c.items():
        rebuilt[k] = v
        if k == "chunk_id":
            rebuilt["source_doc_id"] = source_doc_id
    c.clear()
    c.update(rebuilt)

p.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print("updated", len(manifest["chunks"]), "entries")
PY
```

Then verify the diff adds **only** `source_doc_id` lines:

Run: `git diff --stat data/demo_corpus/manifest.json && git diff data/demo_corpus/manifest.json | grep -E '^[+-]' | grep -v source_doc_id | grep -vE '^(\+\+\+|---)'`
Expected: the second `grep` prints nothing (no non-`source_doc_id` line changed).

- [ ] **Step 6: Write the failing fixtures schema test**

Add to `tests/test_demo_corpus_fixtures.py`:

```python
def test_every_manifest_entry_has_source_doc_id():
    for c in _manifest_chunks():
        assert "source_doc_id" in c, c["chunk_id"]
        if c["is_poisoned"]:
            assert c["source_doc_id"] is None
        else:
            assert isinstance(c["source_doc_id"], str) and c["source_doc_id"].isdigit()
            assert c["source_doc_id"] in c["chunk_id"]
```

- [ ] **Step 7: Run the demo-corpus fast tests, expect pass**

Run: `uv run pytest tests/test_demo_corpus_fixtures.py tests/test_scripts_build_demo_corpus.py -v`
Expected: all pass (the slow `test_full_build_is_deterministic_and_well_formed` is deselected without `-m slow`).

- [ ] **Step 8: Commit**

```bash
git add scripts/build_demo_corpus.py data/demo_corpus/manifest.json tests/test_scripts_build_demo_corpus.py tests/test_demo_corpus_fixtures.py
git commit -m "feat(m10): add source_doc_id to demo-corpus manifest and builder"
```

---

## Task 2: `scripts/evaluate.py` — pure metric & ground-truth functions

**Files:**
- Create: `scripts/evaluate.py`
- Test: `tests/test_scripts_evaluate.py`

**Interfaces:**
- Consumes: `manifest.json` shape from Task 1 (`source_doc_id`, `ground_truth_chunk_id`, `target_queries`, `is_poisoned`, `chunk_id`, `filename`).
- Produces (used by Task 3 and Task 4):
  - `CASE_LABELS: dict[str, str]` — poison `chunk_id` → short display label.
  - `load_manifest_chunks(corpus_dir: Path = CORPUS_DIR) -> list[dict]`
  - `source_doc_id_by_filename(chunks: list[dict]) -> dict[str, str | None]`
  - `build_ground_truth(chunks: list[dict]) -> dict[str, str]` — query string → ground-truth `source_doc_id`
  - `gather_cases(chunks: list[dict]) -> list[dict]` — each: `{"chunk_id","label","query","gt_doc_id","poison_file","legit_file"}`
  - `hits_to_doc_ids(hits: list[dict], sdi_by_file: dict[str, str | None]) -> list[str | None]`
  - `reciprocal_rank(ranked: list[str | None], gt_id: str) -> float`
  - `precision_at_k(ranked: list[str | None], gt_id: str, k: int) -> float`
  - `mean_reciprocal_rank(rr_values: list[float]) -> float`
  - `evaluate_query(ranked: list[str | None], gt_id: str) -> dict` — `{"rr","p_at_1","p_at_3"}`

- [ ] **Step 1: Write failing tests for the metric functions**

Create `tests/test_scripts_evaluate.py`:

```python
import json
from pathlib import Path

import pytest

from scripts.evaluate import (
    build_ground_truth,
    evaluate_query,
    gather_cases,
    hits_to_doc_ids,
    mean_reciprocal_rank,
    precision_at_k,
    reciprocal_rank,
)


def test_reciprocal_rank_first_position():
    assert reciprocal_rank(["7", "1", "2"], "7") == 1.0


def test_reciprocal_rank_third_position():
    assert reciprocal_rank(["1", "2", "7"], "7") == pytest.approx(1 / 3)


def test_reciprocal_rank_absent_is_zero():
    assert reciprocal_rank(["1", "2", None], "7") == 0.0


def test_precision_at_k_single_relevant_in_window():
    assert precision_at_k(["1", "7", "2"], "7", 1) == 0.0
    assert precision_at_k(["7", "1", "2"], "7", 1) == 1.0
    assert precision_at_k(["1", "7", "2"], "7", 3) == pytest.approx(1 / 3)


def test_precision_at_k_relevant_outside_window_is_zero():
    assert precision_at_k(["1", "2", "3", "7"], "7", 3) == 0.0


def test_mean_reciprocal_rank():
    assert mean_reciprocal_rank([1.0, 0.5, 0.0]) == pytest.approx(0.5)
    assert mean_reciprocal_rank([]) == 0.0


def test_evaluate_query_bundles_metrics():
    out = evaluate_query(["1", "2", "7", "9"], "7")
    assert out == {"rr": pytest.approx(1 / 3), "p_at_1": 0.0, "p_at_3": 0.0}
```

- [ ] **Step 2: Run, expect ImportError / collection failure**

Run: `uv run pytest tests/test_scripts_evaluate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.evaluate'`.

- [ ] **Step 3: Create `scripts/evaluate.py` with the pure core**

```python
"""Milestone 10 retrieval-quality evaluation.

Computes MRR@10 and Precision@K for Standard RAG and trust-weighted
rerank across the three Milestone 9 target queries, plus an alpha sweep,
and renders eval/results.md.

Ground truth comes from data/demo_corpus/manifest.json: each target query
has exactly one relevant SciFact document (the real abstract on that
topic). A retrieved hit is relevant iff its source_doc_id equals the
query's ground-truth source_doc_id; poison chunks (source_doc_id is None)
are never relevant.

Live run (needs a live Hardhat node + IPFS daemon + a Chroma store seeded
by scripts/seed_reputation.py; Ollama NOT required):

    set -a; . ./.env; set +a
    uv run python scripts/seed_reputation.py
    uv run python scripts/check_candidate_window.py      # [preflight] PASS
    uv run python scripts/evaluate.py                    # writes eval/results.md
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CORPUS_DIR = Path(__file__).resolve().parent.parent / "data" / "demo_corpus"
EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"
DEFAULT_ALPHAS = (0.0, 0.3, 0.5, 0.7, 1.0)
SHIPPED_ALPHA = 0.7

CASE_LABELS = {
    "poison_tmao_redmeat_neutral": "TMAO / red meat",
    "poison_hpylori_protective": "H. pylori / gastric cancer",
    "poison_antibiotics_cdiff_protective": "antibiotics / C. difficile",
}


def load_manifest_chunks(corpus_dir: Path = CORPUS_DIR) -> list[dict]:
    return json.loads((corpus_dir / "manifest.json").read_text(encoding="utf-8"))["chunks"]


def source_doc_id_by_filename(chunks: list[dict]) -> dict[str, str | None]:
    return {c["filename"]: c["source_doc_id"] for c in chunks}


def build_ground_truth(chunks: list[dict]) -> dict[str, str]:
    by_id = {c["chunk_id"]: c for c in chunks}
    gt: dict[str, str] = {}
    for c in chunks:
        if not c["is_poisoned"] or not c["target_queries"]:
            continue
        legit = by_id[c["ground_truth_chunk_id"]]
        for query in c["target_queries"]:
            gt[query] = legit["source_doc_id"]
    return gt


def gather_cases(chunks: list[dict]) -> list[dict]:
    by_id = {c["chunk_id"]: c for c in chunks}
    cases = []
    for chunk_id, label in CASE_LABELS.items():
        poison = by_id[chunk_id]
        legit = by_id[poison["ground_truth_chunk_id"]]
        cases.append(
            {
                "chunk_id": chunk_id,
                "label": label,
                "query": poison["target_queries"][0],
                "gt_doc_id": legit["source_doc_id"],
                "poison_file": poison["filename"],
                "legit_file": legit["filename"],
            }
        )
    return cases


def hits_to_doc_ids(hits: list[dict], sdi_by_file: dict[str, str | None]) -> list[str | None]:
    return [sdi_by_file.get(h["metadata"]["source_file"]) for h in hits]


def reciprocal_rank(ranked: list[str | None], gt_id: str) -> float:
    for i, doc_id in enumerate(ranked, start=1):
        if doc_id == gt_id:
            return 1.0 / i
    return 0.0


def precision_at_k(ranked: list[str | None], gt_id: str, k: int) -> float:
    top_k = ranked[:k]
    if not top_k:
        return 0.0
    return sum(1 for d in top_k if d == gt_id) / k


def mean_reciprocal_rank(rr_values: list[float]) -> float:
    return sum(rr_values) / len(rr_values) if rr_values else 0.0


def evaluate_query(ranked: list[str | None], gt_id: str) -> dict:
    return {
        "rr": reciprocal_rank(ranked, gt_id),
        "p_at_1": precision_at_k(ranked, gt_id, 1),
        "p_at_3": precision_at_k(ranked, gt_id, 3),
    }
```

- [ ] **Step 4: Run the metric tests, expect pass**

Run: `uv run pytest tests/test_scripts_evaluate.py -v`
Expected: the 7 tests from Step 1 PASS.

- [ ] **Step 5: Write failing tests for the manifest helpers**

Append to `tests/test_scripts_evaluate.py`:

```python
_MANIFEST_SLICE = {
    "chunks": [
        {"filename": "scifact_14803797_tmao_redmeat.txt",
         "chunk_id": "scifact_14803797_tmao_redmeat", "source_doc_id": "14803797",
         "is_poisoned": False, "publisher_key": "PUBLISHER_2_KEY", "target_reputation": 10,
         "target_queries": ["Q-TMAO"], "ground_truth_chunk_id": "scifact_14803797_tmao_redmeat"},
        {"filename": "FABRICATED_MISINFO_tmao_redmeat_neutral.txt",
         "chunk_id": "poison_tmao_redmeat_neutral", "source_doc_id": None,
         "is_poisoned": True, "publisher_key": "PUBLISHER_5_KEY", "target_reputation": -50,
         "target_queries": ["Q-TMAO"], "ground_truth_chunk_id": "scifact_14803797_tmao_redmeat"},
        {"filename": "scifact_999_filler.txt", "chunk_id": "scifact_999_filler",
         "source_doc_id": "999", "is_poisoned": False, "publisher_key": "PUBLISHER_1_KEY",
         "target_reputation": 0, "target_queries": [], "ground_truth_chunk_id": None},
    ]
}


def test_build_ground_truth_resolves_to_legit_counterpart_source_doc_id():
    gt = build_ground_truth(_MANIFEST_SLICE["chunks"])
    assert gt == {"Q-TMAO": "14803797"}


def test_source_doc_id_by_filename_maps_poison_to_none():
    from scripts.evaluate import source_doc_id_by_filename
    sdi = source_doc_id_by_filename(_MANIFEST_SLICE["chunks"])
    assert sdi["FABRICATED_MISINFO_tmao_redmeat_neutral.txt"] is None
    assert sdi["scifact_14803797_tmao_redmeat.txt"] == "14803797"


def test_hits_to_doc_ids_join():
    from scripts.evaluate import source_doc_id_by_filename
    sdi = source_doc_id_by_filename(_MANIFEST_SLICE["chunks"])
    hits = [
        {"metadata": {"source_file": "FABRICATED_MISINFO_tmao_redmeat_neutral.txt"}},
        {"metadata": {"source_file": "scifact_14803797_tmao_redmeat.txt"}},
    ]
    assert hits_to_doc_ids(hits, sdi) == [None, "14803797"]


def test_gather_cases_uses_real_committed_manifest():
    from scripts.evaluate import load_manifest_chunks
    cases = gather_cases(load_manifest_chunks())
    assert {c["label"] for c in cases} == {
        "TMAO / red meat", "H. pylori / gastric cancer", "antibiotics / C. difficile"
    }
    for c in cases:
        assert c["query"].endswith("?")
        assert c["gt_doc_id"].isdigit()
        assert c["poison_file"].startswith("FABRICATED_MISINFO_")
        assert c["legit_file"].startswith("scifact_")
```

- [ ] **Step 6: Run, expect pass** (helpers already implemented in Step 3)

Run: `uv run pytest tests/test_scripts_evaluate.py -v`
Expected: all PASS. If `test_gather_cases_uses_real_committed_manifest` fails, Task 1's manifest edit is incomplete — fix that first.

- [ ] **Step 7: Commit**

```bash
git add scripts/evaluate.py tests/test_scripts_evaluate.py
git commit -m "feat(m10): evaluate.py pure metric and ground-truth functions"
```

---

## Task 3: `scripts/evaluate.py` — orchestration, alpha sweep & `render_results_md`

**Files:**
- Modify: `scripts/evaluate.py`
- Test: `tests/test_scripts_evaluate.py`

**Interfaces:**
- Consumes: everything from Task 2.
- Produces (used by Task 5's live run):
  - `run_evaluation(alphas=DEFAULT_ALPHAS) -> dict` — does live retrieval; returns
    `{"per_query": [ {label, query, gt_doc_id, poison_file, legit_file,
    "standard": {rr,p_at_1,p_at_3}, "derag": {...},
    "standard_dump": [source_file,...], "derag_dump": [...]} ],
    "sweep": [ {"alpha": float, "mrr": float, "mean_p_at_1": float} ],
    "alphas": [float,...]}`
  - `render_results_md(results: dict) -> str`
  - module string constants `FRAMING_HEADLINE`, `NARRATIVE_TMAO`, `NARRATIVE_HPYLORI`, `NARRATIVE_ANTIBIOTICS`, `ALPHA_SANITY_NOTE`
  - `main()` writes `eval/results.md` and prints it.

- [ ] **Step 1: Write failing tests for rendering & sweep aggregation**

Append to `tests/test_scripts_evaluate.py`:

```python
from scripts.evaluate import (  # noqa: E402
    ALPHA_SANITY_NOTE,
    FRAMING_HEADLINE,
    NARRATIVE_ANTIBIOTICS,
    NARRATIVE_HPYLORI,
    NARRATIVE_TMAO,
    render_results_md,
)

_RESULTS = {
    "alphas": [0.0, 0.3, 0.5, 0.7, 1.0],
    "per_query": [
        {"label": "TMAO / red meat", "query": "Q1", "gt_doc_id": "14803797",
         "poison_file": "FABRICATED_MISINFO_tmao_redmeat_neutral.txt",
         "legit_file": "scifact_14803797_tmao_redmeat.txt",
         "standard": {"rr": 0.5, "p_at_1": 0.0, "p_at_3": 1 / 3},
         "derag": {"rr": 1.0, "p_at_1": 1.0, "p_at_3": 1 / 3},
         "standard_dump": ["FABRICATED_MISINFO_tmao_redmeat_neutral.txt", "scifact_14803797_tmao_redmeat.txt"],
         "derag_dump": ["scifact_14803797_tmao_redmeat.txt", "FABRICATED_MISINFO_tmao_redmeat_neutral.txt"]},
        {"label": "H. pylori / gastric cancer", "query": "Q2", "gt_doc_id": "2052720",
         "poison_file": "FABRICATED_MISINFO_hpylori_protective.txt",
         "legit_file": "scifact_2052720_hpylori_gastric_cancer.txt",
         "standard": {"rr": 0.5, "p_at_1": 0.0, "p_at_3": 1 / 3},
         "derag": {"rr": 1.0, "p_at_1": 1.0, "p_at_3": 1 / 3},
         "standard_dump": ["FABRICATED_MISINFO_hpylori_protective.txt", "scifact_2052720_hpylori_gastric_cancer.txt"],
         "derag_dump": ["scifact_2052720_hpylori_gastric_cancer.txt", "FABRICATED_MISINFO_hpylori_protective.txt"]},
        {"label": "antibiotics / C. difficile", "query": "Q3", "gt_doc_id": "6372244",
         "poison_file": "FABRICATED_MISINFO_antibiotics_cdiff_protective.txt",
         "legit_file": "scifact_6372244_antibiotics_cdiff.txt",
         "standard": {"rr": 1.0, "p_at_1": 1.0, "p_at_3": 1 / 3},
         "derag": {"rr": 1.0, "p_at_1": 1.0, "p_at_3": 1 / 3},
         "standard_dump": ["scifact_6372244_antibiotics_cdiff.txt", "FABRICATED_MISINFO_antibiotics_cdiff_protective.txt"],
         "derag_dump": ["scifact_6372244_antibiotics_cdiff.txt", "FABRICATED_MISINFO_antibiotics_cdiff_protective.txt"]},
    ],
    "sweep": [
        {"alpha": 0.0, "mrr": 0.55, "mean_p_at_1": 0.33},
        {"alpha": 0.3, "mrr": 1.0, "mean_p_at_1": 1.0},
        {"alpha": 0.5, "mrr": 1.0, "mean_p_at_1": 1.0},
        {"alpha": 0.7, "mrr": 1.0, "mean_p_at_1": 1.0},
        {"alpha": 1.0, "mrr": 0.667, "mean_p_at_1": 0.667},
    ],
}


def test_render_results_md_carries_the_framing_prose():
    md = render_results_md(_RESULTS)
    # not just tables — the rhetorical framing must survive templating
    assert FRAMING_HEADLINE in md
    assert NARRATIVE_TMAO in md
    assert NARRATIVE_HPYLORI in md
    assert NARRATIVE_ANTIBIOTICS in md
    assert ALPHA_SANITY_NOTE in md


def test_render_results_md_framing_constants_say_the_right_things():
    # guard the constants themselves so an edit can't hollow them out
    assert "two-positive" in FRAMING_HEADLINE and "one-honest-negative" in FRAMING_HEADLINE
    assert "0.198" in NARRATIVE_TMAO
    assert "0.046" in NARRATIVE_HPYLORI and "robustness" in NARRATIVE_HPYLORI
    assert "no-regression" in NARRATIVE_ANTIBIOTICS
    assert "1.0" in ALPHA_SANITY_NOTE and "similarity" in ALPHA_SANITY_NOTE


def test_render_results_md_has_tables_and_dumps():
    md = render_results_md(_RESULTS)
    assert "| TMAO / red meat |" in md
    assert "| H. pylori / gastric cancer |" in md
    assert "| antibiotics / C. difficile |" in md
    assert "Alpha sweep" in md
    assert "0.7" in md and "shipped default" in md
    # raw ranking dumps present for auditability
    assert "scifact_14803797_tmao_redmeat.txt" in md
    assert "uv run python scripts/evaluate.py" in md


def test_render_results_md_plateau_reading_reacts_to_data():
    wide = render_results_md(_RESULTS)  # 0.3/0.5/0.7 all tie at max MRR
    assert "plateau" in wide.lower()
    knife = dict(_RESULTS)
    knife["sweep"] = [
        {"alpha": 0.0, "mrr": 0.4, "mean_p_at_1": 0.33},
        {"alpha": 0.3, "mrr": 0.5, "mean_p_at_1": 0.33},
        {"alpha": 0.5, "mrr": 0.6, "mean_p_at_1": 0.33},
        {"alpha": 0.7, "mrr": 1.0, "mean_p_at_1": 1.0},
        {"alpha": 1.0, "mrr": 0.6, "mean_p_at_1": 0.33},
    ]
    assert "sensitive" in render_results_md(knife).lower()
```

- [ ] **Step 2: Run, expect failure**

Run: `uv run pytest tests/test_scripts_evaluate.py -k render_results_md -v`
Expected: FAIL — `ImportError: cannot import name 'render_results_md'`.

- [ ] **Step 3: Implement the constants, rendering, orchestration & `main()`**

Append to `scripts/evaluate.py`:

```python
FRAMING_HEADLINE = (
    "This is a **two-positive / one-honest-negative** result — not a uniform "
    "success story and not a single cherry-picked win. On two independent "
    "topics (TMAO, H. pylori) trust-weighted rerank correctly overrides a "
    "similarity-based fooling attempt; on a third (antibiotics) raw similarity "
    "already ranks the legitimate chunk first and DeRAG correctly does not "
    "over-trigger. A mechanism that fires when it should and stays quiet when "
    "it should not is the more credible result."
)

NARRATIVE_TMAO = (
    "**TMAO / red meat** — the fabricated chunk beats the real abstract on raw "
    "cosine similarity by a wide margin (~0.198). Standard RAG is fooled (its "
    "top-1 is the poison, so P@1 = 0); trust-weighted rerank uses the on-chain "
    "reputation signal to flip the real abstract back to rank 1 (P@1 = 1)."
)

NARRATIVE_HPYLORI = (
    "**H. pylori / gastric cancer** — the same override on an independent topic, "
    "but at a thinner, more drift-exposed margin (~0.046). That thin margin is "
    "itself an honest, reportable data point about the mechanism's robustness "
    "envelope, not a weakness to hide: the flip still holds and is asserted as a "
    "hard regression case, but with less headroom than TMAO."
)

NARRATIVE_ANTIBIOTICS = (
    "**antibiotics / C. difficile** — a deliberate non-fooling contrast and a "
    "no-regression check. Under its natural query raw similarity already ranks "
    "the legit chunk first, so there is nothing for trust-weighting to override. "
    "DeRAG leaves the already-correct order alone (P@1 stays 1 in both modes) — "
    "proof the mechanism does not distort a case it does not need to touch."
)

ALPHA_SANITY_NOTE = (
    "`alpha = 1.0` collapses the hybrid score to pure similarity, so its row "
    "reproduces Standard RAG's MRR exactly — a built-in sanity check on the sweep."
)


def _fmt(x: float) -> str:
    return f"{x:.3f}"


def _ascii_bar(value: float, max_value: float, width: int = 32) -> str:
    if max_value <= 0:
        return "-" * width
    filled = int(round((value / max_value) * width))
    filled = max(0, min(width, filled))
    return "#" * filled + "-" * (width - filled)


def _plateau_reading(sweep: list[dict]) -> str:
    tunable = [row for row in sweep if row["alpha"] < 1.0]
    best = max(row["mrr"] for row in tunable)
    at_best = [row["alpha"] for row in tunable if abs(row["mrr"] - best) < 1e-9]
    shipped_at_best = any(abs(a - SHIPPED_ALPHA) < 1e-9 for a in at_best)
    if len(at_best) >= 3 and shipped_at_best:
        return (
            f"MRR is flat at its maximum ({_fmt(best)}) across alpha "
            f"{', '.join(_fmt(a) for a in at_best)} — a wide plateau, so the "
            f"shipped alpha = {SHIPPED_ALPHA} is a safe pick, not a knife-edge tune."
        )
    return (
        f"MRR peaks only at alpha {', '.join(_fmt(a) for a in at_best)} — the "
        f"result is sensitive to alpha; the shipped value {SHIPPED_ALPHA} "
        f"{'sits at that peak' if shipped_at_best else 'does NOT sit at that peak — investigate'}."
    )


def _mode_table(per_query: list[dict]) -> str:
    lines = [
        "| Query | SR P@1 | SR P@3 | SR RR | DeRAG P@1 | DeRAG P@3 | DeRAG RR |",
        "|-------|-------:|-------:|------:|----------:|----------:|---------:|",
    ]
    agg = {"s1": [], "s3": [], "sr": [], "d1": [], "d3": [], "dr": []}
    for row in per_query:
        s, d = row["standard"], row["derag"]
        agg["s1"].append(s["p_at_1"]); agg["s3"].append(s["p_at_3"]); agg["sr"].append(s["rr"])
        agg["d1"].append(d["p_at_1"]); agg["d3"].append(d["p_at_3"]); agg["dr"].append(d["rr"])
        lines.append(
            f"| {row['label']} | {_fmt(s['p_at_1'])} | {_fmt(s['p_at_3'])} | {_fmt(s['rr'])} "
            f"| {_fmt(d['p_at_1'])} | {_fmt(d['p_at_3'])} | {_fmt(d['rr'])} |"
        )
    mean = {k: mean_reciprocal_rank(v) for k, v in agg.items()}
    lines.append(
        f"| **Mean** | **{_fmt(mean['s1'])}** | **{_fmt(mean['s3'])}** | **{_fmt(mean['sr'])}** "
        f"| **{_fmt(mean['d1'])}** | **{_fmt(mean['d3'])}** | **{_fmt(mean['dr'])}** |"
    )
    lines.append("")
    lines.append(f"MRR@10 — Standard RAG: **{_fmt(mean['sr'])}**  ·  DeRAG: **{_fmt(mean['dr'])}**")
    return "\n".join(lines)


def _sweep_block(sweep: list[dict]) -> str:
    max_mrr = max((row["mrr"] for row in sweep), default=1.0)
    lines = [
        "| alpha | MRR@10 | mean P@1 | MRR@10 (bar) |",
        "|------:|-------:|---------:|--------------|",
    ]
    for row in sweep:
        tag = "  ← shipped default" if abs(row["alpha"] - SHIPPED_ALPHA) < 1e-9 else ""
        lines.append(
            f"| {_fmt(row['alpha'])}{tag} | {_fmt(row['mrr'])} | {_fmt(row['mean_p_at_1'])} "
            f"| `{_ascii_bar(row['mrr'], max_mrr)}` |"
        )
    lines.append("")
    lines.append(_plateau_reading(sweep))
    lines.append("")
    lines.append(ALPHA_SANITY_NOTE)
    return "\n".join(lines)


_NARRATIVE_BY_LABEL = {
    "TMAO / red meat": NARRATIVE_TMAO,
    "H. pylori / gastric cancer": NARRATIVE_HPYLORI,
    "antibiotics / C. difficile": NARRATIVE_ANTIBIOTICS,
}


def render_results_md(results: dict) -> str:
    pq = results["per_query"]
    parts = [
        "# DeRAG Retrieval Evaluation (Milestone 10)",
        "",
        FRAMING_HEADLINE,
        "",
        "## Metrics",
        "",
        "Ground truth: each of the three target queries has exactly one relevant "
        "SciFact document — the real abstract on that topic (`source_doc_id` in "
        "`data/demo_corpus/manifest.json`). A retrieved hit is relevant iff its "
        "`source_doc_id` matches; poison chunks are never relevant. "
        "`standard_rag_answer_context(q, k=10)` vs "
        "`trust_weighted_rerank(q, k_retrieve=10, k_return=10)` — the `k_return=10` "
        "override is evaluation-only; the product default stays 3.",
        "",
        _mode_table(pq),
        "",
        "## Per-query reading",
        "",
    ]
    for row in pq:
        parts.append(_NARRATIVE_BY_LABEL[row["label"]])
        parts.append("")
    parts += [
        "## Alpha sweep",
        "",
        f"`trust_weighted_rerank` re-run at alpha ∈ "
        f"{', '.join(_fmt(a) for a in results['alphas'])} across all three queries:",
        "",
        _sweep_block(results["sweep"]),
        "",
        "## Reproduction",
        "",
        "```bash",
        "# Hardhat node + deploy + IPFS daemon all live; then:",
        "set -a; . ./.env; set +a",
        "uv run python scripts/seed_reputation.py",
        "uv run python scripts/check_candidate_window.py      # [preflight] PASS",
        "uv run python scripts/evaluate.py                    # regenerates this file",
        "```",
        "",
        "## Raw ranking dumps (top-10, source_file order)",
        "",
    ]
    for row in pq:
        parts.append(f"### {row['label']}")
        parts.append(f"- query: `{row['query']}`")
        parts.append(f"- ground-truth doc: `{row['gt_doc_id']}` (`{row['legit_file']}`)")
        parts.append(f"- Standard RAG: `{row['standard_dump']}`")
        parts.append(f"- DeRAG:        `{row['derag_dump']}`")
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def run_evaluation(alphas: tuple[float, ...] = DEFAULT_ALPHAS) -> dict:
    from retrieval.rerank import trust_weighted_rerank
    from retrieval.standard_rag import standard_rag_answer_context

    chunks = load_manifest_chunks()
    sdi_by_file = source_doc_id_by_filename(chunks)
    cases = gather_cases(chunks)

    per_query = []
    for case in cases:
        std_hits = standard_rag_answer_context(case["query"], k=10)
        drag_hits = trust_weighted_rerank(case["query"], k_retrieve=10, k_return=10)
        per_query.append(
            {
                "label": case["label"],
                "query": case["query"],
                "gt_doc_id": case["gt_doc_id"],
                "poison_file": case["poison_file"],
                "legit_file": case["legit_file"],
                "standard": evaluate_query(hits_to_doc_ids(std_hits, sdi_by_file), case["gt_doc_id"]),
                "derag": evaluate_query(hits_to_doc_ids(drag_hits, sdi_by_file), case["gt_doc_id"]),
                "standard_dump": [h["metadata"]["source_file"] for h in std_hits],
                "derag_dump": [h["metadata"]["source_file"] for h in drag_hits],
            }
        )

    sweep = []
    for alpha in alphas:
        rr_vals, p1_vals = [], []
        for case in cases:
            hits = trust_weighted_rerank(case["query"], k_retrieve=10, k_return=10, alpha=alpha)
            ids = hits_to_doc_ids(hits, sdi_by_file)
            rr_vals.append(reciprocal_rank(ids, case["gt_doc_id"]))
            p1_vals.append(precision_at_k(ids, case["gt_doc_id"], 1))
        sweep.append(
            {
                "alpha": float(alpha),
                "mrr": mean_reciprocal_rank(rr_vals),
                "mean_p_at_1": sum(p1_vals) / len(p1_vals),
            }
        )

    return {"per_query": per_query, "sweep": sweep, "alphas": [float(a) for a in alphas]}


def main() -> None:
    parser = argparse.ArgumentParser(description="DeRAG retrieval evaluation (Milestone 10).")
    parser.add_argument("--out", default=str(EVAL_DIR / "results.md"))
    args = parser.parse_args()

    results = run_evaluation()
    md = render_results_md(results)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    print(md)
    print(f"\n[evaluate] wrote {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the full evaluate test file, expect pass**

Run: `uv run pytest tests/test_scripts_evaluate.py -v`
Expected: all PASS.

- [ ] **Step 5: Sanity-check the module imports cleanly without live services**

Run: `uv run python -c "import scripts.evaluate as e; print(e.render_results_md.__doc__ is None, len(e.CASE_LABELS))"`
Expected: prints `True 3` (no ChromaDB/web3 import at module load — those are inside `run_evaluation`).

- [ ] **Step 6: Commit**

```bash
git add scripts/evaluate.py tests/test_scripts_evaluate.py
git commit -m "feat(m10): evaluate.py orchestration, alpha sweep, results.md renderer"
```

---

## Task 4: `scripts/eval_hallucination.py` — Standard-RAG-vs-DeRAG answers

**Files:**
- Create: `scripts/eval_hallucination.py`
- Test: `tests/test_scripts_eval_hallucination.py`

**Interfaces:**
- Consumes: `gather_cases`, `load_manifest_chunks` from Task 2/3; `build_prompt`, `generate_blocking` from `generation.llm_client`; `standard_rag_answer_context`, `trust_weighted_rerank`.
- Produces (used by Task 6's live run):
  - `retrieve(query: str, mode: str) -> list[dict]` — `mode` in `{"standard","derag"}`, product-default top-3.
  - `run_case(query: str, mode: str) -> dict` — `{"mode","answer","sources": [source_file,...]}` (calls Ollama).
  - `render_md(cases: list[dict]) -> str` — `cases` each `{"label","query","standard": run_case-dict, "derag": run_case-dict}`.
  - module constants `CHANGE_COMMENTARY`, `STABLE_COMMENTARY`.
  - `main()` writes `eval/hallucination_qualitative.md`.

- [ ] **Step 1: Write the failing `render_md` test**

Create `tests/test_scripts_eval_hallucination.py`:

```python
from scripts.eval_hallucination import (
    CHANGE_COMMENTARY,
    STABLE_COMMENTARY,
    render_md,
)

_CASES = [
    {
        "label": "TMAO / red meat", "query": "Q1",
        "standard": {"mode": "standard",
                     "answer": "Red meat L-carnitine does not raise TMAO. [1]",
                     "sources": ["FABRICATED_MISINFO_tmao_redmeat_neutral.txt",
                                 "scifact_14803797_tmao_redmeat.txt", "scifact_x.txt"]},
        "derag": {"mode": "derag",
                  "answer": "Dietary L-carnitine from red meat raises TMAO and "
                            "accelerates atherosclerosis. [1]",
                  "sources": ["scifact_14803797_tmao_redmeat.txt",
                              "FABRICATED_MISINFO_tmao_redmeat_neutral.txt", "scifact_x.txt"]},
    },
    {
        "label": "H. pylori / gastric cancer", "query": "Q2",
        "standard": {"mode": "standard", "answer": "H. pylori is protective. [1]",
                     "sources": ["FABRICATED_MISINFO_hpylori_protective.txt", "a.txt", "b.txt"]},
        "derag": {"mode": "derag", "answer": "H. pylori infection increases gastric cancer risk. [1]",
                  "sources": ["scifact_2052720_hpylori_gastric_cancer.txt", "a.txt", "b.txt"]},
    },
    {
        "label": "antibiotics / C. difficile", "query": "Q3",
        "standard": {"mode": "standard",
                     "answer": "Antibiotics increase susceptibility to C. difficile. [1]",
                     "sources": ["scifact_6372244_antibiotics_cdiff.txt", "a.txt", "b.txt"]},
        "derag": {"mode": "derag",
                  "answer": "Antibiotics increase susceptibility to C. difficile. [1]",
                  "sources": ["scifact_6372244_antibiotics_cdiff.txt", "a.txt", "b.txt"]},
    },
]


def test_render_md_side_by_side_and_commentary():
    md = render_md(_CASES)
    assert "## TMAO / red meat" in md
    assert "## H. pylori / gastric cancer" in md
    assert "## antibiotics / C. difficile" in md
    assert "Standard RAG" in md and "DeRAG" in md
    # feeding sources are shown
    assert "FABRICATED_MISINFO_tmao_redmeat_neutral.txt" in md
    assert "scifact_2052720_hpylori_gastric_cancer.txt" in md
    # the two positives are labelled as the answer CHANGING
    assert md.count(CHANGE_COMMENTARY) == 2
    # the honest negative is labelled as the answer STAYING correct under both modes
    assert md.count(STABLE_COMMENTARY) == 1


def test_commentary_constants_say_the_right_things():
    assert "changes" in CHANGE_COMMENTARY
    assert "both modes" in STABLE_COMMENTARY and "does not distort" in STABLE_COMMENTARY
```

- [ ] **Step 2: Run, expect failure**

Run: `uv run pytest tests/test_scripts_eval_hallucination.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.eval_hallucination'`.

- [ ] **Step 3: Create `scripts/eval_hallucination.py`**

```python
"""Milestone 10 qualitative hallucination comparison.

For each of the three Milestone 9 target queries, generate a grounded
answer under Standard RAG and under trust-weighted rerank (product-default
top-3 context each) and render eval/hallucination_qualitative.md with the
two answers side by side.

Needs a live Hardhat node + IPFS daemon + a seeded Chroma store AND a
running Ollama with gemma4:12b pulled (see README).

    set -a; . ./.env; set +a
    uv run python scripts/eval_hallucination.py   # writes eval/hallucination_qualitative.md
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.evaluate import EVAL_DIR, gather_cases, load_manifest_chunks

CHANGE_COMMENTARY = (
    "**The generated answer changes between modes.** Standard RAG answers from "
    "the fabricated, inverted claim because the poison won retrieval; DeRAG "
    "answers from the real abstract because the on-chain reputation signal "
    "flipped the top result. The hallucination is visible in the Standard RAG "
    "column and absent in the DeRAG column."
)

STABLE_COMMENTARY = (
    "**The generated answer stays correct under both modes.** Raw similarity "
    "already ranks the legitimate chunk first, so trust-weighting does not "
    "distort a case it does not need to touch — exactly the property you want "
    "confirmed before trusting the mechanism on cases you cannot hand-verify."
)

_COMMENTARY_BY_LABEL = {
    "TMAO / red meat": CHANGE_COMMENTARY,
    "H. pylori / gastric cancer": CHANGE_COMMENTARY,
    "antibiotics / C. difficile": STABLE_COMMENTARY,
}


def retrieve(query: str, mode: str) -> list[dict]:
    from retrieval.rerank import trust_weighted_rerank
    from retrieval.standard_rag import standard_rag_answer_context

    if mode == "standard":
        return standard_rag_answer_context(query, k=3)
    if mode == "derag":
        return trust_weighted_rerank(query, k_retrieve=10, k_return=3)
    raise ValueError(f"unknown mode {mode!r}")


def run_case(query: str, mode: str) -> dict:
    from generation.llm_client import build_prompt, generate_blocking

    docs = retrieve(query, mode)
    answer = generate_blocking(build_prompt(query, docs))
    return {
        "mode": mode,
        "answer": answer.strip(),
        "sources": [d["metadata"].get("source_file", d["chroma_id"]) for d in docs],
    }


def _sources_md(sources: list[str]) -> str:
    return "\n".join(f"  {i}. `{s}`" for i, s in enumerate(sources, start=1))


def render_md(cases: list[dict]) -> str:
    parts = [
        "# DeRAG Qualitative Hallucination Comparison (Milestone 10)",
        "",
        "Standard RAG vs DeRAG generated answers for all three Milestone 9 target "
        "queries, each grounded in that mode's product-default top-3 retrieval. "
        "Answers captured verbatim from `gemma4:12b` at `temperature=0.1`.",
        "",
        "For **TMAO** and **H. pylori** the answer changes with the retrieval "
        "winner; for **antibiotics** it stays correct under both modes — the "
        "trust-weighting does not distort a case it does not need to touch.",
        "",
    ]
    for case in cases:
        parts += [
            f"## {case['label']}",
            "",
            f"**Query:** {case['query']}",
            "",
            "### Standard RAG",
            "",
            f"_Context sources:_\n{_sources_md(case['standard']['sources'])}",
            "",
            "> " + case["standard"]["answer"].replace("\n", "\n> "),
            "",
            "### DeRAG (trust-weighted rerank)",
            "",
            f"_Context sources:_\n{_sources_md(case['derag']['sources'])}",
            "",
            "> " + case["derag"]["answer"].replace("\n", "\n> "),
            "",
            _COMMENTARY_BY_LABEL[case["label"]],
            "",
        ]
    return "\n".join(parts).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="DeRAG qualitative hallucination comparison.")
    parser.add_argument("--out", default=str(EVAL_DIR / "hallucination_qualitative.md"))
    args = parser.parse_args()

    cases = []
    for case in gather_cases(load_manifest_chunks()):
        cases.append(
            {
                "label": case["label"],
                "query": case["query"],
                "standard": run_case(case["query"], "standard"),
                "derag": run_case(case["query"], "derag"),
            }
        )
    md = render_md(cases)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    print(md)
    print(f"\n[eval_hallucination] wrote {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run, expect pass**

Run: `uv run pytest tests/test_scripts_eval_hallucination.py -v`
Expected: all PASS.

- [ ] **Step 5: Full fast suite green**

Run: `uv run pytest`
Expected: PASS, no regressions. Note the pre-existing fast-suite count for the Task 5 comparison.

- [ ] **Step 6: Commit**

```bash
git add scripts/eval_hallucination.py tests/test_scripts_eval_hallucination.py
git commit -m "feat(m10): eval_hallucination.py Standard-RAG-vs-DeRAG answer comparison"
```

---

## Task 5: Live window A — run `evaluate.py`, commit `eval/results.md`

**Files:**
- Create: `eval/results.md` (captured output)
- Delete: `eval/.gitkeep` (only if present and now redundant)

**Interfaces:**
- Consumes: `scripts/evaluate.py` `main()` from Task 3; the live pipeline.
- Produces: committed `eval/results.md` with real numbers; the measured MRR/P@1 values recorded for Task 9's DECISIONS entry.

- [ ] **Step 1: Bring up the services**

In the `DeRAG/` directory, background each long-running service (leave them up for Task 6 too):

```bash
( cd contracts && npx hardhat node ) &            # port 8545
sleep 4
( cd contracts && npx hardhat run scripts/deploy.ts --network localhost )
ipfs daemon &                                     # ports 5001 / 8080
sleep 3
ollama serve &                                    # port 11434 (needed in Task 6)
sleep 2
```

Expected: `deploy.ts` prints deployed addresses; `contracts/deployments/localhost.json` is refreshed.

- [ ] **Step 2: Export env, wipe Chroma, seed, pre-flight**

```bash
set -a; . ./.env; set +a
rm -rf "${CHROMA_DB_DIR:-./vector_store}"
uv run python scripts/seed_reputation.py
uv run python scripts/check_candidate_window.py
```

Expected: seeding ends `[seed] done` (47 filler `[seed] skip`, 6 non-zero written); pre-flight ends `[preflight] PASS`.

- [ ] **Step 3: Run the evaluation**

```bash
uv run python scripts/evaluate.py
```

Expected: prints the rendered markdown and `[evaluate] wrote .../eval/results.md`. Sanity-check the output before committing:
- TMAO row: `SR P@1 = 0.000`, `DeRAG P@1 = 1.000`.
- H. pylori row: `SR P@1 = 0.000`, `DeRAG P@1 = 1.000`.
- antibiotics row: `SR P@1 = 1.000`, `DeRAG P@1 = 1.000`.
- alpha sweep `alpha = 1.000` MRR equals Standard-RAG MRR from the main table (the `ALPHA_SANITY_NOTE` claim).

If any of the three rows disagree with the above, **stop** — that contradicts the DEMO.md / DECISIONS.md thesis and the regression test; investigate (is the corpus seeded? is `check_candidate_window` PASS? did the Hardhat chain restart mid-run?) before proceeding.

- [ ] **Step 4: Run the live regression test as a cross-check**

```bash
uv run pytest -m slow tests/test_demo_corpus_regression.py -v
```

Expected: `7 passed, 1 xfailed` (unchanged from Milestone 9). This confirms the live environment matches what `evaluate.py` measured.

- [ ] **Step 5: Commit the captured results**

```bash
git rm -q eval/.gitkeep 2>/dev/null || true
git add eval/results.md
git commit -m "docs(m10): capture retrieval evaluation results (MRR/P@K + alpha sweep)"
```

---

## Task 6: Live window B — `eval_hallucination.py`, antibiotics recompute, DEMO.md re-verification

**Files:**
- Create: `eval/hallucination_qualitative.md` (captured output)
- Create: `docs/superpowers/plans/m10-live-findings.md` (scratch notes consumed by Tasks 7-9, deleted in Task 9)

**Interfaces:**
- Consumes: `scripts/eval_hallucination.py` from Task 4; the still-running services from Task 5; `pipeline.embeddings`.
- Produces: committed `eval/hallucination_qualitative.md`; a findings note with (a) the antibiotics `legit_sim` / `poison_sim` recomputed live, (b) which of DEMO.md / DECISIONS.md matches, (c) any DEMO.md scripted-output mismatches.

- [ ] **Step 1: Confirm Ollama has the model**

```bash
ollama list | grep -q 'gemma4:12b' && echo OK || ollama pull gemma4:12b
```

Expected: `OK` (already pulled in this environment).

- [ ] **Step 2: Generate the qualitative comparison**

```bash
set -a; . ./.env; set +a
uv run python scripts/eval_hallucination.py
```

Expected: prints the rendered markdown and `[eval_hallucination] wrote .../eval/hallucination_qualitative.md`. Verify by eye:
- TMAO Standard RAG answer reflects the **inverted** claim (no TMAO rise / not harmful); DeRAG answer reflects the **real** claim (L-carnitine → TMAO → atherosclerosis).
- H. pylori Standard RAG answer reflects the **inverted** claim (protective / no increased risk); DeRAG answer reflects the **real** claim (increases gastric cancer risk).
- antibiotics: **both** answers say antibiotics increase C. difficile susceptibility.

If Standard RAG's TMAO/H. pylori answers do *not* show the hallucination, check that Chroma still holds the seeded corpus and the Hardhat node has not restarted (in-memory chain state). Re-seed if needed.

- [ ] **Step 3: Commit the captured qualitative comparison**

```bash
git add eval/hallucination_qualitative.md
git commit -m "docs(m10): capture Standard-RAG-vs-DeRAG qualitative hallucination comparison"
```

- [ ] **Step 4: Recompute the antibiotics similarity pair live**

```bash
uv run python - <<'PY'
from pathlib import Path
from pipeline.embeddings import embed, cosine_similarity

d = Path("data/demo_corpus")
q = "Do antibiotics increase susceptibility to Clostridium difficile infection?"
legit = (d / "scifact_6372244_antibiotics_cdiff.txt").read_text(encoding="utf-8")
poison = (d / "FABRICATED_MISINFO_antibiotics_cdiff_protective.txt").read_text(encoding="utf-8")
qv, lv, pv = embed([q, legit, poison])
ls, ps = cosine_similarity(qv, lv), cosine_similarity(qv, pv)
print(f"legit_sim  = {ls:.4f}")
print(f"poison_sim = {ps:.4f}")
print(f"margin (legit - poison) = {ls - ps:.4f}")
PY
```

Record all three numbers.

- [ ] **Step 5: Resolve the DEMO.md vs DECISIONS.md discrepancy**

Compare the recomputed numbers against both documents:
- `DEMO.md` (line ~168): "0.7329 vs 0.7092".
- `DECISIONS.md` (line ~711): "poison 0.709 vs legit 0.733 (**legit wins** by 0.024)".

Decide and write into `docs/superpowers/plans/m10-live-findings.md`:
- the recomputed `legit_sim` / `poison_sim` / margin;
- **"harmless rounding"** if both documents round to the recomputed values (DEMO.md 4dp, DECISIONS.md 3dp, same underlying measurement) — then the only fix is to make DEMO.md and DECISIONS.md quote the *same* precision;
- **"stale remnant"** if DEMO.md's numbers do **not** round to the current measurement (they came from a pre-truncation or otherwise superseded run) — then note which run they were from, if determinable, and that DEMO.md was never re-measured when the corpus changed.
- Either way, record the sentence that will go into the Task 9 completion report and DECISIONS.md: *what the drift was and why it existed*, not just "fixed".

- [ ] **Step 6: Re-run every scripted query in DEMO.md**

For each fenced `uv run python -m scripts.demo_rerank "..."` block in `DEMO.md`, run it and diff the actual ranking lines against the captured "Expected output" block:

```bash
set -a; . ./.env; set +a
uv run python -m scripts.demo_rerank "Does dietary L-carnitine from red meat raise TMAO and accelerate atherosclerosis?"
uv run python -m scripts.demo_rerank "Does H. pylori infection increase the risk of gastric cancer?"
uv run python -m scripts.demo_rerank "Does H. pylori infection increase the risk of gastric cancer?"   # antibiotics contrast block reuses this? re-check DEMO.md
```

Also confirm the automated-proof block: `uv run pytest -m slow tests/test_demo_corpus_regression.py -v` → `7 passed, 1 xfailed`.

Record in `m10-live-findings.md`, per DEMO.md block: **match** or the exact numeric delta. Do **not** edit DEMO.md here — that is Task 9, so all doc edits land together.

- [ ] **Step 7: Commit the findings note**

```bash
git add docs/superpowers/plans/m10-live-findings.md
git commit -m "docs(m10): record live-window findings (antibiotics sims, DEMO.md re-verification)"
```

- [ ] **Step 8: Tear down services** (optional — keep up if Task 9 will re-verify)

```bash
jobs -p | xargs -r kill
```

---

## Task 7: `README.md` — "Services to start before running anything"

**Files:**
- Modify: `README.md` (replace the placeholder block at lines ~15-19; add an M10 status entry)

**Interfaces:**
- Consumes: nothing from other tasks (content is from the spec + `DECISIONS.md` M7 block).
- Produces: a from-clean-clone-runnable README.

- [ ] **Step 1: Replace the placeholder section**

Replace:

```markdown
## Services to start before running anything

_(To be filled in during Milestone 10, once Hardhat, IPFS, Ollama, and
Streamlit are all wired together.)_
```

with:

```markdown
## Services to start before running anything

### Startup order

1. **Hardhat node** — `cd contracts && npx hardhat node` (long-running, port 8545).
   Prints 20 funded accounts; **Account #0** is the reputation oracle.
2. **Deploy contracts** — `cd contracts && npx hardhat run scripts/deploy.ts --network localhost`
   (one-shot; deploys and exits; writes `contracts/deployments/localhost.json`).
3. **IPFS daemon** — `ipfs daemon` (long-running; API on 5001, gateway on 8080).
4. **Ollama** — `ollama serve` (long-running, port 11434). Only needed for
   answer generation: the Streamlit UI, `scripts/demo_generate.py`, and
   `scripts/eval_hallucination.py`. The CLI rerank demo and the automated
   proofs do **not** need it.
5. **Seed the demo corpus** (see "Models / tuning to pre-fetch" for the
   `.env` export this depends on):
   ```bash
   set -a; . ./.env; set +a
   uv run python scripts/seed_reputation.py
   uv run python scripts/check_candidate_window.py    # expect: [preflight] PASS
   ```
6. **Streamlit UI** — `uv run streamlit run app/main.py` (port 8501).

Hardhat's chain state is in-memory only: if you restart the node, re-run
steps 2 and 5 before any retrieval.

### Ports

| Port  | Service          | Config var          | Notes |
|-------|------------------|---------------------|-------|
| 8545  | Hardhat JSON-RPC | `WEB3_PROVIDER_URI` | local chain |
| 5001  | IPFS (Kubo) API  | `IPFS_API_URL`      | chunk bodies |
| 8080  | IPFS gateway     | —                   | hash-verification fetches |
| 11434 | Ollama           | `OLLAMA_HOST`       | answer generation only |
| 8501  | Streamlit UI     | —                   | `app/main.py` |

### Models / tuning to pre-fetch before running anything

**1. Pull the generation model** (~7.6 GB):

```bash
ollama pull gemma4:12b
```

**2. Create `.env` and export it into every shell that runs a script.**
The `scripts/` read `os.environ` directly and do **not** load `.env`
themselves:

```bash
cp .env.example .env
# set PRIVATE_KEY to the Hardhat Account #0 key; with the default mnemonic:
#   0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80
set -a; . ./.env; set +a    # <-- required before seed_reputation.py, evaluate.py, etc.
```

**3. Apply the Ollama systemd tuning — REQUIRED for `gemma4:12b` to be
viable.** Without it the model falls back to CPU or partial GPU offload:
a **silent performance regression, not a crash**, and easy to miss during
setup. `sudo systemctl edit ollama.service`, which creates the drop-in
`/etc/systemd/system/ollama.service.d/override.conf`. Paste it **verbatim,
including the `[Service]` header** — omitting the header makes systemd
ignore every line ("Assignment outside of section" in `journalctl -u
ollama`) while the service still reports `active`:

```ini
[Service]
Environment="OLLAMA_NOPRUNE=1"
Environment="OLLAMA_KV_CACHE_TYPE=q8_0"
Environment="OLLAMA_FLASH_ATTENTION=1"
Environment="OLLAMA_KEEP_ALIVE=-1"
```

Then reload and restart, and verify it took:

```bash
sudo systemctl daemon-reload && sudo systemctl restart ollama
systemctl show ollama -p Environment    # must list all four OLLAMA_* vars
```
```

- [ ] **Step 2: Add the Milestone 10 status entry**

Append to the "## Status" section, after the Milestone 9 paragraph:

```markdown
Milestone 10: quantitative evaluation and final docs. `scripts/evaluate.py`
computes MRR@10 and Precision@K for Standard RAG vs trust-weighted rerank
across the 3 target queries plus an alpha sweep (`eval/results.md`);
`scripts/eval_hallucination.py` captures Standard-RAG-vs-DeRAG generated
answers (`eval/hallucination_qualitative.md`). The result is
two-positive/one-honest-negative: TMAO and H. pylori show trust-weighting
overriding a similarity-based fooling attempt, antibiotics is a
no-regression contrast where DeRAG correctly does not over-trigger.
`VIVA.md` has the decision-grounded viva Q&A. This is the final milestone.
```

- [ ] **Step 3: Verify no placeholder remains & links resolve**

Run: `grep -n "To be filled in\|Milestone 10, once" README.md`
Expected: no matches.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs(m10): fill in README services/ports/tuning; add M10 status"
```

---

## Task 8: `VIVA.md` — decision-grounded viva Q&A

**Files:**
- Create: `VIVA.md`

**Interfaces:**
- Consumes: claude-mem history + `DECISIONS.md` (all milestones).
- Produces: `VIVA.md`.

- [ ] **Step 1: Reconstruct the real decision history**

Use claude-mem before writing a single answer. Either:
- `/timeline-report` for the `college` project, **or**
- targeted `mem-search` on specific terms from each milestone's actual decisions — e.g. `milestone 6 alpha sigmoid normalization`, `milestone 7 gemma systemd think false`, `milestone 4 IPFS gateway verify_hash`, `milestone 3 oracle owner reputation`, `milestone 9 truncation eligible flip`, `milestone 9 hpylori promotion xfail masking` — **not** generic queries like "decisions".

Capture, per milestone, the decisions that were actually made and are on record. Every answer in `VIVA.md` must cite one.

- [ ] **Step 2: Write `VIVA.md`**

Structure (fill each Q with an answer grounded in a real decision — no invented rationale):

```markdown
# DeRAG — Viva Preparation

Anticipated questions and answers, grounded in decisions actually made
during the project (see `DECISIONS.md` and the commit history). Organised
by the three subject areas of the academic overview.

## Deep Learning

- **Why `all-MiniLM-L6-v2` for embeddings?** ...
- **Why `gemma4:12b` and not the spec's Phi-3-mini / Llama-3-8B?** ...
  (quality-vs-memory on the real target hardware: 7.6 GB RAM, RTX 4060
  8 GB; CPU inference measured unusable; see M7 DECISIONS.)
- **Why `think: false` and `/api/chat`?** ...
- **How is reputation turned into a score the ranker can use?** Sigmoid
  normalisation with fixed `SCALE`, not per-batch min-max — because the
  Milestone 10 MRR/P@K evaluation needs a document's trust contribution
  to be identical across queries (M6 DECISIONS).
- **How was `alpha = 0.7` chosen / validated?** Pinned from M6; the M10
  alpha sweep across the 3 real queries confirmed it — <insert the sweep
  reading from eval/results.md>.
- **Chunking?** 500-char split (`pipeline/chunking.py`); the demo corpus
  is truncated to one chunk per file — see the ISR section.

## Information Storage & Retrieval

- **Why SciFact over Climate-FEVER?** Matches the assignment's "Class 1:
  peer-reviewed abstracts or verified claims" wording; downloaded from
  the real S3 release archive (M9 DECISIONS).
- **Vector store?** Local-persistent ChromaDB, cosine space.
- **What does the evaluation actually measure, and what's the result?**
  MRR@10 + P@1/P@3 for both retrieval modes over 3 queries, ground truth
  from `manifest.json`. Two positives (TMAO, H. pylori) + one honest
  negative (antibiotics, a no-regression check). <insert headline
  numbers from eval/results.md>.
- **IPFS / content addressing?** ... `verify_hash()` re-hashes fetched
  content with Milestone 1's `hash_chunk` (M4 DECISIONS).

## Blockchain

- **Why a blockchain here at all / what does it buy you?** ...
- **Contract design?** Owner/oracle-gated reputation ledger, publisher
  attribution, duplicate-hash protection (M3 DECISIONS).
- **Publisher attribution — does the ranker use it?** No —
  `trust_weighted_rerank` never reads `publisher`; publisher assignment
  is demo-narrative only (M9 DECISIONS).
- **Reputation seeding?** Idempotent `delta = target - current` (M9).
- **Chain persistence?** In-memory only; restart ⇒ re-deploy + re-seed.

## The hardest question

**"Antibiotics behaves differently from TMAO and H. pylori — how do you
know that's a real, honest result and not cherry-picked?"**

Answer as the actual sequence of events, not a flattened summary:

1. The single-chunk ingestion constraint
   (`seed_reputation.resolve_doc_hash` requires exactly one chunk;
   `pipeline/chunking.py` splits >500 chars) forced truncating the legit
   SciFact abstracts to leading-sentence excerpts.
2. That truncation **mechanically flipped 2 of the 3 designed similarity
   orderings** — leading OBJECTIVE/DESIGN sentences are less similar to a
   results-oriented question than the full abstract; the spec's §2
   numbers were whole-abstract.
3. `DEMO.md` was initially found to **still describe the original
   (pre-truncation) framing** and was corrected during the H. pylori
   promotion.
4. H. pylori's flip was **re-examined and found to be a real,
   data-supported second fooling case** that had been under-promoted (a
   measured-but-not-asserted oversight), and was added as a hard
   regression case across all three assertions.
5. Doing that rigorously surfaced a **test-isolation bug**: the
   antibiotics `xfail` sat on a monolithic test that also carried the
   trust-weighted assertion, so a genuine DeRAG regression on antibiotics
   could have been **silently masked** by the expected Standard-RAG
   failure. Fixed by splitting the trust-weighted assertion into its own
   **unconditional** test (`test_trust_weighted_top1_is_not_a_poison_chunk`).

This is a **strength**: the process caught its own mistakes and the
evaluation got stronger as a result — two independent hard cases plus an
honest negative, with tighter test isolation.
```

Replace every `...` and `<insert ...>` with real content from the
claude-mem reconstruction and the committed `eval/results.md`. No `...`
or `<insert>` may remain.

- [ ] **Step 3: Placeholder scan**

Run: `grep -nE '\.\.\.|<insert|TODO|TBD' VIVA.md`
Expected: no matches.

- [ ] **Step 4: Commit**

```bash
git add VIVA.md
git commit -m "docs(m10): decision-grounded viva Q&A (DL / ISR / Blockchain)"
```

---

## Task 9: `DEMO.md` re-verification fixes + `DECISIONS.md` closing entry

**Files:**
- Modify: `DEMO.md` (only where Task 6 found a mismatch)
- Modify: `DECISIONS.md` (new closing `## Milestone 10` section)
- Delete: `docs/superpowers/plans/m10-live-findings.md`

**Interfaces:**
- Consumes: `docs/superpowers/plans/m10-live-findings.md` from Task 6; `eval/results.md` from Task 5.
- Produces: final consistent docs.

- [ ] **Step 1: Apply DEMO.md fixes**

For each mismatch recorded in `m10-live-findings.md`, update the relevant
"Expected output" block / narration in `DEMO.md` to the live-measured
value. If Task 6 found **no** ranking mismatches, the only DEMO.md edit is
the antibiotics similarity pair:

- Make the antibiotics numbers in `DEMO.md` (currently "0.7329 vs 0.7092")
  and `DECISIONS.md` quote the **same measurement at the same precision**,
  using the Task 6 recomputed values.
- If Task 6 classified it as a **stale remnant**, also add a one-sentence
  parenthetical in `DEMO.md` at that line noting the number is the
  re-measured value against the current truncated corpus.

- [ ] **Step 2: Verify DEMO.md internal consistency**

Run: `grep -nE '0\.7[0-9]{2,3}' DEMO.md` and confirm every antibiotics-context
similarity figure matches the Task 6 recompute. Confirm the pass/xfail
counts in DEMO.md still say `7 passed, 1 xfailed`.

- [ ] **Step 3: Write the `DECISIONS.md` closing entry**

Append a new section at the end of `DECISIONS.md`:

```markdown
## Milestone 10 — Quantitative Evaluation, Final Docs & Viva Prep (2026-08-27)

- **`source_doc_id` added to `manifest.json`** (one per chunk;
  `str(doc_id)` for `scifact_*`, `null` for `FABRICATED_MISINFO_*`).
  Rationale: the evaluation harness resolves a retrieved chunk to its
  SciFact document without string-parsing `chunk_id`. `null` for poison
  keeps the field meaning "where did this text come from", distinct from
  `ground_truth_chunk_id`'s "what real doc does this attack". Regeneration
  stays byte-identical modulo the new field; `build_manifest()` emits it.
- **Evaluation methodology.** `scripts/evaluate.py`: MRR@10 + P@1/P@3 for
  `standard_rag_answer_context` and `trust_weighted_rerank` across the 3
  target queries, ground truth from `manifest.json` (one relevant SciFact
  doc per query). Evaluation-only `k_return=10` override; the product
  default (3) is untouched. `scripts/eval_hallucination.py` captures the
  generated answers under both modes.
- **Headline result (`eval/results.md`): two positives, one honest
  negative.**
  - TMAO: Standard RAG P@1 = <X>, DeRAG P@1 = <Y> (wide ~0.198 sim margin).
  - H. pylori: Standard RAG P@1 = <X>, DeRAG P@1 = <Y> (thin ~0.046 margin —
    reported honestly as the mechanism's robustness envelope).
  - antibiotics: Standard RAG P@1 = <X>, DeRAG P@1 = <Y> — no-regression
    check; raw similarity already correct, DeRAG does not over-trigger.
  - MRR@10 — Standard RAG <..> vs DeRAG <..>.
- **Alpha sweep (`0.0 / 0.3 / 0.5 / 0.7 / 1.0` across all 3 queries): DONE.**
  <State what it showed: whether MRR is flat across 0.3-0.7 (wide plateau,
  0.7 is a safe pick) or alpha-sensitive; confirm alpha = 1.0 reproduced
  Standard RAG's MRR exactly.>
- **Antibiotics similarity-pair discrepancy (DEMO.md vs DECISIONS.md):
  resolved.** <State the recomputed legit_sim / poison_sim / margin, and
  whether the prior mismatch was harmless rounding (4dp vs 3dp of the same
  measurement) or a stale remnant of a superseded run — and why it
  existed. Both documents now quote the same measurement.>
- **This is the last milestone in the DeRAG roadmap.** No further
  milestones are planned; the project is feature-complete and evaluated.
```

Replace every `<...>` with the real values from `eval/results.md` and the
Task 6 findings note. No angle-bracket placeholder may remain.

- [ ] **Step 4: Delete the scratch findings note**

```bash
git rm docs/superpowers/plans/m10-live-findings.md
```

- [ ] **Step 5: Placeholder scan across all M10 docs**

Run: `grep -nE '<[A-Za-z]|<\.\.\.|TODO|TBD|To be filled' README.md VIVA.md DECISIONS.md DEMO.md eval/results.md eval/hallucination_qualitative.md`
Expected: no matches.

- [ ] **Step 6: Full fast suite green**

Run: `uv run pytest`
Expected: PASS, count = pre-existing + the new `test_scripts_evaluate.py` / `test_scripts_eval_hallucination.py` tests, no regressions.

- [ ] **Step 7: Commit**

```bash
git add DEMO.md DECISIONS.md
git rm --cached docs/superpowers/plans/m10-live-findings.md 2>/dev/null || true
git commit -m "docs(m10): reconcile DEMO.md to live numbers; closing DECISIONS entry"
```

---

## Task 10: Final verification & branch finish

**Files:** none (verification + integration).

- [ ] **Step 1: Whole-suite green, fast + slow**

With services still up (or brought back up per Task 5 Step 1-2):

```bash
uv run pytest
uv run pytest -m slow tests/test_demo_corpus_regression.py tests/test_demo_corpus_fixtures.py -v
```

Expected: fast suite all green; slow regression `7 passed, 1 xfailed`.

- [ ] **Step 2: Definition-of-done checklist (from the spec)**

Confirm each, by opening the file:
- `eval/results.md` — two-positive/one-honest-negative framing prose present (not just tables); alpha sweep table + chart + sanity note present.
- `eval/hallucination_qualitative.md` — 3 queries × 2 modes; TMAO & H. pylori answers differ between modes; antibiotics identical-and-correct.
- `README.md` — startup order, port table, `ollama pull gemma4:12b`, verbatim `[Service]` systemd block, `set -a; . ./.env; set +a` requirement all present; no placeholder.
- `manifest.json` — `source_doc_id` on every entry; `git log -p` shows the diff added only that field.
- `DEMO.md` — antibiotics similarity figures match the live recompute; discrepancy explained, not just patched.
- `VIVA.md` — DL/ISR/Blockchain sections; hardest-question answer is the 5-step sequence; no `...`/`<insert>`.
- `DECISIONS.md` — Milestone 10 section states the alpha-sweep outcome, the discrepancy resolution, and "last milestone in the roadmap".

- [ ] **Step 3: Invoke `superpowers:requesting-code-review`**

Request a whole-branch review of `main..milestone-10-evaluation`. Address findings per `superpowers:receiving-code-review` before merge.

- [ ] **Step 4: Finish the branch**

Invoke `superpowers:finishing-a-development-branch`. Expected: fast-forward / merge `milestone-10-evaluation` into `main`; delete the branch.

- [ ] **Step 5: Report Milestone 10 complete**

In the completion report, state explicitly (per the user's standing request):
- what the antibiotics DEMO.md-vs-DECISIONS.md discrepancy **actually was** — harmless rounding or stale remnant — and **why it existed**;
- whether the alpha sweep was done and **what it showed**;
- that this was the **last milestone in the roadmap**.

---

## Self-Review

**Spec coverage:**
- §1 `source_doc_id` → Task 1. ✓
- §2 `evaluate.py` metrics + ground truth + alpha sweep + `results.md` → Tasks 2, 3; live run Task 5. ✓
- §3 `eval_hallucination.py` + `hallucination_qualitative.md` → Task 4; live run Task 6. ✓
- §4 README services/ports/tuning → Task 7. ✓
- §5 `VIVA.md` incl. hardest-question sequence → Task 8. ✓
- §6 DEMO.md re-verification + antibiotics discrepancy **as its own investigation** → Task 6 (measure + classify), Task 9 (apply + record). ✓
- §7 DECISIONS.md closing entry incl. alpha-sweep outcome + "last milestone" → Task 9. ✓
- Execution order (offline logic/tests first, one live window, then docs, then review) → Tasks 1-4 offline, 5-6 live, 7-9 docs, 10 review. ✓
- Framing language asserted by tests, not just structural presence → Task 3 Step 1 (`test_render_results_md_carries_the_framing_prose`, `..._framing_constants_say_the_right_things`, `..._plateau_reading_reacts_to_data`), Task 4 Step 1 (`test_render_md_side_by_side_and_commentary` counting `CHANGE_COMMENTARY` ×2 / `STABLE_COMMENTARY` ×1). ✓

**Placeholder scan:** `VIVA.md` and the `DECISIONS.md` entry are authored templates with `...` / `<...>` markers that each have an explicit "replace every marker, none may remain" step plus a `grep` gate (Task 8 Step 3, Task 9 Step 5). No plan step defers real content with "add appropriate X".

**Type consistency:** `hits_to_doc_ids(hits, sdi_by_file)`, `reciprocal_rank(ranked, gt_id)`, `precision_at_k(ranked, gt_id, k)`, `evaluate_query(ranked, gt_id) -> {"rr","p_at_1","p_at_3"}`, `run_evaluation() -> {"per_query","sweep","alphas"}`, `render_results_md(results)`, `gather_cases -> [{"chunk_id","label","query","gt_doc_id","poison_file","legit_file"}]` — used consistently across Tasks 2/3/4/5/6. `run_case(query, mode) -> {"mode","answer","sources"}` and `render_md(cases)` with `cases[i] = {"label","query","standard","derag"}` consistent between Task 4's impl and test. `EVAL_DIR` / `CORPUS_DIR` defined once in `evaluate.py`, imported by `eval_hallucination.py`.
