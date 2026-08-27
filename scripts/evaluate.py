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
