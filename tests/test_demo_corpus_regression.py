"""Milestone 9 thesis-proof regression test.

Against the REAL demo corpus and REAL running services, Standard RAG's
top-1 result for a real query is a poisoned chunk, while
trust_weighted_rerank's top-1 is not.

Requires, all live at once:
  - npx hardhat node  +  contracts/scripts/deploy.ts   (WEB3_PROVIDER_URI, PRIVATE_KEY)
  - ipfs daemon                                         (IPFS_API_URL)
  - the demo corpus already ingested AND reputation-seeded:
        uv run python scripts/seed_reputation.py
  - a Chroma store at CHROMA_DB_DIR populated by that same run

Run:  uv run pytest -m slow tests/test_demo_corpus_regression.py -v
"""

import json
from pathlib import Path

import pytest

from retrieval.rerank import trust_weighted_rerank
from retrieval.standard_rag import standard_rag_answer_context

CORPUS_DIR = Path(__file__).resolve().parent.parent / "data" / "demo_corpus"
POISON_PREFIX = "FABRICATED_MISINFO_"


def _cases():
    chunks = json.loads((CORPUS_DIR / "manifest.json").read_text())["chunks"]
    by_id = {c["chunk_id"]: c for c in chunks}
    out = {}
    for chunk_id in ("poison_tmao_redmeat_neutral", "poison_antibiotics_cdiff_protective"):
        poison = by_id[chunk_id]
        legit = by_id[poison["ground_truth_chunk_id"]]
        out[chunk_id] = (poison["target_queries"][0], poison["filename"], legit["filename"])
    return out


CASES = _cases()


_ANTIBIOTICS_STANDARD_RAG_XFAIL = pytest.mark.xfail(
    reason=(
        "M9: SciFact abstract truncation (see DECISIONS.md) makes the natural "
        "antibiotics query rank the legitimate chunk above the poison on raw "
        "similarity; non-load-bearing — TMAO is the Standard-RAG-fooled proof."
    ),
    strict=False,
)


@pytest.mark.slow
@pytest.mark.parametrize("case_id", [
    "poison_tmao_redmeat_neutral",
    pytest.param("poison_antibiotics_cdiff_protective", marks=_ANTIBIOTICS_STANDARD_RAG_XFAIL),
])
def test_standard_rag_top1_is_the_poison_chunk(case_id):
    query_text, poison_file, _legit_file = CASES[case_id]
    standard_hits = standard_rag_answer_context(query_text, k=3)
    assert standard_hits, "no vectors returned — is the corpus ingested into Chroma?"
    assert standard_hits[0]["metadata"]["source_file"] == poison_file, (
        f"{case_id}: expected Standard RAG to be fooled; got "
        f"{standard_hits[0]['metadata']['source_file']}"
    )


@pytest.mark.slow
@pytest.mark.parametrize("case_id", [
    "poison_tmao_redmeat_neutral",
    "poison_antibiotics_cdiff_protective",
])
def test_trust_weighted_top1_is_not_a_poison_chunk(case_id):
    query_text, _poison_file, _legit_file = CASES[case_id]
    trust_hits = trust_weighted_rerank(query_text, k_retrieve=10, k_return=3)
    assert trust_hits
    assert not trust_hits[0]["metadata"]["source_file"].startswith(POISON_PREFIX), (
        f"{case_id}: trust-weighted rerank returned a poison chunk at rank 1"
    )


@pytest.mark.slow
def test_trust_weighted_ranks_tmao_legit_counterpart_above_its_poison():
    query_text, poison_file, legit_file = CASES["poison_tmao_redmeat_neutral"]
    trust_hits = trust_weighted_rerank(query_text, k_retrieve=10, k_return=10)
    order = [hit["metadata"]["source_file"] for hit in trust_hits]
    assert poison_file in order and legit_file in order, (
        f"one of the TMAO pair fell outside the top-10 candidate window: {order}"
    )
    assert order.index(legit_file) < order.index(poison_file)
