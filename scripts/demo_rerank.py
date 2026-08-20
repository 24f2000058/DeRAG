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
