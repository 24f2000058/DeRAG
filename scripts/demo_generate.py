"""Manual demo: stream a grounded answer from the local Ollama model given a
real top-3 retrieval result from either standard RAG or trust-weighted
re-ranking.

Requires a real `ollama serve` running with gemma4:12b pulled, and a
populated Chroma store — run `scripts/ingest.py` first if this is a fresh
environment, or if the Hardhat node has been restarted since data was last
registered (Hardhat's chain state is in-memory only and does not persist
across restarts, so previously-registered documents will fail
verify_integrity() after a restart until re-ingested). Chroma's persistent
directory is the one exception that does survive restarts.

Usage:
    uv run python scripts/demo_generate.py "<query text>"
    uv run python scripts/demo_generate.py "<query text>" --mode trust
"""

import sys

from generation.llm_client import build_prompt, generate_streaming
from retrieval.rerank import trust_weighted_rerank
from retrieval.standard_rag import standard_rag_answer_context


def main() -> None:
    args = sys.argv[1:]
    mode = "trust" if "--mode" in args and args[args.index("--mode") + 1] == "trust" else "standard"
    query_args = [a for a in args if a not in ("--mode", "standard", "trust")]
    query_text = query_args[0] if query_args else "renewable energy sources"

    if mode == "trust":
        context_docs = trust_weighted_rerank(query_text, k_retrieve=10, k_return=3)
    else:
        context_docs = standard_rag_answer_context(query_text, k=3)

    print(f"Mode: {mode}")
    print(f"Query: {query_text!r}")
    print(f"Retrieved {len(context_docs)} source(s).\n")

    prompt = build_prompt(query_text, context_docs)
    print("--- Streaming answer ---")
    for chunk in generate_streaming(prompt):
        print(chunk, end="", flush=True)
    print("\n------------------------")


if __name__ == "__main__":
    main()
