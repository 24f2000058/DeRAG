# DeRAG

**Decentralized, Verifiable Retrieval-Augmented Generation with Trust-Weighted Ranking.**

DeRAG is an academic MVP combining local-only Deep Learning, Information
Storage & Retrieval, and Blockchain. It shields RAG systems from data
poisoning by cross-referencing retrieved documents against an immutable,
crowd-sourced reputation ledger (a local Hardhat blockchain) before
re-ranking them into the LLM's context — penalizing unverified or
low-reputation content instead of trusting semantic similarity alone.

Everything runs locally: no cloud APIs, no public blockchain deployment,
no LLM fine-tuning.

## Services to start before running anything

_(To be filled in during Milestone 10, once Hardhat, IPFS, Ollama, and
Streamlit are all wired together.)_

## Status

Milestone 0: project scaffolding. Milestone 1: preprocessing pipeline
(`pipeline/normalize.py`, `chunking.py`, `hashing.py`, `embeddings.py`,
`ingest_prep.py`) implemented and tested. Run the demo with:

```
uv run python scripts/demo_ingest_prep.py
```

Milestone 2: local-persistent ChromaDB vector store and "Standard RAG"
retrieval (`retrieval/vector_store.py`, `retrieval/standard_rag.py`)
implemented and tested — cosine-similarity search over the pipeline's
embeddings, top-3 retrieval with no re-ranking. Run the demo with:

```
uv run python scripts/demo_standard_rag.py
```

No blockchain or generation logic implemented yet; `ipfs_cid` is a
placeholder (`""`) until Milestones 4-5 wire in real IPFS values.
