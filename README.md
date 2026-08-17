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

Milestone 3: on-chain document registry (`contracts/contracts/DeRAG_Registry.sol`)
and Python bridge (`blockchain/registry_client.py`) implemented and tested —
owner/oracle-gated reputation ledger with publisher attribution and
duplicate-hash protection. Start a local node and deploy with:

```
cd contracts && npx hardhat node
# in a second terminal:
cd contracts && npx hardhat run scripts/deploy.ts --network localhost
```

No generation logic implemented yet; `ipfs_cid` is still a placeholder
(`""`) until Milestones 4-5 wire in real IPFS values.

Milestone 4: IPFS storage client (`storage/ipfs_client.py`) implemented and
tested against a real local Kubo daemon — text upload/fetch via the daemon's
HTTP API and public gateway, plus `verify_hash()` (used by Milestone 8's
audit trail UI) which proves fetched content still hashes identically to
what was uploaded, using Milestone 1's exact `hash_chunk`. Start a local
daemon before running the slow suite or the demo:

```
ipfs daemon
```

Then run the tests or demo with:

```
IPFS_API_URL=/ip4/127.0.0.1/tcp/5001 uv run pytest -m slow tests/test_storage_ipfs_client.py
```
