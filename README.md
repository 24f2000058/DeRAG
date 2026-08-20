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

Milestone 5: end-to-end ingestion pipeline (`scripts/ingest.py`) implemented
and tested — extracts text from `.txt`/`.pdf` input (single file or
directory), chunks it via Milestone 1's pipeline, uploads each chunk to IPFS
for a real CID, verifies the round-trip hash before registering, and writes
to both the on-chain registry and the local Chroma store. Skips chunks
already registered on-chain (idempotent re-runs), and isolates failures
per-chunk and per-file so one bad document doesn't abort the whole run.
Needs a running Hardhat node and IPFS daemon (see Milestones 3-4), plus
`WEB3_PROVIDER_URI` and `PRIVATE_KEY` set (copy `.env.example` to `.env`).
Run it with:

```
ipfs daemon &
cd contracts && npx hardhat node &
cd contracts && npx hardhat run scripts/deploy.ts --network localhost
uv run python scripts/ingest.py --input path/to/file_or_directory
```

Milestone 6: trust-weighted re-ranking engine (`retrieval/rerank.py`)
implemented and tested — the core mechanism the whole project is built
around. Combines each retrieved chunk's cosine similarity with its on-chain
reputation (`blockchain.registry_client.verify_integrity()`) into a single
score, `S(q,d) = alpha*similarity + (1-alpha)*NormReputation`, so a
high-similarity but low- or negative-reputation ("poisoned") chunk can be
outranked by a lower-similarity, well-reputed one. Reputation is normalized
with a sigmoid (a fixed curve, reproducible across queries — not a per-batch
min-max scale) and unregistered chunks (`exists=False`) are floored below
every registered chunk in the batch, since an on-chain reputation of `0` is
a legitimate neutral starting value, not a penalty. `alpha` defaults to
`0.7` and is fully overridable, tuned empirically in later milestones. See
`DECISIONS.md` for the full design rationale. Once data has been ingested
(Milestone 5) against a live Hardhat node, compare standard vs.
trust-weighted rankings side by side with:

```
uv run python scripts/demo_rerank.py "your query text here"
```
