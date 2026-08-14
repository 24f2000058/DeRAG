# Milestone 0: DeRAG Project Scaffolding — Design

## Purpose

Establish the monorepo skeleton for DeRAG (Decentralized, Verifiable RAG with
Trust-Weighted Ranking) with no business logic — folder structure, dependency
manifests, config stubs, and smoke tests only. Every later milestone
(ingestion pipeline, smart contract, retrieval reranking, generation, UI)
builds inside this skeleton.

## Tooling Decisions

- **Python dependency manager:** `uv`. Already installed on the target
  machine; single `pyproject.toml` + `uv.lock`; `uv run` manages the venv
  without manual activation steps.
- **Python version:** pinned to **3.11** via `.python-version` at the repo
  root. Chosen over the system's 3.12 for broader wheel availability across
  the sentence-transformers / torch / chromadb / langchain stack.
- **Node/Hardhat language:** **TypeScript**. Hardhat config, deploy scripts,
  and contract tests are all `.ts`.
- **Node package location:** nested inside `/contracts` as its own
  self-contained npm package (own `package.json`, `node_modules`,
  `tsconfig.json`). Keeps the Node toolchain fully isolated from the Python
  side of the repo.
- **Test frameworks:**
  - Python: `pytest`, run from the repo root, covering `pipeline/`,
    `retrieval/`, `blockchain/`, `storage/`, `generation/`, `app/`, and
    `scripts/` as they gain logic in later milestones.
  - Solidity: Hardhat's native Mocha/Chai via `hardhat-toolbox`, scoped
    inside `/contracts/test`.

## Folder Layout

```
DeRAG/
├── contracts/                  # Solidity + Hardhat (own npm package)
│   ├── contracts/
│   │   └── Placeholder.sol     # trivial stub, replaced in Milestone 3
│   ├── test/
│   │   └── Placeholder.test.ts
│   ├── scripts/                # deploy scripts (later milestones)
│   ├── hardhat.config.ts
│   ├── package.json
│   └── tsconfig.json
├── pipeline/                   # chunking, hashing, embeddings
├── retrieval/                  # ChromaDB + trust-weighted reranking
├── blockchain/                 # web3.py bridge to Hardhat
├── storage/                    # IPFS client
├── generation/                 # Ollama client
├── app/                        # Streamlit
│   └── main.py                 # placeholder "DeRAG" page
├── scripts/                    # ingest/eval/seed CLI entrypoints
├── tests/                      # pytest suite (root-level, cross-module)
│   └── test_smoke.py
├── data/                       # local datasets (gitignored contents)
├── plans/                      # planning docs
├── eval/                       # evaluation scripts/results (MRR, P@K later)
├── pyproject.toml
├── .python-version             # "3.11"
├── .env.example
├── .gitignore
├── README.md
└── DECISIONS.md
```

Rationale: `/contracts` is a self-contained npm package so Python tooling
never touches `node_modules` and vice versa. Everything else is a flat
Python package tree at the repo root, covered by one `pyproject.toml`,
matching how tightly pipeline/retrieval/blockchain/generation interact in
the actual RAG flow (a single Python process spans all of them).

## Python Package Contents (Milestone 0)

Each of `pipeline/`, `retrieval/`, `blockchain/`, `storage/`, `generation/`,
`scripts/` gets an `__init__.py` with no logic — just enough for the package
to be importable. `app/main.py` renders a placeholder Streamlit page titled
"DeRAG". `tests/test_smoke.py` asserts imports succeed and pytest collects
at least one test.

### Test file placement convention

All Python tests live in the single top-level `/tests/` directory — no
per-module `tests/` subfolders. Test files mirror the module they cover by
name: `tests/test_pipeline_<thing>.py`, `tests/test_retrieval_<thing>.py`,
`tests/test_blockchain_<thing>.py`, etc. This keeps pytest discovery to one
root and matches the flat-package rationale above (one Python process spans
all modules, so one test suite covers all modules). Every later milestone's
TDD work follows this convention rather than re-deciding placement per
milestone.

## Dependencies (`pyproject.toml`)

Runtime: `sentence-transformers`, `torch`, `langchain-community`,
`langchain-huggingface`, `chromadb`, `web3` (v6.x), `requests` (for IPFS
HTTP API — no dedicated `ipfshttpclient` needed since Kubo's HTTP API is
plain REST), `streamlit`, `numpy`, `pandas`, `ollama` (official Python
client), `python-dotenv` (reads `.env` — no calls yet in M0, but
`blockchain/` in Milestone 3 and `generation/` in Milestone 7 both need it
for `WEB3_PROVIDER_URI`/`PRIVATE_KEY`/`OLLAMA_HOST`; adding the dependency
now is infrastructure, not business logic).

Dev: `pytest`.

No version pins beyond major/minor floors needed for API compatibility
(e.g. `web3>=6,<7`) — this is an academic MVP, not a package meant for
external consumption.

## Hardhat Config (`contracts/hardhat.config.ts`)

- Network `localhost` at `http://127.0.0.1:8545`, `chainId: 31337`
  explicitly set (matches Hardhat's built-in local network defaults, stated
  explicitly per spec).
- Default Hardhat account generation (20 deterministic funded accounts) is
  left as the implicit default — no custom `accounts` config needed. A
  comment in `hardhat.config.ts` notes that Account #0 is reserved as
  Admin/Oracle and Accounts #1–#5 as publishers, for Milestone 3.
- `contracts/contracts/Placeholder.sol`: a trivial stub contract (e.g. a
  `uint256` getter/setter) — enough for `npx hardhat compile` and
  `npx hardhat test` to succeed. Replaced by `DeRAG_Registry.sol` in
  Milestone 3.

## Environment Variables (`.env.example`)

Exact names from the spec, unchanged:

```env
WEB3_PROVIDER_URI=http://127.0.0.1:8545
CONTRACT_ADDRESS=<Deployed_Hardhat_Address>
PRIVATE_KEY=<Hardhat_Account_0_Key>
CHROMA_DB_DIR=./vector_store
OLLAMA_HOST=http://localhost:11434
IPFS_API_URL=/ip4/127.0.0.1/tcp/5001

# Streamlit UI runs on port 8501 (no env var needed)
# IPFS gateway runs on port 8080 (no env var needed)
```

## .gitignore

Covers: `node_modules/`, `.chroma/`, `vector_store/`, `__pycache__/`,
`*.pyc`, `.env`, `.venv/`, `venv/`, plus standard Hardhat artifacts
(`contracts/artifacts/`, `contracts/cache/`) and `.DS_Store`.

## README.md

Project title, one-paragraph description (from the academic overview), and
a "Services to start before running anything" section left as an empty
placeholder heading — filled in Milestone 10 once all local services
(Hardhat node, IPFS daemon, Ollama, Streamlit) are wired together.

## Definition of Done

- `uv run pytest` (or `pytest` inside the uv-managed venv) runs green with
  at least the smoke test passing.
- `npx hardhat compile` succeeds inside `/contracts`.
- `npx hardhat test` succeeds inside `/contracts` (placeholder contract
  test passes).
- `streamlit run app/main.py` renders a placeholder "DeRAG" page.
- No retrieval, blockchain, embedding, or generation logic is implemented —
  scaffolding only.

## Out of Scope (Milestone 0)

Everything under `<blockchain_spec>`, `<isr_spec>`, `<deep_learning_spec>`,
and `<local_integration_flow>` beyond the placeholder contract and empty
package structure. No IPFS, Ollama, or ChromaDB calls. No `DeRAG_Registry.sol`.
No ingestion or retrieval code.
