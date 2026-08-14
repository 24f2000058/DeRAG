# Milestone 0: DeRAG Project Scaffolding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the DeRAG monorepo skeleton — folder structure, dependency manifests, config stubs, and smoke tests — with zero business logic, so every later milestone has a working foundation to build inside.

**Architecture:** A flat Python package tree at the repo root (`pipeline/`, `retrieval/`, `blockchain/`, `storage/`, `generation/`, `scripts/`, `app/`) managed by a single `uv`-driven `pyproject.toml`, plus a fully isolated `contracts/` npm package running Hardhat with TypeScript. Python tests live in one top-level `/tests/` directory; Solidity tests live inside `contracts/test/`.

**Tech Stack:** Python 3.11, `uv`, `pytest`, Streamlit, Node 24 / npm 11, TypeScript, Hardhat + `@nomicfoundation/hardhat-toolbox`, Solidity ^0.8.20.

**Spec:** `docs/superpowers/specs/2026-08-14-milestone-0-scaffolding-design.md`

## Global Constraints

- Python version pinned to **3.11** via `.python-version` (spec: broader wheel availability across the ML stack vs. system 3.12).
- Python dependency manager is **`uv`** — `pyproject.toml` + `uv.lock`, `uv run` for all Python execution.
- Hardhat network `localhost` at `http://127.0.0.1:8545`, `chainId: 31337` explicitly set in `hardhat.config.ts`.
- `contracts/` is a self-contained npm package (own `package.json`, `node_modules`, `tsconfig.json`) — Python tooling never touches it and vice versa.
- All Python tests live in the single top-level `/tests/` directory, named `tests/test_<module>_<thing>.py` — no per-module `tests/` subfolders.
- `.env.example` variable names are exact, copied verbatim from spec: `WEB3_PROVIDER_URI`, `CONTRACT_ADDRESS`, `PRIVATE_KEY`, `CHROMA_DB_DIR`, `OLLAMA_HOST`, `IPFS_API_URL`.
- No retrieval, blockchain, embedding, or generation logic — scaffolding only. No IPFS/Ollama/ChromaDB calls. No `DeRAG_Registry.sol`.
- Never commit `.env`, only `.env.example`. `PRIVATE_KEY` in any local `.env` must only ever be a Hardhat-printed deterministic test key.

---

## File Structure

```
DeRAG/
├── contracts/
│   ├── contracts/Placeholder.sol
│   ├── test/Placeholder.test.ts
│   ├── scripts/                  # empty, later milestones
│   ├── hardhat.config.ts
│   ├── package.json
│   ├── tsconfig.json
│   └── .gitignore                # node_modules, artifacts, cache
├── pipeline/__init__.py
├── retrieval/__init__.py
├── blockchain/__init__.py
├── storage/__init__.py
├── generation/__init__.py
├── scripts/__init__.py
├── app/main.py
├── tests/
│   ├── __init__.py
│   └── test_smoke.py
├── data/.gitkeep
├── plans/.gitkeep
├── eval/.gitkeep
├── pyproject.toml
├── .python-version
├── .env.example
├── .gitignore
├── README.md
└── DECISIONS.md
```

---

### Task 1: Python project skeleton, dependencies, and smoke test

**Files:**
- Create: `pyproject.toml`
- Create: `.python-version`
- Create: `pipeline/__init__.py`
- Create: `retrieval/__init__.py`
- Create: `blockchain/__init__.py`
- Create: `storage/__init__.py`
- Create: `generation/__init__.py`
- Create: `scripts/__init__.py`
- Create: `tests/__init__.py`
- Test: `tests/test_smoke.py`

**Interfaces:**
- Consumes: nothing (first task).
- Produces: importable top-level packages `pipeline`, `retrieval`, `blockchain`, `storage`, `generation`, `scripts` (all empty). A `uv`-managed venv at `.venv/` with all runtime + dev dependencies installed, invoked via `uv run <cmd>`. Later milestones add real code inside these packages without changing their import paths.

- [ ] **Step 1: Create `.python-version`**

```
3.11
```

- [ ] **Step 2: Create `pyproject.toml`**

```toml
[project]
name = "derag"
version = "0.1.0"
description = "DeRAG: Decentralized, Verifiable RAG with Trust-Weighted Ranking"
requires-python = ">=3.11,<3.12"
dependencies = [
    "sentence-transformers>=3.0",
    "torch>=2.0",
    "langchain-community>=0.3",
    "langchain-huggingface>=0.1",
    "chromadb>=0.5",
    "web3>=6,<7",
    "requests>=2.31",
    "streamlit>=1.35",
    "numpy>=1.26",
    "pandas>=2.2",
    "ollama>=0.3",
    "python-dotenv>=1.0",
]

[tool.uv]
package = false
dev-dependencies = [
    "pytest>=8.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

`package = false` tells `uv` this is an application, not an installable library — it manages the venv and dependencies without trying to build/install a `derag` package (there is no single top-level module matching that name; the project is several sibling packages).

- [ ] **Step 3: Run `uv sync` to create the venv and lockfile**

Run: `cd /home/danny/projects/college/DeRAG && uv sync`
Expected: creates `.venv/` and `uv.lock`, installs all dependencies successfully (this will take a while due to `torch`/`sentence-transformers`).

- [ ] **Step 4: Write the failing smoke test**

`tests/__init__.py` (empty file — its presence makes pytest walk up to the repo root and add it to `sys.path`, which is what makes `import pipeline` etc. resolve without installing the project as a package):

```python
```

`tests/test_smoke.py`:

```python
import importlib

MODULES = [
    "pipeline",
    "retrieval",
    "blockchain",
    "storage",
    "generation",
    "scripts",
]


def test_all_packages_importable():
    for module_name in MODULES:
        importlib.import_module(module_name)


def test_pytest_collects_and_runs():
    assert True
```

- [ ] **Step 5: Run pytest to verify the smoke test fails**

Run: `uv run pytest tests/test_smoke.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pipeline'` (the packages don't exist yet).

- [ ] **Step 6: Create the empty package directories**

`pipeline/__init__.py`, `retrieval/__init__.py`, `blockchain/__init__.py`, `storage/__init__.py`, `generation/__init__.py`, `scripts/__init__.py` — each file empty (0 bytes).

- [ ] **Step 7: Run pytest to verify the smoke test passes**

Run: `uv run pytest tests/test_smoke.py -v`
Expected: PASS — both `test_all_packages_importable` and `test_pytest_collects_and_runs` pass.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml .python-version uv.lock pipeline/__init__.py retrieval/__init__.py blockchain/__init__.py storage/__init__.py generation/__init__.py scripts/__init__.py tests/__init__.py tests/test_smoke.py
git commit -m "Scaffold Python project skeleton with uv and smoke test"
```

---

### Task 2: Env, gitignore, README, DECISIONS, and empty data dirs

**Files:**
- Create: `.env.example`
- Create: `.gitignore`
- Create: `README.md`
- Create: `DECISIONS.md`
- Create: `data/.gitkeep`
- Create: `plans/.gitkeep`
- Create: `eval/.gitkeep`

**Interfaces:**
- Consumes: nothing.
- Produces: `.env.example` names later milestones read from real `.env` (`WEB3_PROVIDER_URI`, `CONTRACT_ADDRESS`, `PRIVATE_KEY`, `CHROMA_DB_DIR`, `OLLAMA_HOST`, `IPFS_API_URL`). `data/`, `plans/`, `eval/` exist as tracked-but-empty directories for later milestones to populate.

- [ ] **Step 1: Create `.env.example`**

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

- [ ] **Step 2: Create `.gitignore`**

```gitignore
# Python
__pycache__/
*.pyc
.venv/
venv/
.env

# Node / Hardhat (contracts/ has its own nested .gitignore too)
node_modules/

# Vector store / ChromaDB
.chroma/
vector_store/

# OS
.DS_Store

# Local Claude tooling state (machine-local, not project artifacts)
.claude/settings.local.json
.claude-mem/
*.claude-mem.db
```

- [ ] **Step 3: Create `data/.gitkeep`, `plans/.gitkeep`, `eval/.gitkeep`**

Each file empty (0 bytes) — makes git track the otherwise-empty directories.

- [ ] **Step 4: Create `README.md`**

```markdown
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

Milestone 0: project scaffolding only. No retrieval, blockchain, embedding,
or generation logic implemented yet.
```

- [ ] **Step 5: Create `DECISIONS.md`**

```markdown
# DECISIONS

## Milestone 0 — Project Scaffolding (2026-08-14)

- **Python dependency manager: `uv`.** Already installed on the dev
  machine; single `pyproject.toml` + `uv.lock`; `uv run` manages the venv
  without manual activation.
- **Python version: 3.11**, pinned via `.python-version`. Chosen over the
  system's 3.12 for broader wheel availability across
  sentence-transformers/torch/chromadb/langchain.
- **Hardhat language: TypeScript.** Config, deploy scripts, and contract
  tests are all `.ts`.
- **Node package location: nested inside `/contracts`** as its own
  self-contained npm package, isolating the Node toolchain from the
  Python side of the repo entirely.
- **Folder layout: flat Python package tree at repo root**
  (`pipeline/`, `retrieval/`, `blockchain/`, `storage/`, `generation/`,
  `scripts/`, `app/`) under one `pyproject.toml`, since a single Python
  process spans all of them at runtime. `/contracts` is the one exception,
  kept as an isolated npm package.
- **Test placement: single top-level `/tests/` directory**, files named
  `tests/test_<module>_<thing>.py`, no per-module `tests/` subfolders.
  Chosen to keep pytest discovery to one root and avoid re-deciding
  placement ad hoc in every later milestone.
- **Test frameworks:** `pytest` for Python, Hardhat's native Mocha/Chai
  (via `hardhat-toolbox`) for Solidity.
```

- [ ] **Step 6: Verify gitignore correctness**

Run: `cd /home/danny/projects/college/DeRAG && git check-ignore -v .venv/ .env node_modules/ .chroma/ 2>&1 || true`
Expected: each path reported as ignored by `.gitignore`.

- [ ] **Step 7: Commit**

```bash
git add .env.example .gitignore README.md DECISIONS.md data/.gitkeep plans/.gitkeep eval/.gitkeep
git commit -m "Add env template, gitignore, README, and DECISIONS log"
```

---

### Task 3: Streamlit placeholder page

**Files:**
- Create: `app/main.py`

**Interfaces:**
- Consumes: nothing.
- Produces: a runnable `streamlit run app/main.py` entrypoint. Later milestones (retrieval UI, audit trail panel, RAG toggle) extend this file or import from `pipeline`/`retrieval`/`blockchain`/`storage`/`generation` into it.

- [ ] **Step 1: Create `app/main.py`**

```python
import streamlit as st

st.set_page_config(page_title="DeRAG", page_icon="🔗")

st.title("DeRAG")
st.caption("Decentralized, Verifiable RAG with Trust-Weighted Ranking")
st.write(
    "Project scaffolding is in place. Retrieval, blockchain, and "
    "generation features will appear here in later milestones."
)
```

- [ ] **Step 2: Verify the file is syntactically valid**

Run: `cd /home/danny/projects/college/DeRAG && uv run python -m py_compile app/main.py`
Expected: no output, exit code 0.

- [ ] **Step 3: Verify Streamlit actually renders it**

Run: `cd /home/danny/projects/college/DeRAG && uv run streamlit run app/main.py --server.headless true --server.port 8501 & sleep 5 && curl -s http://localhost:8501 | grep -o "<title>[^<]*" ; kill %1`
Expected: the `curl` output includes `<title>DeRAG` (Streamlit's default page title mirrors `page_title`), then the background server is killed.

- [ ] **Step 4: Commit**

```bash
git add app/main.py
git commit -m "Add placeholder Streamlit DeRAG page"
```

---

### Task 4: Hardhat contracts package (TypeScript, placeholder contract)

**Files:**
- Create: `contracts/package.json`
- Create: `contracts/tsconfig.json`
- Create: `contracts/hardhat.config.ts`
- Create: `contracts/.gitignore`
- Create: `contracts/contracts/Placeholder.sol`
- Test: `contracts/test/Placeholder.test.ts`

**Interfaces:**
- Consumes: nothing.
- Produces: a compiling, testable Hardhat project at `contracts/`. `Placeholder.sol` exposes `setValue(uint256)` / `getValue() returns (uint256)` — replaced entirely by `DeRAG_Registry.sol` in Milestone 3, so nothing downstream depends on these names surviving.

- [ ] **Step 1: Create `contracts/package.json`**

```json
{
  "name": "derag-contracts",
  "version": "0.1.0",
  "private": true,
  "scripts": {
    "compile": "hardhat compile",
    "test": "hardhat test"
  },
  "devDependencies": {
    "@nomicfoundation/hardhat-toolbox": "^5.0.0",
    "hardhat": "^2.22.0",
    "ts-node": "^10.9.0",
    "typescript": "^5.4.0"
  }
}
```

- [ ] **Step 2: Create `contracts/tsconfig.json`**

```json
{
  "compilerOptions": {
    "target": "es2020",
    "module": "commonjs",
    "esModuleInterop": true,
    "outDir": "dist",
    "declaration": true,
    "resolveJsonModule": true,
    "strict": true,
    "skipLibCheck": true
  },
  "include": ["./scripts", "./test"],
  "files": ["./hardhat.config.ts"]
}
```

- [ ] **Step 3: Create `contracts/hardhat.config.ts`**

```typescript
import { HardhatUserConfig } from "hardhat/config";
import "@nomicfoundation/hardhat-toolbox";

// Hardhat's built-in "hardhat" network (and its "localhost" mirror, once
// `npx hardhat node` is running) auto-generates 20 deterministic funded
// accounts by default. Per the DeRAG spec: Account #0 is reserved as
// Admin/Oracle, Accounts #1-#5 as publishers. No code assigns these roles
// yet — that lands in Milestone 3 alongside DeRAG_Registry.sol.
const config: HardhatUserConfig = {
  solidity: "0.8.20",
  networks: {
    localhost: {
      url: "http://127.0.0.1:8545",
      chainId: 31337,
    },
  },
};

export default config;
```

- [ ] **Step 4: Create `contracts/.gitignore`**

```gitignore
node_modules/
artifacts/
cache/
dist/
typechain-types/
```

- [ ] **Step 5: Create `contracts/contracts/Placeholder.sol`**

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

// Trivial stub so `npx hardhat compile` / `npx hardhat test` succeed in
// Milestone 0. Replaced by DeRAG_Registry.sol in Milestone 3.
contract Placeholder {
    uint256 private value;

    function setValue(uint256 _value) external {
        value = _value;
    }

    function getValue() external view returns (uint256) {
        return value;
    }
}
```

- [ ] **Step 6: Write the contract test**

`contracts/test/Placeholder.test.ts`:

```typescript
import { expect } from "chai";
import { ethers } from "hardhat";

describe("Placeholder", function () {
  it("stores and returns a value", async function () {
    const Placeholder = await ethers.getContractFactory("Placeholder");
    const placeholder = await Placeholder.deploy();
    await placeholder.setValue(42);
    expect(await placeholder.getValue()).to.equal(42);
  });
});
```

- [ ] **Step 7: Install dependencies**

Run: `cd /home/danny/projects/college/DeRAG/contracts && npm install`
Expected: `node_modules/` created, `package-lock.json` generated, no errors.

- [ ] **Step 8: Run `npx hardhat compile`**

Run: `cd /home/danny/projects/college/DeRAG/contracts && npx hardhat compile`
Expected: `Compiled 1 Solidity file successfully`.

- [ ] **Step 9: Run `npx hardhat test` and confirm it fails first if the contract were missing, then passes**

Since the contract already exists from Step 5, directly verify the pass case:

Run: `cd /home/danny/projects/college/DeRAG/contracts && npx hardhat test`
Expected: `Placeholder\n    ✔ stores and returns a value` — 1 passing test.

- [ ] **Step 10: Commit**

```bash
git add contracts/package.json contracts/package-lock.json contracts/tsconfig.json contracts/hardhat.config.ts contracts/.gitignore contracts/contracts/Placeholder.sol contracts/test/Placeholder.test.ts
git commit -m "Scaffold Hardhat TypeScript contracts package with placeholder contract"
```

---

### Task 5: End-to-end Definition of Done verification

**Files:** none created — verification only.

**Interfaces:**
- Consumes: everything from Tasks 1–4.
- Produces: confirmation that the Milestone 0 Definition of Done holds project-wide.

- [ ] **Step 1: Run the full Python test suite**

Run: `cd /home/danny/projects/college/DeRAG && uv run pytest -v`
Expected: all tests pass (both smoke tests from Task 1), exit code 0.

- [ ] **Step 2: Run the full Hardhat compile + test suite**

Run: `cd /home/danny/projects/college/DeRAG/contracts && npx hardhat compile && npx hardhat test`
Expected: compile succeeds, 1 passing test.

- [ ] **Step 3: Confirm Streamlit renders the placeholder page**

Run: `cd /home/danny/projects/college/DeRAG && uv run streamlit run app/main.py --server.headless true --server.port 8502 & sleep 5 && curl -s http://localhost:8502 | grep -o "<title>[^<]*"; kill %1`
Expected: `<title>DeRAG` present in the response.

- [ ] **Step 4: Confirm no `.env`, `node_modules/`, `.venv/`, or `.chroma/` are tracked by git**

Run: `cd /home/danny/projects/college/DeRAG && git status --short && git ls-files | grep -E "^\.env$|node_modules|\.venv|\.chroma" || echo "clean"`
Expected: working tree clean (all Task 1–4 commits already made), and the `grep` finds nothing (prints `clean`).

- [ ] **Step 5: No commit needed for this task** — it is verification-only. If any check fails, fix the underlying file from the relevant earlier task and re-commit there.

---

## Self-Review Notes

- **Spec coverage:** Tooling decisions (Task 1, 4), folder layout (all tasks), Python package contents + test placement convention (Task 1), dependencies (Task 1), Hardhat config incl. chainId/network/account comment (Task 4), `.env.example` exact names (Task 2), `.gitignore` incl. Claude tooling state (Task 2), security/git hygiene (documented in Global Constraints + `.env.example` never committed), README "Services to start" placeholder (Task 2), Definition of Done (Task 5), Out of Scope respected throughout (no real logic in any package).
- **Placeholder scan:** No TBD/TODO markers; the only "empty" content is intentional (empty `__init__.py` files, `.gitkeep` files) per spec.
- **Type consistency:** `Placeholder.sol`'s `setValue`/`getValue` names match between the contract (Task 4 Step 5) and its test (Task 4 Step 6). `MODULES` list in `tests/test_smoke.py` matches exactly the six packages created in Task 1 Step 6.
