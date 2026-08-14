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
