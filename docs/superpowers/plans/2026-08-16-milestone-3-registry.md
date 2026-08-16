# Milestone 3: DeRAG_Registry Smart Contract + Python Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `DeRAG_Registry.sol` (owner/oracle-gated on-chain document
ledger) and `blockchain/registry_client.py` (web3.py bridge), fully
isolated from IPFS/ChromaDB/Streamlit.

**Architecture:** A single Solidity contract holds a `bytes32 => Document`
mapping with owner-gated oracle authorization, publisher-attributed
registration, and oracle-gated reputation updates. A Hardhat deploy
script writes the deployed address + ABI to a JSON artifact that a thin
Python client reads to build a `web3.py` contract instance; each Python
write function accepts an optional signing key so callers can act as
different on-chain accounts.

**Tech Stack:** Solidity `0.8.20`, Hardhat `2.29.0` +
`@nomicfoundation/hardhat-toolbox` `5.0.0` (Mocha/Chai, ethers v6,
TypeChain), `web3.py` `6.20.3`, `pytest`.

**Spec:** `docs/superpowers/specs/2026-08-15-milestone-3-registry-design.md`

## Global Constraints

- Solidity version: `^0.8.20` (already set in `contracts/hardhat.config.ts`).
- `web3.py` signing pattern (verified against the real 6.20.3 install):
  `Account.from_key(key)` → `account.sign_transaction(tx_dict)` →
  `w3.eth.send_raw_transaction(signed.rawTransaction)` — the attribute is
  `rawTransaction` (camelCase), not `raw_transaction`.
- bytes32 conversion: `Web3.to_bytes(hexstr="0x" + hex_string_without_prefix)`
  (verified round-trips a 64-char SHA-256 hex string).
- `ipfs_cid` is stored as `""` (empty string) in Milestone 2's Chroma
  metadata, never `None` — not touched by this milestone's code, but
  relevant to keep in mind since `registry_client.py`'s `ipfs_cid`
  parameter will eventually receive that same field from Milestone 5.
- `documentLedger`'s Solidity auto-generated getter returns all six
  `Document` struct fields (verified by compiling a scratch contract) —
  no custom six-field getter needed in the contract; `get_document()` in
  the Python client calls this auto-getter directly.
- `contracts/deployments/` is a new directory, not yet in
  `contracts/.gitignore` — must be added (build output, like
  `artifacts/`/`cache/`, not a tracked deliverable).
- No pytest markers are registered yet in `pyproject.toml` — the `slow`
  marker used by the Python integration test must be registered there to
  avoid an unknown-marker warning.

---

### Task 1: DeRAG_Registry.sol contract + Hardhat tests

**Files:**
- Create: `contracts/contracts/DeRAG_Registry.sol`
- Create: `contracts/test/DeRAG_Registry.test.ts`
- Delete: `contracts/contracts/Placeholder.sol`
- Delete: `contracts/test/Placeholder.test.ts`

**Interfaces:**
- Produces: Solidity contract `DeRAG_Registry` with:
  - `struct Document { address publisher; bytes32 contentHash; string ipfsCID; int256 reputationScore; uint256 timestamp; bool isRegistered; }`
  - `address public owner`
  - `mapping(address => bool) public authorizedOracles`
  - `mapping(bytes32 => Document) public documentLedger` (auto-getter returns all 6 fields)
  - `constructor()` — sets `owner = msg.sender`, `authorizedOracles[msg.sender] = true`
  - `function authorizeOracle(address _account, bool _authorized) external` — reverts unless `msg.sender == owner`
  - `function registerDocument(bytes32 _hash, string memory _cid) external` — reverts if `documentLedger[_hash].isRegistered`
  - `function updateReputation(bytes32 _hash, int256 _delta) external` — reverts unless `authorizedOracles[msg.sender]`; reverts unless `documentLedger[_hash].isRegistered`
  - `function verifyIntegrity(bytes32 _hash) external view returns (bool exists, int256 reputation, address publisher)`
- Later tasks (deploy script, `registry_client.py`) consume this contract's ABI post-compile — no other task reads this contract's Solidity source directly.

- [ ] **Step 1: Write the failing Hardhat test file**

Create `contracts/test/DeRAG_Registry.test.ts`:

```typescript
import { expect } from "chai";
import { ethers } from "hardhat";

describe("DeRAG_Registry", function () {
  async function deployFixture() {
    const [owner, publisher1, publisher2, oracle2] = await ethers.getSigners();
    const Registry = await ethers.getContractFactory("DeRAG_Registry");
    const registry = await Registry.deploy();
    return { registry, owner, publisher1, publisher2, oracle2 };
  }

  const HASH_1 = ethers.keccak256(ethers.toUtf8Bytes("doc-1"));
  const HASH_UNREGISTERED = ethers.keccak256(ethers.toUtf8Bytes("never-registered"));

  it("sets publisher to msg.sender and reputation to 0 on successful registration", async function () {
    const { registry, publisher1 } = await deployFixture();
    await registry.connect(publisher1).registerDocument(HASH_1, "cid-1");
    const doc = await registry.documentLedger(HASH_1);
    expect(doc.publisher).to.equal(publisher1.address);
    expect(doc.reputationScore).to.equal(0);
    expect(doc.isRegistered).to.equal(true);
  });

  it("reverts when registering an already-registered hash", async function () {
    const { registry, publisher1, publisher2 } = await deployFixture();
    await registry.connect(publisher1).registerDocument(HASH_1, "cid-1");
    await expect(
      registry.connect(publisher2).registerDocument(HASH_1, "cid-2")
    ).to.be.reverted;
  });

  it("auto-authorizes the deployer as an oracle at construction", async function () {
    const { registry, owner } = await deployFixture();
    expect(await registry.authorizedOracles(owner.address)).to.equal(true);
  });

  it("reverts when a non-owner calls authorizeOracle", async function () {
    const { registry, publisher1, oracle2 } = await deployFixture();
    await expect(
      registry.connect(publisher1).authorizeOracle(oracle2.address, true)
    ).to.be.reverted;
  });

  it("allows the owner to authorize a new oracle", async function () {
    const { registry, owner, oracle2 } = await deployFixture();
    await registry.connect(owner).authorizeOracle(oracle2.address, true);
    expect(await registry.authorizedOracles(oracle2.address)).to.equal(true);
  });

  it("reverts when a non-authorized address calls updateReputation", async function () {
    const { registry, publisher1 } = await deployFixture();
    await registry.connect(publisher1).registerDocument(HASH_1, "cid-1");
    await expect(
      registry.connect(publisher1).updateReputation(HASH_1, -5)
    ).to.be.reverted;
  });

  it("allows an authorized oracle to adjust reputation, including into negative territory", async function () {
    const { registry, owner, publisher1 } = await deployFixture();
    await registry.connect(publisher1).registerDocument(HASH_1, "cid-1");
    await registry.connect(owner).updateReputation(HASH_1, -10);
    const doc = await registry.documentLedger(HASH_1);
    expect(doc.reputationScore).to.equal(-10);
    await registry.connect(owner).updateReputation(HASH_1, -5);
    const doc2 = await registry.documentLedger(HASH_1);
    expect(doc2.reputationScore).to.equal(-15);
  });

  it("reverts updateReputation for an unregistered hash even from an authorized oracle", async function () {
    const { registry, owner } = await deployFixture();
    await expect(
      registry.connect(owner).updateReputation(HASH_UNREGISTERED, 5)
    ).to.be.reverted;
  });

  it("verifyIntegrity returns false/0/zero-address for an unregistered hash", async function () {
    const { registry } = await deployFixture();
    const [exists, reputation, publisher] = await registry.verifyIntegrity(HASH_UNREGISTERED);
    expect(exists).to.equal(false);
    expect(reputation).to.equal(0);
    expect(publisher).to.equal(ethers.ZeroAddress);
  });

  it("verifyIntegrity returns correct values for a registered hash", async function () {
    const { registry, publisher1 } = await deployFixture();
    await registry.connect(publisher1).registerDocument(HASH_1, "cid-1");
    const [exists, reputation, publisher] = await registry.verifyIntegrity(HASH_1);
    expect(exists).to.equal(true);
    expect(reputation).to.equal(0);
    expect(publisher).to.equal(publisher1.address);
  });

  it("documentLedger getter returns all six fields for a registered document", async function () {
    const { registry, publisher1 } = await deployFixture();
    await registry.connect(publisher1).registerDocument(HASH_1, "cid-1");
    const doc = await registry.documentLedger(HASH_1);
    expect(doc.publisher).to.equal(publisher1.address);
    expect(doc.contentHash).to.equal(HASH_1);
    expect(doc.ipfsCID).to.equal("cid-1");
    expect(doc.reputationScore).to.equal(0);
    expect(doc.timestamp).to.be.greaterThan(0);
    expect(doc.isRegistered).to.equal(true);
  });
});
```

Also delete the placeholder files now (they're being replaced, and leaving
them in place would let the old `Placeholder` test pass while the new
suite fails, muddying the RED signal):

```bash
rm contracts/contracts/Placeholder.sol contracts/test/Placeholder.test.ts
```

- [ ] **Step 2: Run tests to verify they fail**

Run (from `contracts/`): `npx hardhat test`
Expected: compilation error / `HH700: Artifact for contract "DeRAG_Registry" not found` — the contract doesn't exist yet.

- [ ] **Step 3: Write the contract**

Create `contracts/contracts/DeRAG_Registry.sol`:

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract DeRAG_Registry {
    struct Document {
        address publisher;
        bytes32 contentHash;
        string ipfsCID;
        int256 reputationScore;
        uint256 timestamp;
        bool isRegistered;
    }

    address public owner;
    mapping(address => bool) public authorizedOracles;
    mapping(bytes32 => Document) public documentLedger;

    modifier onlyOwner() {
        require(msg.sender == owner, "DeRAG_Registry: caller is not the owner");
        _;
    }

    modifier onlyAuthorizedOracle() {
        require(authorizedOracles[msg.sender], "DeRAG_Registry: caller is not an authorized oracle");
        _;
    }

    constructor() {
        owner = msg.sender;
        authorizedOracles[msg.sender] = true;
    }

    function authorizeOracle(address _account, bool _authorized) external onlyOwner {
        authorizedOracles[_account] = _authorized;
    }

    function registerDocument(bytes32 _hash, string memory _cid) external {
        require(!documentLedger[_hash].isRegistered, "DeRAG_Registry: document already registered");
        documentLedger[_hash] = Document({
            publisher: msg.sender,
            contentHash: _hash,
            ipfsCID: _cid,
            reputationScore: 0,
            timestamp: block.timestamp,
            isRegistered: true
        });
    }

    function updateReputation(bytes32 _hash, int256 _delta) external onlyAuthorizedOracle {
        require(documentLedger[_hash].isRegistered, "DeRAG_Registry: document not registered");
        documentLedger[_hash].reputationScore += _delta;
    }

    function verifyIntegrity(bytes32 _hash)
        external
        view
        returns (bool exists, int256 reputation, address publisher)
    {
        Document storage doc = documentLedger[_hash];
        return (doc.isRegistered, doc.reputationScore, doc.publisher);
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run (from `contracts/`): `npx hardhat test`
Expected: all 11 tests pass, 0 failures.

- [ ] **Step 5: Commit**

```bash
git add contracts/contracts/DeRAG_Registry.sol contracts/test/DeRAG_Registry.test.ts
git rm contracts/contracts/Placeholder.sol contracts/test/Placeholder.test.ts
git commit -m "Add DeRAG_Registry.sol with owner/oracle model, replacing Placeholder"
```

---

### Task 2: Deploy script writing the deployment artifact

**Files:**
- Create: `contracts/scripts/deploy.ts`
- Modify: `contracts/.gitignore` (add `deployments/`)

**Interfaces:**
- Consumes: `DeRAG_Registry` contract factory (from Task 1, via
  `ethers.getContractFactory("DeRAG_Registry")`) and its compiled artifact
  (via `artifacts.readArtifact("DeRAG_Registry")`).
- Produces: `contracts/deployments/localhost.json` with shape
  `{ "address": string, "abi": array }` — Task 3's `registry_client.py`
  reads this file directly by this exact path and these exact two keys.

- [ ] **Step 1: Write the deploy script**

Create `contracts/scripts/deploy.ts`:

```typescript
import { ethers, artifacts } from "hardhat";
import * as fs from "fs";
import * as path from "path";

async function main() {
  const Registry = await ethers.getContractFactory("DeRAG_Registry");
  const registry = await Registry.deploy();
  await registry.waitForDeployment();
  const address = await registry.getAddress();

  const artifact = await artifacts.readArtifact("DeRAG_Registry");

  const deploymentsDir = path.join(__dirname, "..", "deployments");
  fs.mkdirSync(deploymentsDir, { recursive: true });
  fs.writeFileSync(
    path.join(deploymentsDir, "localhost.json"),
    JSON.stringify({ address, abi: artifact.abi }, null, 2)
  );

  console.log(`DeRAG_Registry deployed to ${address}`);
  console.log("Deployment artifact written to contracts/deployments/localhost.json");
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
```

- [ ] **Step 2: Add `deployments/` to `contracts/.gitignore`**

Append to `contracts/.gitignore`:

```
deployments/
```

- [ ] **Step 3: Start a local Hardhat node in the background**

Run (from `contracts/`, background process): `npx hardhat node`
Expected: prints 20 funded accounts and private keys, listens on
`http://127.0.0.1:8545`. Leave it running for this step and Task 4.

- [ ] **Step 4: Run the deploy script against the local node and verify the artifact**

Run (from `contracts/`): `npx hardhat run scripts/deploy.ts --network localhost`
Expected: prints `DeRAG_Registry deployed to 0x...`.

Then verify the artifact file:
```bash
cat contracts/deployments/localhost.json | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['address']); print(len(d['abi']), 'ABI entries')"
```
Expected: a `0x`-prefixed 42-char address, and an ABI entry count matching
the contract's public functions/constructor/mapping getters (roughly 8-10
entries).

- [ ] **Step 5: Commit**

```bash
git add contracts/scripts/deploy.ts contracts/.gitignore
git commit -m "Add deploy script that writes address+ABI to deployments/localhost.json"
```

---

### Task 3: registry_client.py — connect, write functions, read functions

**Files:**
- Create: `blockchain/registry_client.py`
- Test: `tests/test_blockchain_registry_client.py` (unit tests only in this
  task — the integration test comes in Task 4)

**Interfaces:**
- Consumes: `contracts/deployments/localhost.json` (from Task 2), read via
  a path computed relative to this file:
  `Path(__file__).resolve().parent.parent / "contracts" / "deployments" / "localhost.json"`.
  Consumes env vars `WEB3_PROVIDER_URI` and `PRIVATE_KEY`.
- Produces (consumed by Task 4's integration test and by Milestone 5/8
  later):
  - `connect() -> tuple[Web3, Contract]`
  - `register_document(doc_hash: str, ipfs_cid: str, signing_key: str | None = None) -> dict`
  - `update_reputation(doc_hash: str, delta: int, signing_key: str | None = None) -> dict`
  - `verify_integrity(doc_hash: str) -> dict` — keys `exists` (bool),
    `reputation` (int), `publisher` (str)
  - `get_document(doc_hash: str) -> dict` — keys `publisher` (str),
    `content_hash` (str, `0x`-prefixed hex — `"0x" + bytes.hex()`),
    `ipfs_cid` (str), `reputation_score` (int), `timestamp` (int),
    `is_registered` (bool)
  - `_doc_hash_to_bytes32(doc_hash: str) -> bytes` (private helper, tested
    directly since it's pure logic with no node dependency)

- [ ] **Step 1: Write the failing unit tests for the pure bytes32 helper**

Create `tests/test_blockchain_registry_client.py`:

```python
from blockchain.registry_client import _doc_hash_to_bytes32


def test_doc_hash_to_bytes32_round_trips_hex_string_without_prefix():
    doc_hash = "a" * 64
    result = _doc_hash_to_bytes32(doc_hash)
    assert isinstance(result, bytes)
    assert len(result) == 32
    assert result.hex() == doc_hash


def test_doc_hash_to_bytes32_accepts_0x_prefixed_string():
    doc_hash = "0x" + "b" * 64
    result = _doc_hash_to_bytes32(doc_hash)
    assert len(result) == 32
    assert result.hex() == "b" * 64
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_blockchain_registry_client.py -v`
Expected: FAIL with `ImportError: cannot import name '_doc_hash_to_bytes32'` (module doesn't exist yet).

- [ ] **Step 3: Write registry_client.py**

Create `blockchain/registry_client.py`:

```python
import json
import os
from pathlib import Path

from eth_account import Account
from web3 import Web3
from web3.contract import Contract

DEPLOYMENT_PATH = (
    Path(__file__).resolve().parent.parent / "contracts" / "deployments" / "localhost.json"
)


def _doc_hash_to_bytes32(doc_hash: str) -> bytes:
    hexstr = doc_hash if doc_hash.startswith("0x") else "0x" + doc_hash
    return Web3.to_bytes(hexstr=hexstr)


def connect() -> tuple[Web3, Contract]:
    provider_uri = os.environ["WEB3_PROVIDER_URI"]
    w3 = Web3(Web3.HTTPProvider(provider_uri))
    with open(DEPLOYMENT_PATH) as f:
        deployment = json.load(f)
    contract = w3.eth.contract(address=deployment["address"], abi=deployment["abi"])
    return w3, contract


def _send_transaction(w3: Web3, function_call, signing_key: str) -> dict:
    account = Account.from_key(signing_key)
    tx = function_call.build_transaction(
        {
            "from": account.address,
            "nonce": w3.eth.get_transaction_count(account.address),
            "gasPrice": w3.eth.gas_price,
            "chainId": w3.eth.chain_id,
        }
    )
    signed = account.sign_transaction(tx)
    tx_hash = w3.eth.send_raw_transaction(signed.rawTransaction)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    return dict(receipt)


def register_document(doc_hash: str, ipfs_cid: str, signing_key: str | None = None) -> dict:
    w3, contract = connect()
    key = signing_key or os.environ["PRIVATE_KEY"]
    hash_bytes = _doc_hash_to_bytes32(doc_hash)
    function_call = contract.functions.registerDocument(hash_bytes, ipfs_cid)
    return _send_transaction(w3, function_call, key)


def update_reputation(doc_hash: str, delta: int, signing_key: str | None = None) -> dict:
    w3, contract = connect()
    key = signing_key or os.environ["PRIVATE_KEY"]
    hash_bytes = _doc_hash_to_bytes32(doc_hash)
    function_call = contract.functions.updateReputation(hash_bytes, delta)
    return _send_transaction(w3, function_call, key)


def verify_integrity(doc_hash: str) -> dict:
    w3, contract = connect()
    hash_bytes = _doc_hash_to_bytes32(doc_hash)
    exists, reputation, publisher = contract.functions.verifyIntegrity(hash_bytes).call()
    return {"exists": exists, "reputation": reputation, "publisher": publisher}


def get_document(doc_hash: str) -> dict:
    w3, contract = connect()
    hash_bytes = _doc_hash_to_bytes32(doc_hash)
    publisher, content_hash, ipfs_cid, reputation_score, timestamp, is_registered = (
        contract.functions.documentLedger(hash_bytes).call()
    )
    return {
        "publisher": publisher,
        "content_hash": "0x" + content_hash.hex(),
        "ipfs_cid": ipfs_cid,
        "reputation_score": reputation_score,
        "timestamp": timestamp,
        "is_registered": is_registered,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_blockchain_registry_client.py -v`
Expected: both tests pass.

- [ ] **Step 5: Commit**

```bash
git add blockchain/registry_client.py tests/test_blockchain_registry_client.py
git commit -m "Add registry_client.py web3.py bridge with bytes32 helper"
```

---

### Task 4: Python integration test (slow, requires live Hardhat node)

**Files:**
- Modify: `tests/test_blockchain_registry_client.py` (append integration tests)
- Modify: `pyproject.toml` (register `slow` pytest marker)

**Interfaces:**
- Consumes: `blockchain.registry_client.{register_document, update_reputation,
  verify_integrity, get_document}` (Task 3), a live `npx hardhat node` at
  `http://127.0.0.1:8545`, and `contracts/deployments/localhost.json`
  (Task 2, requires `deploy.ts` to have been run against that node).
- Produces: nothing consumed by later tasks — this is the final
  verification layer for Milestone 3.

- [ ] **Step 1: Register the `slow` marker**

In `pyproject.toml`, add under `[tool.pytest.ini_options]`:

```toml
markers = [
    "slow: requires a live `npx hardhat node` running on WEB3_PROVIDER_URI",
]
```

- [ ] **Step 2: Write the failing integration test**

Append to `tests/test_blockchain_registry_client.py`:

```python
import os

import pytest

from blockchain.registry_client import (
    get_document,
    register_document,
    update_reputation,
    verify_integrity,
)

# Hardhat's deterministic default account #1 and #2 private keys (printed by
# `npx hardhat node` on startup) — publicly known test-only keys, no real funds.
PUBLISHER_1_KEY = "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690"
PUBLISHER_2_KEY = "0x5de4111afa1a4b94908f83103eb1f1706367c2e68ca870fc3fb9a804cdab365"


@pytest.mark.slow
def test_register_update_and_read_back_end_to_end():
    doc_hash = "1" * 64
    register_document(doc_hash, ipfs_cid="QmTestCID", signing_key=PUBLISHER_1_KEY)

    integrity = verify_integrity(doc_hash)
    assert integrity["exists"] is True
    assert integrity["reputation"] == 0

    update_reputation(doc_hash, delta=-7)
    integrity_after = verify_integrity(doc_hash)
    assert integrity_after["reputation"] == -7

    doc = get_document(doc_hash)
    assert doc["ipfs_cid"] == "QmTestCID"
    assert doc["reputation_score"] == -7
    assert doc["is_registered"] is True


@pytest.mark.slow
def test_publisher_attribution_is_per_account_not_global():
    hash_a = "2" * 64
    hash_b = "3" * 64

    register_document(hash_a, ipfs_cid="cid-a", signing_key=PUBLISHER_1_KEY)
    register_document(hash_b, ipfs_cid="cid-b", signing_key=PUBLISHER_2_KEY)

    doc_a = get_document(hash_a)
    doc_b = get_document(hash_b)
    assert doc_a["publisher"] != doc_b["publisher"]

    integrity_a = verify_integrity(hash_a)
    assert integrity_a["publisher"] == doc_a["publisher"]

    # Same hash from a second account must revert (dedup rule), not silently
    # reassign the publisher.
    with pytest.raises(Exception):
        register_document(hash_a, ipfs_cid="cid-a-again", signing_key=PUBLISHER_2_KEY)
```

- [ ] **Step 3: Confirm the test fails without a running node**

Run: `uv run pytest tests/test_blockchain_registry_client.py -v -m slow`
Expected: FAIL — connection error (`ConnectionError` / `CannotHandleRequest`)
since no Hardhat node is listening yet, or `FileNotFoundError` if
`contracts/deployments/localhost.json` doesn't exist yet. This confirms
the test genuinely exercises the live path rather than trivially passing.

- [ ] **Step 4: Start the node, deploy, and run the integration tests**

In one terminal (background), from `contracts/`: `npx hardhat node`

In a second terminal, from `contracts/`: `npx hardhat run scripts/deploy.ts --network localhost`

Set environment and run the integration tests from the repo root:
```bash
export WEB3_PROVIDER_URI=http://127.0.0.1:8545
export PRIVATE_KEY=0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80
uv run pytest tests/test_blockchain_registry_client.py -v -m slow
```
(`PRIVATE_KEY` here is Hardhat's deterministic default Account #0 key,
printed by `npx hardhat node` on startup — the oracle/deployer role;
`update_reputation` in the first test defaults to it.)

Expected: both `slow`-marked tests pass.

- [ ] **Step 5: Run the full test suite (contracts + Python) to confirm nothing regressed**

```bash
cd contracts && npx hardhat test && cd ..
uv run pytest
```
Expected: all Hardhat tests pass; all pytest tests pass (the two new
`slow` tests will only pass if the node from Step 4 is still running —
otherwise they're expected to fail here, which is fine for this
verification pass since Step 4 already confirmed them green).

- [ ] **Step 6: Commit**

```bash
git add tests/test_blockchain_registry_client.py pyproject.toml
git commit -m "Add slow-marked integration tests for registry_client end-to-end flow"
```

---

### Task 5: DECISIONS.md entry and README update

**Files:**
- Modify: `DECISIONS.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: nothing (documentation only).
- Produces: nothing consumed by later tasks — this is the milestone's
  written record for Milestone 5/8 to read.

- [ ] **Step 1: Add the Milestone 3 entry to DECISIONS.md**

Append to `DECISIONS.md`:

```markdown

## Milestone 3 — Blockchain Registry (2026-08-16)

- **Ownership/oracle model**: `address public owner` set to `msg.sender`
  in the constructor; constructor also auto-authorizes the deployer as
  the first oracle (`authorizedOracles[msg.sender] = true`), matching the
  spec's "Account #0 acts as Admin/Oracle" framing since Account #0 is
  the deploying account in Milestone 5's standard Hardhat workflow.
  `authorizeOracle(address, bool)` is owner-gated and lets additional
  oracles be added later without a redeploy.
- **`registerDocument` reverts on duplicate hash** rather than
  silently no-opping — protects on-chain integrity even if called
  out-of-band. Milestone 5's ingestion script is responsible for calling
  `verify_integrity()` or `get_document()` before calling
  `register_document()`, to avoid wasted gas on an expected revert. This
  is a deliberately separate, explicit check from Milestone 2's Chroma
  upsert-dedup behavior (which is coincidental idempotency, not a
  designed guarantee).
- **Publisher-per-call signing design**: `registry_client.py`'s
  `register_document(doc_hash, ipfs_cid, signing_key=None)` accepts an
  explicit signing key, defaulting to the `PRIVATE_KEY` env var (Account
  #0) only when not given. Milestone 5's ingestion script passes
  Accounts #1-#5's keys explicitly to simulate distinct publishers, since
  the spec assigns those accounts as publishers (plural, different
  accounts) — this was designed in now rather than retrofitted later.
  `update_reputation(doc_hash, delta, signing_key=None)` follows the same
  optional-override pattern but is expected to always use the
  default (oracle account), since oracle is a single designated role in
  this milestone's scope.
- **Deployment artifact handoff**: `contracts/scripts/deploy.ts` writes
  `contracts/deployments/localhost.json` (`{"address": ..., "abi": ...}`)
  on every deploy. `blockchain/registry_client.py`'s `connect()` reads
  this file directly (path computed relative to the module file), rather
  than requiring a human to copy the address into `CONTRACT_ADDRESS`
  after every fresh `npx hardhat node` restart. `contracts/deployments/`
  is gitignored (build output, like `artifacts/`/`cache/`).
- **bytes32 conversion, verified working**: `Web3.to_bytes(hexstr="0x" +
  doc_hash)` where `doc_hash` is the 64-char SHA-256 hex string from
  Milestone 1. Verified round-trip (`to_bytes(...).hex() ==
  doc_hash`) directly against the real `web3==6.20.3` install before
  writing `registry_client.py`.
- **`documentLedger`'s auto-generated getter returns the full six struct
  fields** (publisher, contentHash, ipfsCID, reputationScore, timestamp,
  isRegistered) — confirmed by compiling a scratch contract with the
  exact struct shape and inspecting the generated ABI, not assumed from
  general Solidity docs. No separate getter function was added.
  `registry_client.py`'s `get_document()` calls this auto-getter
  directly; `verify_integrity()` remains a thin wrapper on the contract's
  3-tuple `verifyIntegrity()` for the cheaper existence/reputation-only
  check. Milestone 8's audit trail UI should call `get_document()` when
  it needs `ipfs_cid`/`timestamp`, `verify_integrity()` otherwise.
```

- [ ] **Step 2: Update README's Status section**

In `README.md`, extend the `## Status` section with a Milestone 3 line
following the existing style (see the Milestone 1/2 lines already
there):

```markdown

Milestone 3: on-chain document registry (`contracts/contracts/DeRAG_Registry.sol`)
and Python bridge (`blockchain/registry_client.py`) implemented and tested —
owner/oracle-gated reputation ledger with publisher attribution and
duplicate-hash protection. Start a local node and deploy with:

```
cd contracts && npx hardhat node
# in a second terminal:
cd contracts && npx hardhat run scripts/deploy.ts --network localhost
```
```

- [ ] **Step 3: Commit**

```bash
git add DECISIONS.md README.md
git commit -m "Document Milestone 3 decisions: registry ownership, dedup, deployment artifact"
```
