# Milestone 3: DeRAG_Registry Smart Contract + Python Bridge — Design

## Purpose

Build the on-chain document registry (`DeRAG_Registry.sol`) and its Python
bridge (`blockchain/registry_client.py`), fully isolated from IPFS,
ChromaDB, and Streamlit. This is the trust ledger the rest of DeRAG reads
from and writes to: publisher attribution, content-hash integrity, and a
mutable reputation score per document. Milestone 5 (ingestion) writes to
it; Milestone 8 (audit trail UI) reads from it.

The spec fixes the `Document` struct, the `authorizedOracles` mapping, and
three function signatures (`registerDocument`, `updateReputation`,
`verifyIntegrity`). It leaves ownership and oracle-authorization
unspecified — this design resolves that gap.

## Verified Environment (do not re-derive — checked against the real
installs, not assumed from training data)

- **Solidity `0.8.20`** — already set in `contracts/hardhat.config.ts`
  (`solidity: "0.8.20"`), matches the spec's `^0.8.20`. No change needed.
- **Hardhat `2.29.0`**, `@nomicfoundation/hardhat-toolbox` `5.0.0` —
  installed in `contracts/node_modules`, `npx hardhat compile` verified
  working against the placeholder contract.
- **web3.py `6.20.3`** — installed via `pyproject.toml`'s `web3>=6,<7`
  pin. Verified real signing pattern by direct execution:
  ```python
  acct = Account.from_key(private_key)
  signed = acct.sign_transaction(tx_dict)
  tx_hash = w3.eth.send_raw_transaction(signed.rawTransaction)
  ```
  Note: the signed-transaction attribute is **`rawTransaction`**
  (camelCase), not `raw_transaction` — confirmed via `dir()` on a real
  `SignedTransaction` object, not assumed from docs.
- **bytes32 conversion** — `Web3.to_bytes(hexstr="0x" + doc_hash_hex)`
  round-trips a 64-char SHA-256 hex string to a 32-byte value and back
  correctly. Verified directly (`to_bytes(...).hex() == original_hex`).
- **`documentLedger` auto-getter returns all six struct fields** —
  compiled a scratch contract with the exact `Document` struct
  (`address publisher; bytes32 contentHash; string ipfsCID; int256
  reputationScore; uint256 timestamp; bool isRegistered;`) behind
  `mapping(bytes32 => Document) public documentLedger;` and inspected the
  generated ABI. Confirmed: the getter takes the `bytes32` key and returns
  **all six named fields**, not a subset. No custom getter is needed for
  Milestone 8's audit trail. (Scratch contract and its typechain output
  were deleted after verification — not part of the deliverable.)

## Carried Forward from Milestone 2

`ipfs_cid` is stored in ChromaDB metadata as `""` (empty string), never
`None` — Chroma 1.5.9 silently drops `None`-valued metadata fields on
upsert. Milestone 5's "does this chunk still need blockchain
registration" check must test `ipfs_cid == ""`, not `is None`. This
milestone's contract and client don't implement that check themselves
(Milestone 5's job), but the `register_document()` signature is shaped to
make that check trivial to write there.

## Contract Design (`contracts/contracts/DeRAG_Registry.sol`)

Replaces `Placeholder.sol`.

```solidity
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
```

- **Constructor**: `owner = msg.sender`; `authorizedOracles[msg.sender] =
  true`. Matches the spec's "Account #0 acts as Admin/Oracle" framing —
  Account #0 is the deploying account in Milestone 5's standard Hardhat
  workflow, so this auto-authorization requires no follow-up call.
- **`authorizeOracle(address _account, bool _authorized) external
  onlyOwner`** — owner-gated. Populates `authorizedOracles` for any
  account beyond the deployer. Kept in scope (not cut as YAGNI) because a
  `mapping(address => bool) authorizedOracles` with no way to ever add an
  entry beyond the constructor is the kind of gap a reviewer flags, and
  the cost is a five-line function plus one modifier.
- **`registerDocument(bytes32 _hash, string memory _cid) external`** —
  publisher is `msg.sender` (no explicit publisher parameter in the
  spec's signature — the calling account IS the publisher). **Reverts**
  if `documentLedger[_hash].isRegistered` is already `true` — protects
  on-chain integrity even if called out-of-band, outside Milestone 5's
  ingestion script. Milestone 5's ingestion script is responsible for
  calling `verifyIntegrity()` (or `get_document()`) before calling
  `register_document()`, to avoid wasted gas on an expected revert. This
  is a deliberately separate, explicit check — not inherited from
  Milestone 2's Chroma upsert-dedup behavior, which is coincidental
  idempotency, not a designed guarantee (see Milestone 2 DECISIONS.md).
- **`updateReputation(bytes32 _hash, int256 _delta) external`** — gated
  on `authorizedOracles[msg.sender] == true`; reverts if
  `documentLedger[_hash].isRegistered` is `false`. `reputationScore` has
  **no floor** — it is allowed to go negative, per spec ("Can be negative
  for known poisoned data").
- **`verifyIntegrity(bytes32 _hash) external view returns (bool exists,
  int256 reputation, address publisher)`** — spec's exact 3-tuple
  signature, for cheap existence/reputation checks.
- **`documentLedger` public getter** — auto-generated by Solidity from
  the public mapping declaration, returns all six fields (verified
  above). Serves Milestone 8's audit trail need for `ipfsCID` and
  `timestamp` without any additional contract code.

## Python Bridge Design (`blockchain/registry_client.py`)

- **`connect()`** — reads `WEB3_PROVIDER_URI` from environment, connects
  via `Web3(Web3.HTTPProvider(...))`, loads contract address + ABI from
  `contracts/deployments/localhost.json` (written by the deploy script —
  see below). Returns a configured `w3.eth.contract(address=..., abi=...)`
  instance (or a small wrapper holding it).
- **`register_document(doc_hash: str, ipfs_cid: str, signing_key: str |
  None = None) -> tx receipt/hash`** — `signing_key` defaults to the
  `PRIVATE_KEY` env var (Account #0 / deployer / oracle role) when not
  given. Milestone 5's ingestion script passes Accounts #1-#5's keys
  explicitly to simulate distinct publishers, since the spec assigns
  those accounts as publishers (plural) — this parameter is designed in
  now rather than retrofitted later. Converts `doc_hash` (64-char hex
  string) to `bytes32` via `Web3.to_bytes(hexstr="0x" + doc_hash)`, builds
  the transaction, signs with `Account.from_key(signing_key)`, sends via
  `w3.eth.send_raw_transaction(signed.rawTransaction)`, waits for the
  receipt.
- **`update_reputation(doc_hash: str, delta: int, signing_key: str | None
  = None)`** — same shape; `signing_key` defaults to `PRIVATE_KEY`
  (oracle account), since oracle is a single designated role in this
  milestone's scope (no per-call oracle override needed — only
  `authorizeOracle` changes who else can act as one).
- **`verify_integrity(doc_hash: str) -> dict`** — thin wrapper on the
  contract's `verifyIntegrity()`, returns `{"exists": bool, "reputation":
  int, "publisher": str}`.
- **`get_document(doc_hash: str) -> dict`** — calls the `documentLedger`
  getter directly, returns all six fields as a dict (`publisher`,
  `content_hash`, `ipfs_cid`, `reputation_score`, `timestamp`,
  `is_registered`). Implemented alongside `verify_integrity()` rather than
  instead of it — both are free (already exposed on-chain), and it isn't
  yet certain which shape Milestone 8 will consume, so having both now
  avoids a retrofit.

## Deployment Artifact Handoff (`contracts/scripts/deploy.ts`)

Deploys `DeRAG_Registry` to the local Hardhat node, then writes
`contracts/deployments/localhost.json`:

```json
{ "address": "0x...", "abi": [ ... ] }
```

`registry_client.py`'s `connect()` reads this file directly. Chosen over
requiring a human to copy the address into `CONTRACT_ADDRESS` manually
after every fresh `npx hardhat node` restart — this file regenerates on
every `deploy.ts` run, so repeated local dev/demo cycles stay
error-proof. `contracts/deployments/` is added to `.gitignore` (matches
`contracts/artifacts/` and `contracts/cache/` — build output, not a
tracked deliverable).

## Testing Plan (TDD — Hardhat tests must fail red before the contract
exists)

`contracts/test/DeRAG_Registry.test.ts`:
1. Successful `registerDocument` sets `publisher == msg.sender` and
   `reputationScore == 0`.
2. Duplicate `registerDocument` on the same hash reverts.
3. `authorizeOracle` reverts when called by a non-owner.
4. `updateReputation` reverts when called by a non-authorized address.
5. `updateReputation` correctly adjusts `reputationScore`, including
   into negative territory.
6. `verifyIntegrity` returns `(false, 0, address(0))` for an unregistered
   hash, and correct values for a registered one.
7. `documentLedger(hash)` (the auto-getter) returns all six fields
   correctly for a registered document — the load-bearing test for
   Milestone 8.

Python integration test (`tests/test_blockchain_registry_client.py`,
marked slow, documented as requiring `npx hardhat node` running in
another terminal):
- End-to-end register → update reputation → read back via both
  `verify_integrity()` and `get_document()`.
- Registers the same document hash from two different signing accounts
  (two different `signing_key` values) to confirm publisher attribution
  is per-account/per-transaction, not global — i.e., the second call with
  a *different* hash from a different account succeeds and records that
  account as publisher (the *same* hash from a second account must
  revert, per the dedup rule — this test also exercises that revert
  path).

## Definition of Done

- `npx hardhat test` passes fully inside `/contracts`.
- `npx hardhat node` running in one terminal lets the Python integration
  tests pass in another (`uv run pytest -m slow` or equivalent).
- `DECISIONS.md` gets a new Milestone 3 entry recording: the
  ownership/oracle model, the publisher-per-call signing design (and
  why, for Milestone 5), the deployment-artifact file path and read
  pattern, the verified bytes32 conversion approach, and explicit
  confirmation that `documentLedger`'s auto-getter returns the full six
  fields with no separate getter required.

## Out of Scope (Milestone 3)

IPFS, ChromaDB, Streamlit, generation/reranking logic. Milestone 5's
actual ingestion-time dedup check against `ipfs_cid == ""` (this
milestone only makes that check possible to write cheaply — it doesn't
implement it). Any UI for the audit trail (Milestone 8).
