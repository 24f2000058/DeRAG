"""Ingest the Milestone 9 demo corpus with curated publisher keys, then
seed each chunk's on-chain reputation to its manifest target value.

Idempotent by design: for every chunk it reads the current on-chain
reputation and applies delta = target_reputation - current, skipping any
chunk already at target. Safe to re-run after a partial failure.

Requires, all live at once:
  - a Hardhat node on WEB3_PROVIDER_URI, contract deployed via
    contracts/scripts/deploy.ts
  - an IPFS (Kubo) daemon on IPFS_API_URL
  - PRIVATE_KEY set to Hardhat Account #0 (the deploy-time reputation
    oracle) in the environment / .env

Usage:
    uv run python scripts/seed_reputation.py
    uv run python scripts/seed_reputation.py --corpus-dir data/demo_corpus
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from blockchain.registry_client import RegistryError, get_document, update_reputation
from pipeline.ingest_prep import ingest_prep
from scripts.ingest import PUBLISHER_KEYS, run_ingest

DEFAULT_CORPUS_DIR = Path("data/demo_corpus")


class SeedError(Exception):
    pass


def load_manifest(corpus_dir: Path) -> list[dict]:
    return json.loads((corpus_dir / "manifest.json").read_text(encoding="utf-8"))["chunks"]


def resolve_doc_hash(text_path: Path) -> str:
    records = ingest_prep(text_path.read_text(encoding="utf-8"))
    if len(records) != 1:
        raise SeedError(f"{text_path.name}: expected exactly 1 chunk, got {len(records)}")
    return records[0]["hash"]


def seed(corpus_dir: Path = DEFAULT_CORPUS_DIR) -> None:
    chunks = load_manifest(corpus_dir)

    # 1. Ingest every file with its own publisher key. run_ingest is itself
    #    idempotent (it skips already-registered chunks), so a re-run is cheap.
    for entry in chunks:
        signing_key = PUBLISHER_KEYS[entry["publisher_key"]]
        rc = run_ingest(str(corpus_dir / entry["filename"]), publisher_key=signing_key)
        if rc != 0:
            raise SeedError(f"ingest failed (exit {rc}) for {entry['filename']}")

    # 2. Drive reputation to target idempotently.
    for entry in chunks:
        target = entry["target_reputation"]
        doc_hash = resolve_doc_hash(corpus_dir / entry["filename"])
        try:
            document = get_document(doc_hash)
        except RegistryError as exc:
            raise SeedError(f"could not read reputation for {entry['filename']}: {exc}") from exc
        if not document["is_registered"]:
            raise SeedError(
                f"{entry['filename']} is not registered on-chain after ingest — "
                "check the ingest logs above for a skipped chunk"
            )
        current = document["reputation_score"]
        if current == target:
            print(f"[seed] skip  {entry['filename']}: already at {target}")
            continue
        delta = target - current
        try:
            update_reputation(doc_hash, delta)  # signs with PRIVATE_KEY (oracle / Account #0)
        except RegistryError as exc:
            raise SeedError(f"updateReputation failed for {entry['filename']}: {exc}") from exc
        print(f"[seed] set   {entry['filename']}: {current} -> {target} (delta {delta:+d})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed demo-corpus reputations (idempotent).")
    parser.add_argument("--corpus-dir", default=str(DEFAULT_CORPUS_DIR))
    args = parser.parse_args()
    try:
        seed(corpus_dir=Path(args.corpus_dir))
    except SeedError as exc:
        sys.exit(f"[seed] ERROR: {exc}")
    print("[seed] done")


if __name__ == "__main__":
    main()
