"""End-to-end ingestion pipeline: extract text, upload to IPFS, register on-chain, store in Chroma.

Requires these environment variables to be set (see .env.example):
    WEB3_PROVIDER_URI   e.g. http://127.0.0.1:8545
    IPFS_API_URL        e.g. /ip4/127.0.0.1/tcp/5001

Usage:
    python scripts/ingest.py --input data/sample.txt
    python scripts/ingest.py --input data/demo_corpus/
"""

import argparse
import sys
from pathlib import Path

import requests
from web3.exceptions import ContractLogicError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pypdf
import pypdf.errors

from blockchain.registry_client import get_document, register_document
from pipeline.ingest_prep import ingest_prep
from retrieval.vector_store import add_documents
from storage.ipfs_client import IPFSError, upload_text, verify_hash

# Hardhat's deterministic default Account #1 private key — same constant
# used in tests/test_blockchain_registry_client.py as PUBLISHER_1_KEY.
# Publicly known test-only key, no real funds.
PUBLISHER_1_KEY = "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d"

_TEXT_SUFFIXES = {".txt", ".pdf"}


def _extract_text(path: Path) -> str:
    if path.suffix == ".pdf":
        reader = pypdf.PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8")


def _collect_input_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(
        p for p in input_path.iterdir() if p.is_file() and p.suffix in _TEXT_SUFFIXES
    )


def _log_chunk_failure(source_file: str, chunk_id: int, doc_hash: str, reason: str) -> None:
    print(
        f"[ingest] FAILED chunk source_file={source_file} chunk_id={chunk_id} "
        f"doc_hash={doc_hash}: {reason}",
        file=sys.stderr,
    )


def _log_file_failure(source_file: str, reason: str) -> None:
    print(
        f"[ingest] FAILED file source_file={source_file}: {reason}",
        file=sys.stderr,
    )


def _resolve_chunk_cid(record: dict, source_file: str, publisher_key: str) -> str | None:
    doc_hash = record["hash"]
    try:
        document = get_document(doc_hash)
        if document["is_registered"]:
            return document["ipfs_cid"]

        cid = upload_text(record["text"])
        verified = verify_hash(cid, doc_hash)
        if not verified:
            _log_chunk_failure(
                source_file, record["chunk_id"], doc_hash,
                f"verify_hash returned False for cid={cid}",
            )
            return None

        register_document(doc_hash, cid, signing_key=publisher_key)
        return cid
    except (IPFSError, requests.exceptions.ConnectionError, ContractLogicError) as exc:
        _log_chunk_failure(source_file, record["chunk_id"], doc_hash, repr(exc))
        return None


def run_ingest(input_path: str, publisher_key: str = PUBLISHER_1_KEY, collection=None) -> int:
    if not Path(input_path).exists():
        print(f"[ingest] ERROR: input path does not exist: {input_path}", file=sys.stderr)
        return 1

    any_failures = False

    input_files = _collect_input_files(Path(input_path))
    if not input_files:
        print(f"[ingest] WARNING: no .txt/.pdf files found in {input_path}", file=sys.stderr)

    for file_path in input_files:
        try:
            raw_text = _extract_text(file_path)
            records = ingest_prep(raw_text)
        except (UnicodeDecodeError, OSError, pypdf.errors.PyPdfError) as exc:
            _log_file_failure(file_path.name, repr(exc))
            any_failures = True
            continue

        cid_by_hash: dict[str, str] = {}
        surviving_records = []
        for record in records:
            cid = _resolve_chunk_cid(record, source_file=file_path.name, publisher_key=publisher_key)
            if cid is None:
                any_failures = True
                continue
            cid_by_hash[record["hash"]] = cid
            surviving_records.append(record)

        add_documents(
            surviving_records,
            source_file=file_path.name,
            collection=collection,
            cid_by_hash=cid_by_hash,
        )
        print(f"[ingest] {file_path.name}: {len(surviving_records)} chunk(s) stored")

    return 1 if any_failures else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest text/PDF documents into DeRAG.")
    parser.add_argument("--input", required=True, help="Path to a file or directory of .txt/.pdf files")
    parser.add_argument(
        "--publisher-key",
        default=PUBLISHER_1_KEY,
        help="Private key used to sign on-chain registrations (defaults to Hardhat Account #1)",
    )
    args = parser.parse_args()
    sys.exit(run_ingest(args.input, publisher_key=args.publisher_key))


if __name__ == "__main__":
    main()
