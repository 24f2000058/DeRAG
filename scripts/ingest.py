import sys
from pathlib import Path

import requests
from web3.exceptions import ContractLogicError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pypdf

from blockchain.registry_client import get_document, register_document
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
