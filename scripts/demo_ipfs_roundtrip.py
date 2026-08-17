import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.hashing import hash_chunk
from storage.ipfs_client import fetch_text, upload_text, verify_hash


def main() -> None:
    text = "DeRAG Milestone 4: this chunk travels through real IPFS."
    expected_hash = hash_chunk(text)
    print(f"Original text: {text!r}")
    print(f"Pre-upload hash: {expected_hash}\n")

    cid = upload_text(text)
    print(f"Uploaded to IPFS, CID: {cid}\n")

    fetched = fetch_text(cid)
    print(f"Fetched text: {fetched!r}")
    assert fetched == text, "fetched text does not match original"

    is_valid = verify_hash(cid, expected_hash)
    print(f"verify_hash(cid, pre-upload hash): {is_valid}")
    assert is_valid, "hash verification failed after real IPFS round trip"

    print("\nRound trip succeeded: upload -> CID -> fetch -> hash match.")


if __name__ == "__main__":
    main()
