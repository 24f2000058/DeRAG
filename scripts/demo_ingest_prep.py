from pathlib import Path

from pipeline.ingest_prep import ingest_prep


def main() -> None:
    sample_path = Path(__file__).resolve().parent.parent / "data" / "sample.txt"
    raw_text = sample_path.read_text(encoding="utf-8")

    records = ingest_prep(raw_text)

    print(f"Ingested {len(records)} chunk(s) from {sample_path}\n")
    for record in records:
        print(f"chunk_id={record['chunk_id']}")
        print(f"  text ({len(record['text'])} chars): {record['text'][:80]}...")
        print(f"  hash: {record['hash']}")
        print(f"  embedding dims: {len(record['embedding'])}")
        print()


if __name__ == "__main__":
    main()
