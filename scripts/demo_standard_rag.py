import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.ingest_prep import ingest_prep
from retrieval.standard_rag import standard_rag_answer_context
from retrieval.vector_store import add_documents, init_store


def main() -> None:
    sample_path = Path(__file__).resolve().parent.parent / "data" / "sample.txt"
    raw_text = sample_path.read_text(encoding="utf-8")

    collection = init_store()
    records = ingest_prep(raw_text)
    add_documents(records, source_file=str(sample_path.name), collection=collection)
    print(f"Stored {len(records)} chunk(s) from {sample_path} into ChromaDB\n")

    query_text = "How does DeRAG defend against data poisoning in retrieval?"
    hits = standard_rag_answer_context(query_text, collection=collection)

    print(f"Query: {query_text}\n")
    for rank, hit in enumerate(hits, start=1):
        print(f"#{rank} similarity={hit['similarity']:.4f} source={hit['metadata']['source_file']}")
        print(f"    {hit['text'][:120]}...")
        print()


if __name__ == "__main__":
    main()
