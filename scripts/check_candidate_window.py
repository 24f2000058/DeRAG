"""Spec section 3a pre-flight check. Before the regression test is allowed
to rely on the reranker margins, confirm that for each poison chunk's
target query BOTH the poison chunk AND its ground-truth legit counterpart
actually survive into vector_store.query(query, k=10). Reputation seeding
cannot rescue a chunk that never entered the candidate pool.

Run AFTER scripts/seed_reputation.py, against the live-ingested corpus.

Usage:
    uv run python scripts/check_candidate_window.py
    uv run python scripts/check_candidate_window.py --k 10
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval.vector_store import query

DEFAULT_CORPUS_DIR = Path("data/demo_corpus")


def check(corpus_dir: Path = DEFAULT_CORPUS_DIR, k: int = 10) -> int:
    manifest_path = corpus_dir / "manifest.json"
    chunks = json.loads(manifest_path.read_text(encoding="utf-8"))["chunks"]
    by_chunk_id = {c["chunk_id"]: c for c in chunks}
    all_ok = True
    pairs_checked = 0

    for entry in chunks:
        if not entry["is_poisoned"] or not entry["target_queries"]:
            continue
        gt_id = entry["ground_truth_chunk_id"]
        if not gt_id or gt_id not in by_chunk_id:
            print(
                f"[preflight] ERROR: poison chunk {entry['chunk_id']!r} has "
                f"ground_truth_chunk_id={gt_id!r}, which is not a chunk in "
                f"{manifest_path}"
            )
            return 1
        ground_truth = by_chunk_id[gt_id]
        for question in entry["target_queries"]:
            pairs_checked += 1
            hits = query(question, k=k)
            present = {hit["metadata"]["source_file"] for hit in hits}
            poison_ok = entry["filename"] in present
            legit_ok = ground_truth["filename"] in present
            all_ok = all_ok and poison_ok and legit_ok
            print(f"query: {question!r}")
            print(f"  poison {entry['filename']}: {'OK' if poison_ok else 'MISSING'}")
            print(f"  legit  {ground_truth['filename']}: {'OK' if legit_ok else 'MISSING'}")
            print(f"  top-{k}: {[hit['metadata']['source_file'] for hit in hits]}")

    if pairs_checked == 0:
        print(
            f"[preflight] ERROR: no poisoned entries with target_queries found in "
            f"{manifest_path} — wrong --corpus-dir?"
        )
        return 1

    print("[preflight] PASS" if all_ok else "[preflight] FAIL — see MISSING rows above")
    return 0 if all_ok else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Candidate-window pre-flight for the demo corpus.")
    parser.add_argument("--corpus-dir", default=str(DEFAULT_CORPUS_DIR))
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    sys.exit(check(corpus_dir=Path(args.corpus_dir), k=args.k))


if __name__ == "__main__":
    main()
