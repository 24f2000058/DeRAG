"""Build the Milestone 9 demo corpus from real SciFact data.

Downloads (and caches under data/.scifact_cache/) the SciFact release
archive, filters claims to short single-cite SUPPORTED abstracts, and
writes 50 legitimate .txt chunks + 3 fabricated poison chunk stubs +
manifest.json into data/demo_corpus/.

Deterministic: same archive in -> byte-identical corpus + manifest out.
The 3 poison .txt files themselves are hand-authored in a later task;
this script only writes their manifest entries.

Usage:
    uv run python scripts/build_demo_corpus.py
    uv run python scripts/build_demo_corpus.py --corpus-dir data/demo_corpus
"""

import argparse
import io
import json
import re
import sys
import tarfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SCIFACT_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
CACHE_DIR = Path("data/.scifact_cache")
DEFAULT_CORPUS_DIR = Path("data/demo_corpus")
MAX_CHUNK_CHARS = 480  # pipeline.chunking splits at 500; margin -> exactly one chunk

# doc_id -> targeting metadata. Slugs are fixed by the spec (section 5).
TARGETS = {
    2052720: {
        "slug": "hpylori_gastric_cancer",
        "query": "Does H. pylori infection increase the risk of gastric cancer?",
    },
    6372244: {
        "slug": "antibiotics_cdiff",
        "query": "Do antibiotics increase susceptibility to Clostridium difficile infection?",
    },
    14803797: {
        "slug": "tmao_redmeat",
        "query": "Does dietary L-carnitine from red meat raise TMAO and accelerate atherosclerosis?",
    },
}

# manifest chunk_id -> poison file metadata.
POISON = {
    "poison_hpylori_protective": {
        "filename": "FABRICATED_MISINFO_hpylori_protective.txt",
        "target_doc_id": 2052720,
    },
    "poison_antibiotics_cdiff_protective": {
        "filename": "FABRICATED_MISINFO_antibiotics_cdiff_protective.txt",
        "target_doc_id": 6372244,
    },
    "poison_tmao_redmeat_neutral": {
        "filename": "FABRICATED_MISINFO_tmao_redmeat_neutral.txt",
        "target_doc_id": 14803797,
    },
}


def abstract_text(entry: dict) -> str:
    return " ".join(s.strip() for s in entry["abstract"]).strip()


def is_single_chunk_sized(text: str) -> bool:
    return len(text) <= MAX_CHUNK_CHARS


def slugify(title: str, max_words: int = 5) -> str:
    words = re.findall(r"[a-z0-9]+", title.lower())
    return "_".join(words[:max_words]) or "untitled"


def fit_to_single_chunk(text: str) -> str:
    sentences = [s.strip() for s in text.split(". ") if s.strip()]
    if not sentences:
        return text.strip()
    out = sentences[0].rstrip(".") + "."
    for sentence in sentences[1:]:
        candidate = out + " " + sentence.rstrip(".") + "."
        if not is_single_chunk_sized(candidate):
            break
        out = candidate
    return out


def eligible(claim: dict, corpus: dict[int, dict]) -> bool:
    cited = claim.get("cited_doc_ids", [])
    if len(cited) != 1:
        return False
    evidence = claim.get("evidence", {})
    if not evidence:
        return False
    labels = {e["label"] for sents in evidence.values() for e in sents}
    if labels != {"SUPPORT"}:
        return False
    entry = corpus.get(cited[0])
    if entry is None:
        return False
    if not (4 <= len(entry["abstract"]) <= 8):
        return False
    # Real SciFact abstracts run 600-2000 chars, so no full abstract is
    # single-chunk-sized; the chunk we actually write is the leading
    # sentences that fit. Gate on that fitted text (also rejects docs
    # whose very first sentence already exceeds MAX_CHUNK_CHARS).
    return is_single_chunk_sized(fit_to_single_chunk(abstract_text(entry)))


def select_filler_claims(
    claims: list[dict],
    corpus: dict[int, dict],
    exclude_doc_ids: set[int],
    n: int = 47,
) -> list[dict]:
    best_claim_id: dict[int, int] = {}
    for claim in claims:
        if not eligible(claim, corpus):
            continue
        doc_id = claim["cited_doc_ids"][0]
        if doc_id in exclude_doc_ids:
            continue
        if doc_id not in best_claim_id or claim["id"] < best_claim_id[doc_id]:
            best_claim_id[doc_id] = claim["id"]
    picked = []
    for doc_id in best_claim_id:
        entry = corpus[doc_id]
        slug = slugify(entry["title"])
        picked.append(
            {
                "doc_id": doc_id,
                "title": entry["title"],
                "slug": slug,
                "text": fit_to_single_chunk(abstract_text(entry)),
            }
        )
    picked.sort(key=lambda p: f"scifact_{p['doc_id']}_{p['slug']}")
    return picked[:n]


def build_manifest(filler: list[dict]) -> dict:
    chunks: list[dict] = []
    for f in filler:
        chunk_id = f"scifact_{f['doc_id']}_{f['slug']}"
        chunks.append(
            {
                "filename": f"{chunk_id}.txt",
                "chunk_id": chunk_id,
                "source_doc_id": str(f["doc_id"]),
                "is_poisoned": False,
                "publisher_key": "PUBLISHER_1_KEY",
                "target_reputation": 0,
                "target_queries": [],
                "ground_truth_chunk_id": None,
            }
        )
    for doc_id, meta in TARGETS.items():
        chunk_id = f"scifact_{doc_id}_{meta['slug']}"
        chunks.append(
            {
                "filename": f"{chunk_id}.txt",
                "chunk_id": chunk_id,
                "source_doc_id": str(doc_id),
                "is_poisoned": False,
                "publisher_key": "PUBLISHER_2_KEY",
                "target_reputation": 10,
                "target_queries": [meta["query"]],
                "ground_truth_chunk_id": chunk_id,
            }
        )
    for chunk_id, pmeta in POISON.items():
        target = TARGETS[pmeta["target_doc_id"]]
        gt_chunk_id = f"scifact_{pmeta['target_doc_id']}_{target['slug']}"
        chunks.append(
            {
                "filename": pmeta["filename"],
                "chunk_id": chunk_id,
                "source_doc_id": None,
                "is_poisoned": True,
                "publisher_key": "PUBLISHER_5_KEY",
                "target_reputation": -50,
                "target_queries": [target["query"]],
                "ground_truth_chunk_id": gt_chunk_id,
            }
        )
    return {"chunks": chunks}


def _download_archive() -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    archive = CACHE_DIR / "scifact-data.tar.gz"
    if not archive.exists():
        print(f"[build] downloading {SCIFACT_URL}")
        with urllib.request.urlopen(SCIFACT_URL, timeout=60) as resp:  # noqa: S310 (fixed https URL)
            archive.write_bytes(resp.read())
    return archive


def _load_jsonl_from_archive(archive: Path, member_suffix: str) -> list[dict]:
    with tarfile.open(archive, "r:gz") as tar:
        member = next(m for m in tar.getmembers() if m.name.endswith(member_suffix))
        raw = tar.extractfile(member).read().decode("utf-8")
    return [json.loads(line) for line in io.StringIO(raw) if line.strip()]


def _load_scifact(archive: Path) -> tuple[list[dict], dict[int, dict]]:
    claims = _load_jsonl_from_archive(archive, "claims_train.jsonl")
    claims += _load_jsonl_from_archive(archive, "claims_dev.jsonl")
    corpus_rows = _load_jsonl_from_archive(archive, "corpus.jsonl")
    corpus = {row["doc_id"]: row for row in corpus_rows}
    return claims, corpus


def _write_corpus(corpus_dir: Path, filler: list[dict], corpus: dict[int, dict]) -> None:
    corpus_dir.mkdir(parents=True, exist_ok=True)
    # Clear stale legit excerpts from a prior build (e.g. after a filter
    # change) so they cannot linger. Only scifact_*.txt — never the
    # hand-authored FABRICATED_MISINFO_*.txt files or manifest.json.
    for stale in corpus_dir.glob("scifact_*.txt"):
        stale.unlink()
    for f in filler:
        path = corpus_dir / f"scifact_{f['doc_id']}_{f['slug']}.txt"
        path.write_text(f["text"] + "\n", encoding="utf-8")
    for doc_id, meta in TARGETS.items():
        text = fit_to_single_chunk(abstract_text(corpus[doc_id]))
        path = corpus_dir / f"scifact_{doc_id}_{meta['slug']}.txt"
        path.write_text(text + "\n", encoding="utf-8")
    manifest = build_manifest(filler)
    (corpus_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Milestone 9 SciFact demo corpus.")
    parser.add_argument("--corpus-dir", default=str(DEFAULT_CORPUS_DIR))
    args = parser.parse_args()

    archive = _download_archive()
    claims, corpus = _load_scifact(archive)
    filler = select_filler_claims(claims, corpus, exclude_doc_ids=set(TARGETS))
    if len(filler) != 47:
        sys.exit(f"[build] expected 47 filler claims, got {len(filler)} — widen the filter")

    corpus_dir = Path(args.corpus_dir)
    _write_corpus(corpus_dir, filler, corpus)
    print(f"[build] wrote {len(filler)} filler + 3 targeted legit + manifest to {corpus_dir}")
    print(f"[build] poison .txt files still to be authored: {[p['filename'] for p in POISON.values()]}")


if __name__ == "__main__":
    main()
