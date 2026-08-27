"""Fast offline invariants for the Milestone 9 targeted corpus files.

No live services — only the local embedding model (downloaded on first
use by pipeline.embeddings, same as the rest of the pipeline tests).
"""

import json
from pathlib import Path

import pytest

from pipeline.embeddings import cosine_similarity, embed
from pipeline.ingest_prep import ingest_prep

CORPUS_DIR = Path(__file__).resolve().parent.parent / "data" / "demo_corpus"


def _manifest_chunks():
    return json.loads((CORPUS_DIR / "manifest.json").read_text())["chunks"]


def _by_chunk_id():
    return {c["chunk_id"]: c for c in _manifest_chunks()}


def _targeted_filenames():
    chunks = _by_chunk_id()
    names = []
    for c in _manifest_chunks():
        if c["target_queries"]:  # 3 poison + 3 targeted legit
            names.append(c["filename"])
    return names


@pytest.mark.parametrize("filename", sorted(set(_targeted_filenames())))
def test_targeted_file_is_exactly_one_chunk(filename):
    text = (CORPUS_DIR / filename).read_text(encoding="utf-8")
    records = ingest_prep(text)
    assert len(records) == 1, f"{filename} split into {len(records)} chunks"


def _sim(query: str, text: str) -> float:
    q_vec, t_vec = embed([query, text])
    return cosine_similarity(q_vec, t_vec)


@pytest.mark.parametrize("poison_chunk_id", [
    "poison_tmao_redmeat_neutral",
    "poison_antibiotics_cdiff_protective",
])
def test_poison_beats_or_ties_legit_on_raw_similarity(poison_chunk_id):
    chunks = _by_chunk_id()
    poison = chunks[poison_chunk_id]
    legit = chunks[poison["ground_truth_chunk_id"]]
    query = poison["target_queries"][0]
    poison_sim = _sim(query, (CORPUS_DIR / poison["filename"]).read_text())
    legit_sim = _sim(query, (CORPUS_DIR / legit["filename"]).read_text())
    assert poison_sim >= legit_sim, (
        f"{poison_chunk_id}: poison {poison_sim:.4f} < legit {legit_sim:.4f} — "
        "reword the query in scripts/build_demo_corpus.py TARGETS (not the poison text)"
    )


def test_hpylori_poison_similarity_direction_is_recorded():
    # Spec section 2a measured legit_sim > poison_sim on the FULL SciFact
    # abstract. In this corpus the legit file is a truncated leading-sentence
    # excerpt (OBJECTIVE + DESIGN only, no results), so its raw cosine
    # similarity to the query drops below the poison's and the direction
    # flips to poison_sim >= legit_sim. This is acceptable: the H. pylori
    # case is NOT a Task 6 retrieval-regression assertion — it exists only to
    # demonstrate the reputation layer — so its similarity direction blocks
    # nothing. This test pins the direction that actually holds against the
    # corpus files Task 6 will retrieve.
    chunks = _by_chunk_id()
    poison = chunks["poison_hpylori_protective"]
    legit = chunks[poison["ground_truth_chunk_id"]]
    query = poison["target_queries"][0]
    poison_sim = _sim(query, (CORPUS_DIR / poison["filename"]).read_text())
    legit_sim = _sim(query, (CORPUS_DIR / legit["filename"]).read_text())
    assert poison_sim >= legit_sim
