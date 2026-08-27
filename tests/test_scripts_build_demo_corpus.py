import json as _json

import pytest

from scripts import build_demo_corpus
from scripts.build_demo_corpus import (
    POISON,
    TARGETS,
    abstract_text,
    build_manifest,
    eligible,
    fit_to_single_chunk,
    is_single_chunk_sized,
    select_filler_claims,
    slugify,
)

_CORPUS = {
    100: {"doc_id": 100, "title": "Effects of Vitamin D on Bone Density", "abstract": [
        "Vitamin D supplementation was studied in a randomized trial.",
        "Participants received either vitamin D or placebo for one year.",
        "Bone mineral density increased in the treatment arm.",
        "No serious adverse events were reported.",
    ], "structured": False},
    200: {"doc_id": 200, "title": "Long Review Abstract", "abstract": ["S." * 300]},  # too long
    300: {"doc_id": 300, "title": "Too Short", "abstract": ["One.", "Two.", "Three."]},  # <4 sentences
}


def _claim(cid, doc_id, label="SUPPORT", cited=None):
    cited = [doc_id] if cited is None else cited
    return {"id": cid, "claim": "c", "cited_doc_ids": cited,
            "evidence": {str(doc_id): [{"sentences": [0], "label": label}]}}


def test_abstract_text_joins_and_strips():
    assert abstract_text(_CORPUS[100]).startswith("Vitamin D supplementation")
    assert "  " not in abstract_text(_CORPUS[100])


def test_is_single_chunk_sized_boundary():
    assert is_single_chunk_sized("x" * 480)
    assert not is_single_chunk_sized("x" * 481)


def test_slugify_takes_leading_alnum_words():
    assert slugify("Effects of Vitamin D on Bone Density!") == "effects_of_vitamin_d_on"
    assert slugify("***") == "untitled"


def test_fit_to_single_chunk_truncates_to_leading_sentences():
    long = ". ".join(f"Sentence number {i} has some words" for i in range(40)) + "."
    fitted = fit_to_single_chunk(long)
    assert is_single_chunk_sized(fitted)
    assert fitted.startswith("Sentence number 0")


def test_fit_to_single_chunk_keeps_at_least_first_sentence():
    one = "A single very long sentence " + "word " * 200
    assert fit_to_single_chunk(one).startswith("A single very long sentence")


def test_eligible_accepts_supported_single_cite_short_abstract():
    assert eligible(_claim(1, 100), _CORPUS)


def test_eligible_rejects_multi_cite():
    assert not eligible(_claim(1, 100, cited=[100, 999]), _CORPUS)


def test_eligible_rejects_contradict_label():
    assert not eligible(_claim(1, 100, label="CONTRADICT"), _CORPUS)


def test_eligible_rejects_long_or_short_abstract():
    assert not eligible(_claim(1, 200), _CORPUS)
    assert not eligible(_claim(1, 300), _CORPUS)


def test_select_filler_dedupes_by_doc_id_and_excludes_targets():
    claims = [_claim(5, 100), _claim(2, 100), _claim(9, 100)]
    picked = select_filler_claims(claims, _CORPUS, exclude_doc_ids={2052720}, n=10)
    assert [p["doc_id"] for p in picked] == [100]  # deduped
    assert picked[0]["slug"] == "effects_of_vitamin_d_on"


def test_select_filler_excludes_the_three_poison_targets():
    targets = set(TARGETS)
    claims = [_claim(1, t) for t in targets]  # only target docs on offer
    assert select_filler_claims(claims, {t: {"doc_id": t, "title": "x", "abstract": ["a."] * 5} for t in targets},
                                exclude_doc_ids=targets, n=5) == []


def test_build_manifest_has_53_entries_with_correct_role_counts():
    filler = [{"doc_id": i, "title": f"Doc {i}", "slug": f"doc_{i}", "text": "t"} for i in range(47)]
    manifest = build_manifest(filler)
    chunks = manifest["chunks"]
    assert len(chunks) == 53
    assert sum(c["is_poisoned"] for c in chunks) == 3
    assert sum(c["target_reputation"] == 10 for c in chunks) == 3
    assert sum(c["target_reputation"] == -50 for c in chunks) == 3
    assert sum(c["target_reputation"] == 0 for c in chunks) == 47


def test_build_manifest_poison_entries_point_at_ground_truth():
    manifest = build_manifest([])
    poison = [c for c in manifest["chunks"] if c["is_poisoned"]]
    for p in poison:
        assert p["filename"].startswith("FABRICATED_MISINFO_")
        assert p["publisher_key"] == "PUBLISHER_5_KEY"
        assert p["ground_truth_chunk_id"].startswith("scifact_")
        assert p["target_queries"] and p["target_queries"][0].endswith("?")


def test_targets_and_poison_constants_are_consistent():
    assert set(TARGETS) == {2052720, 6372244, 14803797}
    assert {p["target_doc_id"] for p in POISON.values()} == set(TARGETS)


def test_build_manifest_sets_source_doc_id():
    filler = [
        {"doc_id": 111, "title": "Doc 111", "slug": "doc_111", "text": "t"},
        {"doc_id": 222, "title": "Doc 222", "slug": "doc_222", "text": "t"},
    ]
    chunks = build_manifest(filler)["chunks"]
    by_id = {c["chunk_id"]: c for c in chunks}

    # filler + targeted legit -> stringified SciFact doc id, taken from chunk_id
    for c in chunks:
        if c["chunk_id"].startswith("scifact_"):
            expected = c["chunk_id"].split("_")[1]
            assert c["source_doc_id"] == expected
            assert c["source_doc_id"].isdigit()

    # poison -> None
    for c in chunks:
        if c["is_poisoned"]:
            assert c["source_doc_id"] is None

    # key order: source_doc_id immediately follows chunk_id
    sample = by_id["scifact_111_doc_111"]
    keys = list(sample.keys())
    assert keys[keys.index("chunk_id") + 1] == "source_doc_id"


@pytest.mark.slow
def test_full_build_is_deterministic_and_well_formed(tmp_path, monkeypatch):
    """Requires network access on first run to fetch the SciFact archive
    (cached under data/.scifact_cache/ thereafter). No live Hardhat/IPFS needed.
    """
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    monkeypatch.setattr("sys.argv", ["build_demo_corpus.py", "--corpus-dir", str(out_a)])
    build_demo_corpus.main()
    monkeypatch.setattr("sys.argv", ["build_demo_corpus.py", "--corpus-dir", str(out_b)])
    build_demo_corpus.main()

    manifest_a = (out_a / "manifest.json").read_text()
    assert manifest_a == (out_b / "manifest.json").read_text()  # deterministic

    chunks = _json.loads(manifest_a)["chunks"]
    assert len(chunks) == 53
    legit_txts = sorted(p.name for p in out_a.glob("scifact_*.txt"))
    assert len(legit_txts) == 50
    # Every non-poison manifest entry has a matching file on disk.
    for c in chunks:
        if not c["is_poisoned"]:
            assert (out_a / c["filename"]).exists()
    # The 3 targeted legit files are present and single-chunk-sized.
    for doc_id in (2052720, 6372244, 14803797):
        matches = list(out_a.glob(f"scifact_{doc_id}_*.txt"))
        assert len(matches) == 1
        assert len(matches[0].read_text().strip()) <= 480
