"""Unit tests for the idempotent reputation-seeding logic. All chain/IPFS
access is mocked; no live services.
"""

import json
from pathlib import Path
from unittest.mock import call, patch

import pytest

from scripts.seed_reputation import SeedError, seed


def _write_manifest(corpus_dir: Path, chunks: list[dict]) -> None:
    corpus_dir.mkdir(parents=True, exist_ok=True)
    for c in chunks:
        (corpus_dir / c["filename"]).write_text("dummy text\n", encoding="utf-8")
    (corpus_dir / "manifest.json").write_text(json.dumps({"chunks": chunks}))


_POISON = {
    "filename": "FABRICATED_MISINFO_x.txt", "chunk_id": "poison_x", "is_poisoned": True,
    "publisher_key": "PUBLISHER_5_KEY", "target_reputation": -50,
    "target_queries": ["q?"], "ground_truth_chunk_id": "scifact_1_x",
}
_LEGIT_TARGET = {
    "filename": "scifact_1_x.txt", "chunk_id": "scifact_1_x", "is_poisoned": False,
    "publisher_key": "PUBLISHER_2_KEY", "target_reputation": 10,
    "target_queries": ["q?"], "ground_truth_chunk_id": "scifact_1_x",
}
_FILLER = {
    "filename": "scifact_2_y.txt", "chunk_id": "scifact_2_y", "is_poisoned": False,
    "publisher_key": "PUBLISHER_1_KEY", "target_reputation": 0,
    "target_queries": [], "ground_truth_chunk_id": None,
}


def _run(tmp_path, chunks, reputations_by_hash, hash_by_filename):
    _write_manifest(tmp_path, chunks)

    def fake_ingest_prep(text):
        return [{"hash": "unused", "chunk_id": 0, "text": text, "embedding": []}]

    def fake_resolve(text_path):
        return hash_by_filename[Path(text_path).name]

    def fake_get_document(doc_hash):
        return {"reputation_score": reputations_by_hash[doc_hash], "is_registered": True,
                "publisher": "0x", "content_hash": "0x", "ipfs_cid": "cid", "timestamp": 0}

    with (
        patch("scripts.seed_reputation.run_ingest", return_value=0) as m_ingest,
        patch("scripts.seed_reputation.resolve_doc_hash", side_effect=fake_resolve),
        patch("scripts.seed_reputation.get_document", side_effect=fake_get_document),
        patch("scripts.seed_reputation.update_reputation") as m_update,
    ):
        seed(corpus_dir=tmp_path)
    return m_ingest, m_update


def test_seed_applies_delta_target_minus_current_for_fresh_poison(tmp_path):
    _, m_update = _run(tmp_path, [_POISON], {"h_poison": 0}, {"FABRICATED_MISINFO_x.txt": "h_poison"})
    m_update.assert_called_once_with("h_poison", -50)


def test_seed_skips_chunk_already_at_target(tmp_path):
    _, m_update = _run(tmp_path, [_POISON], {"h_poison": -50}, {"FABRICATED_MISINFO_x.txt": "h_poison"})
    m_update.assert_not_called()


def test_seed_filler_at_zero_default_is_a_noop(tmp_path):
    _, m_update = _run(tmp_path, [_FILLER], {"h_filler": 0}, {"scifact_2_y.txt": "h_filler"})
    m_update.assert_not_called()


def test_seed_resumes_after_partial_run(tmp_path):
    chunks = [_POISON, _LEGIT_TARGET]
    _, m_update = _run(
        tmp_path, chunks,
        {"h_poison": -50, "h_legit": 0},  # poison already done, legit not yet
        {"FABRICATED_MISINFO_x.txt": "h_poison", "scifact_1_x.txt": "h_legit"},
    )
    m_update.assert_called_once_with("h_legit", 10)


def test_seed_raises_if_chunk_unregistered_after_ingest(tmp_path):
    _write_manifest(tmp_path, [_POISON])
    with (
        patch("scripts.seed_reputation.run_ingest", return_value=0),
        patch("scripts.seed_reputation.resolve_doc_hash", return_value="h_poison"),
        patch("scripts.seed_reputation.get_document", return_value={
            "reputation_score": 0, "is_registered": False, "publisher": "0x",
            "content_hash": "0x", "ipfs_cid": "", "timestamp": 0}),
        patch("scripts.seed_reputation.update_reputation") as m_update,
    ):
        with pytest.raises(SeedError):
            seed(corpus_dir=tmp_path)
    m_update.assert_not_called()


def test_seed_raises_if_ingest_returns_nonzero(tmp_path):
    _write_manifest(tmp_path, [_POISON])
    with (
        patch("scripts.seed_reputation.run_ingest", return_value=1),
        patch("scripts.seed_reputation.update_reputation"),
    ):
        with pytest.raises(SeedError):
            seed(corpus_dir=tmp_path)
