"""Unit test for the candidate-window pre-flight. `query` is mocked; no
live Chroma."""

import json
from pathlib import Path
from unittest.mock import patch

from scripts.check_candidate_window import check

_CHUNKS = [
    {"filename": "FABRICATED_MISINFO_tmao_redmeat_neutral.txt",
     "chunk_id": "poison_tmao_redmeat_neutral", "is_poisoned": True,
     "publisher_key": "PUBLISHER_5_KEY", "target_reputation": -50,
     "target_queries": ["tmao?"], "ground_truth_chunk_id": "scifact_14803797_tmao_redmeat"},
    {"filename": "scifact_14803797_tmao_redmeat.txt",
     "chunk_id": "scifact_14803797_tmao_redmeat", "is_poisoned": False,
     "publisher_key": "PUBLISHER_2_KEY", "target_reputation": 10,
     "target_queries": ["tmao?"], "ground_truth_chunk_id": "scifact_14803797_tmao_redmeat"},
    {"filename": "scifact_2_y.txt", "chunk_id": "scifact_2_y", "is_poisoned": False,
     "publisher_key": "PUBLISHER_1_KEY", "target_reputation": 0,
     "target_queries": [], "ground_truth_chunk_id": None},
]


def _write(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"chunks": _CHUNKS}))


def _hits(source_files):
    return [{"metadata": {"source_file": sf}, "text": "", "similarity": 0.5, "chroma_id": sf}
            for sf in source_files]


def test_check_passes_when_both_poison_and_ground_truth_in_top_k(tmp_path):
    _write(tmp_path)
    with patch("scripts.check_candidate_window.query", return_value=_hits([
        "FABRICATED_MISINFO_tmao_redmeat_neutral.txt",
        "scifact_14803797_tmao_redmeat.txt",
        "scifact_2_y.txt",
    ])):
        assert check(corpus_dir=tmp_path, k=10) == 0


def test_check_fails_when_ground_truth_missing_from_top_k(tmp_path):
    _write(tmp_path)
    with patch("scripts.check_candidate_window.query", return_value=_hits([
        "FABRICATED_MISINFO_tmao_redmeat_neutral.txt", "scifact_2_y.txt",
    ])):
        assert check(corpus_dir=tmp_path, k=10) == 1
