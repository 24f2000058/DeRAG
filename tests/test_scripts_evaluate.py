import json
from pathlib import Path

import pytest

from scripts.evaluate import (
    build_ground_truth,
    evaluate_query,
    gather_cases,
    hits_to_doc_ids,
    mean_reciprocal_rank,
    precision_at_k,
    reciprocal_rank,
)


def test_reciprocal_rank_first_position():
    assert reciprocal_rank(["7", "1", "2"], "7") == 1.0


def test_reciprocal_rank_third_position():
    assert reciprocal_rank(["1", "2", "7"], "7") == pytest.approx(1 / 3)


def test_reciprocal_rank_absent_is_zero():
    assert reciprocal_rank(["1", "2", None], "7") == 0.0


def test_precision_at_k_single_relevant_in_window():
    assert precision_at_k(["1", "7", "2"], "7", 1) == 0.0
    assert precision_at_k(["7", "1", "2"], "7", 1) == 1.0
    assert precision_at_k(["1", "7", "2"], "7", 3) == pytest.approx(1 / 3)


def test_precision_at_k_relevant_outside_window_is_zero():
    assert precision_at_k(["1", "2", "3", "7"], "7", 3) == 0.0


def test_mean_reciprocal_rank():
    assert mean_reciprocal_rank([1.0, 0.5, 0.0]) == pytest.approx(0.5)
    assert mean_reciprocal_rank([]) == 0.0


def test_evaluate_query_bundles_metrics():
    out = evaluate_query(["1", "2", "7", "9"], "7")
    assert out == {"rr": pytest.approx(1 / 3), "p_at_1": 0.0, "p_at_3": pytest.approx(1 / 3)}


_MANIFEST_SLICE = {
    "chunks": [
        {"filename": "scifact_14803797_tmao_redmeat.txt",
         "chunk_id": "scifact_14803797_tmao_redmeat", "source_doc_id": "14803797",
         "is_poisoned": False, "publisher_key": "PUBLISHER_2_KEY", "target_reputation": 10,
         "target_queries": ["Q-TMAO"], "ground_truth_chunk_id": "scifact_14803797_tmao_redmeat"},
        {"filename": "FABRICATED_MISINFO_tmao_redmeat_neutral.txt",
         "chunk_id": "poison_tmao_redmeat_neutral", "source_doc_id": None,
         "is_poisoned": True, "publisher_key": "PUBLISHER_5_KEY", "target_reputation": -50,
         "target_queries": ["Q-TMAO"], "ground_truth_chunk_id": "scifact_14803797_tmao_redmeat"},
        {"filename": "scifact_999_filler.txt", "chunk_id": "scifact_999_filler",
         "source_doc_id": "999", "is_poisoned": False, "publisher_key": "PUBLISHER_1_KEY",
         "target_reputation": 0, "target_queries": [], "ground_truth_chunk_id": None},
    ]
}


def test_build_ground_truth_resolves_to_legit_counterpart_source_doc_id():
    gt = build_ground_truth(_MANIFEST_SLICE["chunks"])
    assert gt == {"Q-TMAO": "14803797"}


def test_source_doc_id_by_filename_maps_poison_to_none():
    from scripts.evaluate import source_doc_id_by_filename
    sdi = source_doc_id_by_filename(_MANIFEST_SLICE["chunks"])
    assert sdi["FABRICATED_MISINFO_tmao_redmeat_neutral.txt"] is None
    assert sdi["scifact_14803797_tmao_redmeat.txt"] == "14803797"


def test_hits_to_doc_ids_join():
    from scripts.evaluate import source_doc_id_by_filename
    sdi = source_doc_id_by_filename(_MANIFEST_SLICE["chunks"])
    hits = [
        {"metadata": {"source_file": "FABRICATED_MISINFO_tmao_redmeat_neutral.txt"}},
        {"metadata": {"source_file": "scifact_14803797_tmao_redmeat.txt"}},
    ]
    assert hits_to_doc_ids(hits, sdi) == [None, "14803797"]


def test_gather_cases_uses_real_committed_manifest():
    from scripts.evaluate import load_manifest_chunks
    cases = gather_cases(load_manifest_chunks())
    assert {c["label"] for c in cases} == {
        "TMAO / red meat", "H. pylori / gastric cancer", "antibiotics / C. difficile"
    }
    for c in cases:
        assert c["query"].endswith("?")
        assert c["gt_doc_id"].isdigit()
        assert c["poison_file"].startswith("FABRICATED_MISINFO_")
        assert c["legit_file"].startswith("scifact_")
