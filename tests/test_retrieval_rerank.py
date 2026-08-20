import math

import pytest
from unittest.mock import patch

from retrieval.rerank import (
    FLOOR_PENALTY,
    SCALE,
    get_candidate_reputation,
    normalize_reputation,
    hybrid_score,
    trust_weighted_rerank,
)


def test_get_candidate_reputation_returns_exists_and_reputation_tuple():
    with patch(
        "retrieval.rerank.verify_integrity",
        return_value={"exists": True, "reputation": 7, "publisher": "0xabc"},
    ):
        result = get_candidate_reputation("somehash")

    assert result == (True, 7)


def test_get_candidate_reputation_passes_through_exists_false():
    with patch(
        "retrieval.rerank.verify_integrity",
        return_value={"exists": False, "reputation": 0, "publisher": "0x" + "0" * 40},
    ):
        result = get_candidate_reputation("unregistered")

    assert result == (False, 0)


def test_normalize_reputation_applies_sigmoid_to_existing_candidates():
    candidates = [
        {"exists": True, "reputation": 0},
        {"exists": True, "reputation": 10},
    ]

    result = normalize_reputation(candidates)

    assert result[0]["normalized_reputation"] == pytest.approx(0.5)
    assert result[1]["normalized_reputation"] == pytest.approx(1 / (1 + math.exp(-1)))


def test_normalize_reputation_preserves_original_keys():
    candidates = [{"exists": True, "reputation": 3, "doc_hash": "abc"}]

    result = normalize_reputation(candidates)

    assert result[0]["doc_hash"] == "abc"
    assert result[0]["reputation"] == 3


def test_normalize_reputation_floors_exists_false_below_batch_minimum():
    candidates = [
        {"exists": True, "reputation": -5},
        {"exists": False, "reputation": 0},
    ]

    result = normalize_reputation(candidates)

    expected_floor = -5 - FLOOR_PENALTY
    assert result[1]["normalized_reputation"] == pytest.approx(
        1 / (1 + math.exp(-expected_floor / SCALE))
    )
    assert result[1]["normalized_reputation"] < result[0]["normalized_reputation"]


def test_normalize_reputation_all_exists_false_falls_back_to_zero_baseline():
    candidates = [
        {"exists": False, "reputation": 0},
        {"exists": False, "reputation": 0},
    ]

    result = normalize_reputation(candidates)

    expected = 1 / (1 + math.exp(-(0 - FLOOR_PENALTY) / SCALE))
    assert result[0]["normalized_reputation"] == pytest.approx(expected)
    assert result[1]["normalized_reputation"] == pytest.approx(expected)


def test_normalize_reputation_identical_reputations_all_candidates_produce_same_value():
    candidates = [
        {"exists": True, "reputation": 5},
        {"exists": True, "reputation": 5},
        {"exists": True, "reputation": 5},
    ]

    result = normalize_reputation(candidates)

    values = {c["normalized_reputation"] for c in result}
    assert len(values) == 1


def test_normalize_reputation_single_candidate_does_not_crash():
    candidates = [{"exists": True, "reputation": 3}]

    result = normalize_reputation(candidates)

    assert len(result) == 1
    assert 0.0 < result[0]["normalized_reputation"] < 1.0


def test_normalize_reputation_extreme_negative_reputation_does_not_overflow():
    candidates = [{"exists": True, "reputation": -100000}]

    result = normalize_reputation(candidates)

    assert result[0]["normalized_reputation"] == pytest.approx(0.0, abs=1e-9)


def test_hybrid_score_default_alpha_weights_similarity_at_seventy_percent():
    score = hybrid_score(similarity=1.0, normalized_reputation=0.0, alpha=0.7)
    assert score == pytest.approx(0.7)


def test_hybrid_score_default_alpha_weights_reputation_at_thirty_percent():
    score = hybrid_score(similarity=0.0, normalized_reputation=1.0, alpha=0.7)
    assert score == pytest.approx(0.3)


def test_hybrid_score_respects_custom_alpha():
    score = hybrid_score(similarity=0.5, normalized_reputation=0.5, alpha=0.3)
    assert score == pytest.approx(0.5)


def test_hybrid_score_uses_default_alpha_when_not_specified():
    score = hybrid_score(similarity=1.0, normalized_reputation=0.0)
    assert score == pytest.approx(0.7)


def _poisoning_scenario_hits():
    return [
        {
            "chroma_id": "poisoned_hash",
            "text": "poisoned chunk with strong keyword overlap",
            "similarity": 0.95,
            "metadata": {"doc_hash": "poisoned_hash", "source_file": "poison.txt"},
        },
        {
            "chroma_id": "factual_hash",
            "text": "factual chunk with slightly lower overlap",
            "similarity": 0.90,
            "metadata": {"doc_hash": "factual_hash", "source_file": "factual.txt"},
        },
    ]


def _poisoning_scenario_reputations(doc_hash):
    reputations = {
        "poisoned_hash": {"exists": True, "reputation": -50, "publisher": "0xabc"},
        "factual_hash": {"exists": True, "reputation": 10, "publisher": "0xdef"},
    }
    return reputations[doc_hash]


def test_trust_weighted_rerank_demotes_high_similarity_negative_reputation_doc():
    # Prove the pure-cosine baseline (Standard RAG) would rank the poisoned
    # doc #1: it has the higher similarity score.
    standard_rag_order = sorted(
        _poisoning_scenario_hits(), key=lambda hit: hit["similarity"], reverse=True
    )
    assert standard_rag_order[0]["metadata"]["doc_hash"] == "poisoned_hash"

    with (
        patch("retrieval.rerank.query", return_value=_poisoning_scenario_hits()),
        patch("retrieval.rerank.verify_integrity", side_effect=_poisoning_scenario_reputations),
    ):
        results = trust_weighted_rerank("test query", k_retrieve=2, k_return=2)

    assert results[0]["metadata"]["doc_hash"] == "factual_hash"
    assert results[1]["metadata"]["doc_hash"] == "poisoned_hash"


def _unregistered_scenario_hits():
    return [
        {
            "chroma_id": "unregistered_hash",
            "text": "highest similarity but never registered on-chain",
            "similarity": 0.95,
            "metadata": {"doc_hash": "unregistered_hash", "source_file": "unregistered.txt"},
        },
        {
            "chroma_id": "neutral_hash",
            "text": "lower similarity but registered with neutral reputation",
            "similarity": 0.80,
            "metadata": {"doc_hash": "neutral_hash", "source_file": "neutral.txt"},
        },
    ]


def _unregistered_scenario_reputations(doc_hash):
    reputations = {
        "unregistered_hash": {"exists": False, "reputation": 0, "publisher": "0x" + "0" * 40},
        "neutral_hash": {"exists": True, "reputation": 0, "publisher": "0xabc"},
    }
    return reputations[doc_hash]


def test_trust_weighted_rerank_ranks_unregistered_candidate_below_neutral_registered_one():
    hits = _unregistered_scenario_hits()
    assert hits[0]["metadata"]["doc_hash"] == "unregistered_hash"
    assert hits[0]["similarity"] > hits[1]["similarity"]

    with (
        patch("retrieval.rerank.query", return_value=hits),
        patch("retrieval.rerank.verify_integrity", side_effect=_unregistered_scenario_reputations),
    ):
        results = trust_weighted_rerank("test query", k_retrieve=2, k_return=2)

    assert results[0]["metadata"]["doc_hash"] == "neutral_hash"
    assert results[1]["metadata"]["doc_hash"] == "unregistered_hash"


def test_trust_weighted_rerank_respects_k_return():
    hits = _poisoning_scenario_hits()

    with (
        patch("retrieval.rerank.query", return_value=hits),
        patch("retrieval.rerank.verify_integrity", side_effect=_poisoning_scenario_reputations),
    ):
        results = trust_weighted_rerank("test query", k_retrieve=2, k_return=1)

    assert len(results) == 1
