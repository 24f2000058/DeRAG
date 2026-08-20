import math

import pytest
from unittest.mock import patch

from retrieval.rerank import FLOOR_PENALTY, SCALE, get_candidate_reputation, normalize_reputation


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
