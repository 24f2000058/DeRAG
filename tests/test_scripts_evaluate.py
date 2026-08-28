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


def test_hits_to_doc_ids_raises_on_unknown_source_file():
    from scripts.evaluate import source_doc_id_by_filename
    sdi = source_doc_id_by_filename(_MANIFEST_SLICE["chunks"])
    import pytest
    with pytest.raises(KeyError, match="not in manifest"):
        hits_to_doc_ids([{"metadata": {"source_file": "not_a_real_file.txt"}}], sdi)


def test_build_ground_truth_agrees_with_gather_cases_on_real_manifest():
    from scripts.evaluate import build_ground_truth, gather_cases, load_manifest_chunks
    chunks = load_manifest_chunks()
    gt = build_ground_truth(chunks)
    for case in gather_cases(chunks):
        assert gt[case["query"]] == case["gt_doc_id"]


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


from scripts.evaluate import (  # noqa: E402
    ALPHA_SANITY_NOTE,
    FRAMING_HEADLINE,
    NARRATIVE_ANTIBIOTICS,
    NARRATIVE_HPYLORI,
    NARRATIVE_TMAO,
    render_results_md,
)

_RESULTS = {
    "alphas": [0.0, 0.3, 0.5, 0.7, 1.0],
    "per_query": [
        {"label": "TMAO / red meat", "query": "Q1", "gt_doc_id": "14803797",
         "poison_file": "FABRICATED_MISINFO_tmao_redmeat_neutral.txt",
         "legit_file": "scifact_14803797_tmao_redmeat.txt",
         "standard": {"rr": 0.5, "p_at_1": 0.0, "p_at_3": 1 / 3},
         "derag": {"rr": 1.0, "p_at_1": 1.0, "p_at_3": 1 / 3},
         "standard_dump": ["FABRICATED_MISINFO_tmao_redmeat_neutral.txt", "scifact_14803797_tmao_redmeat.txt"],
         "derag_dump": ["scifact_14803797_tmao_redmeat.txt", "FABRICATED_MISINFO_tmao_redmeat_neutral.txt"]},
        {"label": "H. pylori / gastric cancer", "query": "Q2", "gt_doc_id": "2052720",
         "poison_file": "FABRICATED_MISINFO_hpylori_protective.txt",
         "legit_file": "scifact_2052720_hpylori_gastric_cancer.txt",
         "standard": {"rr": 0.5, "p_at_1": 0.0, "p_at_3": 1 / 3},
         "derag": {"rr": 1.0, "p_at_1": 1.0, "p_at_3": 1 / 3},
         "standard_dump": ["FABRICATED_MISINFO_hpylori_protective.txt", "scifact_2052720_hpylori_gastric_cancer.txt"],
         "derag_dump": ["scifact_2052720_hpylori_gastric_cancer.txt", "FABRICATED_MISINFO_hpylori_protective.txt"]},
        {"label": "antibiotics / C. difficile", "query": "Q3", "gt_doc_id": "6372244",
         "poison_file": "FABRICATED_MISINFO_antibiotics_cdiff_protective.txt",
         "legit_file": "scifact_6372244_antibiotics_cdiff.txt",
         "standard": {"rr": 1.0, "p_at_1": 1.0, "p_at_3": 1 / 3},
         "derag": {"rr": 1.0, "p_at_1": 1.0, "p_at_3": 1 / 3},
         "standard_dump": ["scifact_6372244_antibiotics_cdiff.txt", "FABRICATED_MISINFO_antibiotics_cdiff_protective.txt"],
         "derag_dump": ["scifact_6372244_antibiotics_cdiff.txt", "FABRICATED_MISINFO_antibiotics_cdiff_protective.txt"]},
    ],
    "sweep": [
        {"alpha": 0.0, "mrr": 0.55, "mean_p_at_1": 0.33},
        {"alpha": 0.3, "mrr": 1.0, "mean_p_at_1": 1.0},
        {"alpha": 0.5, "mrr": 1.0, "mean_p_at_1": 1.0},
        {"alpha": 0.7, "mrr": 1.0, "mean_p_at_1": 1.0},
        {"alpha": 1.0, "mrr": 0.667, "mean_p_at_1": 0.667},
    ],
}


def test_render_results_md_carries_the_framing_prose():
    md = render_results_md(_RESULTS)
    # not just tables — the rhetorical framing must survive templating
    assert FRAMING_HEADLINE in md
    assert NARRATIVE_TMAO in md
    assert NARRATIVE_HPYLORI in md
    assert NARRATIVE_ANTIBIOTICS in md
    assert ALPHA_SANITY_NOTE in md


def test_render_results_md_framing_constants_say_the_right_things():
    # guard the constants themselves so an edit can't hollow them out
    assert "two-positive" in FRAMING_HEADLINE and "one-honest-negative" in FRAMING_HEADLINE
    assert "0.198" in NARRATIVE_TMAO
    assert "0.046" in NARRATIVE_HPYLORI and "robustness" in NARRATIVE_HPYLORI
    assert "no-regression" in NARRATIVE_ANTIBIOTICS
    assert "1.0" in ALPHA_SANITY_NOTE and "similarity" in ALPHA_SANITY_NOTE


def test_render_results_md_has_tables_and_dumps():
    md = render_results_md(_RESULTS)
    assert "| TMAO / red meat |" in md
    assert "| H. pylori / gastric cancer |" in md
    assert "| antibiotics / C. difficile |" in md
    assert "Alpha sweep" in md
    assert "0.7" in md and "shipped default" in md
    # raw ranking dumps present for auditability
    assert "scifact_14803797_tmao_redmeat.txt" in md
    assert "uv run python scripts/evaluate.py" in md


def test_render_results_md_plateau_reading_reacts_to_data():
    wide = render_results_md(_RESULTS)  # 0.3/0.5/0.7 all tie at max MRR
    assert "plateau" in wide.lower()
    knife = dict(_RESULTS)
    knife["sweep"] = [
        {"alpha": 0.0, "mrr": 0.4, "mean_p_at_1": 0.33},
        {"alpha": 0.3, "mrr": 0.5, "mean_p_at_1": 0.33},
        {"alpha": 0.5, "mrr": 0.6, "mean_p_at_1": 0.33},
        {"alpha": 0.7, "mrr": 1.0, "mean_p_at_1": 1.0},
        {"alpha": 1.0, "mrr": 0.6, "mean_p_at_1": 0.33},
    ]
    assert "sensitive" in render_results_md(knife).lower()
