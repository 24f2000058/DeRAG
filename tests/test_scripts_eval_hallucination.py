from scripts.eval_hallucination import (
    CHANGE_COMMENTARY,
    STABLE_COMMENTARY,
    _CASE_NUANCE,
    render_md,
)

_CASES = [
    {
        "label": "TMAO / red meat", "query": "Q1",
        "standard": {"mode": "standard",
                     "answer": "Red meat L-carnitine does not raise TMAO. [1]",
                     "sources": ["FABRICATED_MISINFO_tmao_redmeat_neutral.txt",
                                 "scifact_14803797_tmao_redmeat.txt", "scifact_x.txt"]},
        "derag": {"mode": "derag",
                  "answer": "Dietary L-carnitine from red meat raises TMAO and "
                            "accelerates atherosclerosis. [1]",
                  "sources": ["scifact_14803797_tmao_redmeat.txt",
                              "FABRICATED_MISINFO_tmao_redmeat_neutral.txt", "scifact_x.txt"]},
    },
    {
        "label": "H. pylori / gastric cancer", "query": "Q2",
        "standard": {"mode": "standard", "answer": "H. pylori is protective. [1]",
                     "sources": ["FABRICATED_MISINFO_hpylori_protective.txt", "a.txt", "b.txt"]},
        "derag": {"mode": "derag", "answer": "H. pylori infection increases gastric cancer risk. [1]",
                  "sources": ["scifact_2052720_hpylori_gastric_cancer.txt", "a.txt", "b.txt"]},
    },
    {
        "label": "antibiotics / C. difficile", "query": "Q3",
        "standard": {"mode": "standard",
                     "answer": "Antibiotics increase susceptibility to C. difficile. [1]",
                     "sources": ["scifact_6372244_antibiotics_cdiff.txt", "a.txt", "b.txt"]},
        "derag": {"mode": "derag",
                  "answer": "Antibiotics increase susceptibility to C. difficile. [1]",
                  "sources": ["scifact_6372244_antibiotics_cdiff.txt", "a.txt", "b.txt"]},
    },
]


def test_render_md_side_by_side_and_commentary():
    md = render_md(_CASES)
    assert "## TMAO / red meat" in md
    assert "## H. pylori / gastric cancer" in md
    assert "## antibiotics / C. difficile" in md
    assert "Standard RAG" in md and "DeRAG" in md
    # feeding sources are shown
    assert "FABRICATED_MISINFO_tmao_redmeat_neutral.txt" in md
    assert "scifact_2052720_hpylori_gastric_cancer.txt" in md
    # the two positives are labelled as the answer CHANGING
    assert md.count(CHANGE_COMMENTARY) == 2
    # the honest negative is labelled as the answer STAYING correct under both modes
    assert md.count(STABLE_COMMENTARY) == 1


def test_commentary_constants_say_the_right_things():
    assert "changes" in CHANGE_COMMENTARY
    assert "absent in the DeRAG column" not in CHANGE_COMMENTARY
    assert "both modes" in STABLE_COMMENTARY and "does not distort" in STABLE_COMMENTARY


def test_render_md_emits_hpylori_caveat():
    md = render_md(_CASES)
    assert md.count("Retrieval flips") == 1
    assert md.count("slot #2") == 1
    assert md.count("drift-exposed") == 1
    assert md.count("muddled-hedge vs clean-correct") == 1
    assert "antibiotics / C. difficile" not in _CASE_NUANCE


def test_run_case_joins_streamed_chunks_and_strips(monkeypatch):
    import scripts.eval_hallucination as eh
    from generation import llm_client

    monkeypatch.setattr(eh, "retrieve", lambda q, m: [
        {"metadata": {"source_file": "scifact_1_a.txt"}, "chroma_id": "c1", "text": "t"},
    ])
    monkeypatch.setattr(llm_client, "generate_streaming", lambda prompt: iter(["  Hello", ", ", "world  "]))

    out = eh.run_case("Q", "standard")
    assert out == {"mode": "standard", "answer": "Hello, world", "sources": ["scifact_1_a.txt"]}
