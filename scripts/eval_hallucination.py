"""Milestone 10 qualitative hallucination comparison.

For each of the three Milestone 9 target queries, generate a grounded
answer under Standard RAG and under trust-weighted rerank (product-default
top-3 context each) and render eval/hallucination_qualitative.md with the
two answers side by side.

Needs a live Hardhat node + IPFS daemon + a seeded Chroma store AND a
running Ollama with gemma4:12b pulled (see README).

    set -a; . ./.env; set +a
    uv run python scripts/eval_hallucination.py   # writes eval/hallucination_qualitative.md
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.evaluate import EVAL_DIR, gather_cases, load_manifest_chunks

CHANGE_COMMENTARY = (
    "**The generated answer changes between modes.** Standard RAG's answer is "
    "shaped by the fabricated chunk that won raw-similarity retrieval; DeRAG's "
    "answer is shaped by the real abstract that the on-chain reputation signal "
    "promoted to rank 1. Compare the two blockquotes above."
)

STABLE_COMMENTARY = (
    "**The generated answer stays correct under both modes.** Raw similarity "
    "already ranks the legitimate chunk first, so trust-weighting does not "
    "distort a case it does not need to touch — exactly the property you want "
    "confirmed before trusting the mechanism on cases you cannot hand-verify."
)

_COMMENTARY_BY_LABEL = {
    "TMAO / red meat": CHANGE_COMMENTARY,
    "H. pylori / gastric cancer": CHANGE_COMMENTARY,
    "antibiotics / C. difficile": STABLE_COMMENTARY,
}

# Optional per-case caveat, printed after the templated commentary. Keyed by
# label. Only cases whose generation-level behaviour needs qualification appear
# here; a case absent from this map has no caveat.
_CASE_NUANCE = {
    "H. pylori / gastric cancer": (
        "_Caveat — read the DeRAG answer above carefully._ Retrieval flips "
        "cleanly (DeRAG's rank-1 context is the real abstract), but the "
        "generated DeRAG answer does not reach a clean affirmative the way "
        "TMAO's does: it tends to hedge, presenting the fabricated \"protective\" "
        "claim as one side. Two compounding, on-record reasons: (1) the real "
        "H. pylori chunk is truncated to its objective/design sentence with no "
        "results (the M9 leading-sentence truncation), so it carries the "
        "question but not the finding; (2) after rerank the -50 poison chunk "
        "still occupies context slot #2 of the 3-item window -- trust-weighting "
        "fixes rank-1, not membership. This is the generation-level face of "
        "H. pylori being the thinner-margin, more drift-exposed case; Standard "
        "RAG's H. pylori answer is still an unambiguous, confident hallucination."
    ),
    "TMAO / red meat": (
        "_Note._ Standard RAG's TMAO answer is itself a both-sides hedge (it "
        "actually surfaces the true finding first); DeRAG's is an unhedged, "
        "correct affirmative. The generation-level contrast here is "
        "muddled-hedge vs clean-correct, not confident-hallucination vs "
        "correction."
    ),
}


def retrieve(query: str, mode: str) -> list[dict]:
    from retrieval.rerank import trust_weighted_rerank
    from retrieval.standard_rag import standard_rag_answer_context

    if mode == "standard":
        return standard_rag_answer_context(query, k=3)
    if mode == "derag":
        return trust_weighted_rerank(query, k_retrieve=10, k_return=3)
    raise ValueError(f"unknown mode {mode!r}")


def run_case(query: str, mode: str) -> dict:
    from generation.llm_client import build_prompt, generate_streaming

    docs = retrieve(query, mode)
    answer = "".join(generate_streaming(build_prompt(query, docs)))
    return {
        "mode": mode,
        "answer": answer.strip(),
        "sources": [d["metadata"].get("source_file", d["chroma_id"]) for d in docs],
    }


def _sources_md(sources: list[str]) -> str:
    return "\n".join(f"  {i}. `{s}`" for i, s in enumerate(sources, start=1))


def render_md(cases: list[dict]) -> str:
    parts = [
        "# DeRAG Qualitative Hallucination Comparison (Milestone 10)",
        "",
        "Standard RAG vs DeRAG generated answers for all three Milestone 9 target "
        "queries, each grounded in that mode's product-default top-3 retrieval. "
        "Answers captured verbatim from `gemma4:12b` at `temperature=0.1`.",
        "",
        "For **TMAO** and **H. pylori** the answer changes with the retrieval "
        "winner; for **antibiotics** it stays correct under both modes — the "
        "trust-weighting does not distort a case it does not need to touch.",
        "",
    ]
    for case in cases:
        parts += [
            f"## {case['label']}",
            "",
            f"**Query:** {case['query']}",
            "",
            "### Standard RAG",
            "",
            f"_Context sources:_\n{_sources_md(case['standard']['sources'])}",
            "",
            "> " + case["standard"]["answer"].replace("\n", "\n> "),
            "",
            "### DeRAG (trust-weighted rerank)",
            "",
            f"_Context sources:_\n{_sources_md(case['derag']['sources'])}",
            "",
            "> " + case["derag"]["answer"].replace("\n", "\n> "),
            "",
            _COMMENTARY_BY_LABEL[case["label"]],
            "",
        ]
        if case["label"] in _CASE_NUANCE:
            parts += ["", _CASE_NUANCE[case["label"]]]
    return "\n".join(parts).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="DeRAG qualitative hallucination comparison.")
    parser.add_argument("--out", default=str(EVAL_DIR / "hallucination_qualitative.md"))
    args = parser.parse_args()

    cases = []
    for case in gather_cases(load_manifest_chunks()):
        cases.append(
            {
                "label": case["label"],
                "query": case["query"],
                "standard": run_case(case["query"], "standard"),
                "derag": run_case(case["query"], "derag"),
            }
        )
    md = render_md(cases)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    print(md)
    print(f"\n[eval_hallucination] wrote {out_path}")


if __name__ == "__main__":
    main()
