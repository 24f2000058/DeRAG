# Milestone 10 — live-window findings (scratch; consumed by Tasks 7-9, then deleted)

## Antibiotics similarity-pair discrepancy (plan §6) — RESOLVED: harmless rounding

Recomputed live against the CURRENT committed corpus files + current embedding
pipeline (`pipeline.embeddings.embed`, all-MiniLM-L6-v2):

    legit_sim  (scifact_6372244_antibiotics_cdiff.txt)              = 0.7329
    poison_sim (FABRICATED_MISINFO_antibiotics_cdiff_protective.txt) = 0.7092
    margin (legit - poison)                                         = 0.0237

- **DEMO.md** currently says "0.7329 vs 0.7092" — reproduces EXACTLY (4 dp).
- **DECISIONS.md** (M9 measured-similarity bullet) says "poison 0.709 vs
  legit 0.733 (legit wins by 0.024)" — the SAME measurement rounded to 3 dp.
- **tests/test_demo_corpus_fixtures.py** comment says "~0.733 / ~0.709" — same.

**What the drift was:** a per-document display-precision convention, NOT a
stale measurement. DEMO.md quotes similarity figures to 4 dp throughout its
blocks (e.g. TMAO "0.7789"); the DECISIONS.md M9 section quotes them to 3 dp
as its house style ("0.779", "0.802", "0.709"). Both describe one identical
measurement. There is no data drift and no superseded run — DEMO.md's numbers
are current and correct.

**Task 9 fix:** make DEMO.md and DECISIONS.md quote the antibiotics pair at
the same precision. Adopt the 4-dp form everywhere it appears in an
antibiotics context: `legit 0.7329 vs poison 0.7092 (margin 0.0237)`.
No other DEMO.md numbers change.

## DEMO.md scripted-query re-verification (plan §6 Step 6)

- `pytest -m slow tests/test_demo_corpus_regression.py` → **7 passed, 1 xfailed**
  (unchanged from M9). Ran twice this window, both identical.
- `scripts/evaluate.py` re-run after a full chain re-seed produced a
  **byte-identical** `eval/results.md` (`git diff` empty) — retrieval is
  deterministic.
- Direct `demo_rerank` re-runs of the two DEMO.md blocks were interrupted by
  an environment crash (see below) and then superseded by the stable
  evaluate.py + 3x direct-probe verification of the H. pylori / TMAO
  candidate windows and rerank order, which match DEMO.md's captured output
  (poison #1 under Standard RAG; legit #1 after trust-weighted rerank for
  both TMAO and H. pylori). **DEMO.md ranking/similarity blocks: no change
  needed** beyond the antibiotics-precision normalization above.
- **Recheck at Task 9:** confirm DEMO.md still says `7 passed, 1 xfailed`
  and the TMAO/H. pylori rerank score figures (0.6257 / 0.7484 etc.) — the DeRAG hybrid-score figures (0.6257 / 0.7484) AND the raw similarity
  decimals (0.7789 / 0.5806 / 0.8020 / 0.7558) were re-measured by the
  calm-box demo_rerank spot-run below and all match DEMO.md exactly. Low risk (deterministic, regression green), but a
  Task 9 `demo_rerank` spot-run on a calm box would fully close it.

## H. pylori qualitative generation nuance (honest, reportable)

`eval/hallucination_qualitative.md` (committed, clean run, all 6 retrieval-
context sanity checks pass):

- **TMAO** — the generation-level win is on the DeRAG side. Standard RAG's
  answer is itself a both-sides HEDGE ("The provided sources contain
  conflicting information...") that actually surfaces the TRUE claim first
  ([2] = the real abstract) and the fabricated "cardiovascular-neutral"
  claim second — it does not confidently hallucinate. DeRAG's answer is an
  unhedged, correct "Yes, ... produces TMAO and accelerates atherosclerosis
  [1]". So the contrast to report for TMAO is muddled-hedge (Standard RAG)
  vs clean-correct (DeRAG), NOT confident-hallucination vs correction.
- **H. pylori** — retrieval flips cleanly (DeRAG #1 = legit), but the
  *generated* DeRAG answer HEDGES ("sources contain conflicting
  information ... one source describes a study designed to investigate the
  association [1] ... another states protective [2]") rather than giving a
  clean "Yes". Two compounding reasons, both already on record:
  (1) the legit H. pylori excerpt is truncated to its OBJECTIVE/DESIGN
  sentence (no results/conclusion — the same leading-sentence truncation
  that flipped the M9 similarity orderings), so it cannot drive a strong
  affirmative; (2) the -50 poison chunk still sits at #2 in the 3-item
  context window (rerank fixes rank-1, not membership). This is the
  generation-level face of "H. pylori is the thinner, more drift-exposed
  case" — report it honestly in DECISIONS.md M10 and VIVA.md, do NOT
  paper over it. Standard RAG's H. pylori answer is a full, confident
  hallucination ("No ... protective ... odds ratio 0.36"), so the
  Standard-RAG-is-dangerous point stands unambiguously.
- **antibiotics** — identical context and identical correct answer under
  both modes. No-regression check passes cleanly.

## Environment fragility (note for README + DECISIONS)

- Cold `gemma4:12b` on this box: ~60 s model load + partial CPU offload
  (`ollama ps`: 23%/77% CPU/GPU — the 7.9 GB model + KV cache exceeds the
  8 GB RTX 4060). `prompt_eval` ~65 s even for a 20-token prompt.
- The cold-load spike OOM-killed the Hardhat node mid-window; a later IPFS
  daemon crash aborted the first re-seed. Both recovered by restart +
  redeploy (deterministic addr `0x5FbD…aa3`) + full re-seed; evaluate.py
  then reproduced `eval/results.md` byte-identically.
- The FIRST `eval_hallucination.py` run (box still thrashing) produced a
  transiently CORRUPT H. pylori retrieval (legit chunk absent from the
  k=10 window). 3 stable direct probes + a calm re-run confirmed it was
  transient; the committed file is from the clean re-run.
- `generate_blocking` (`generation/llm_client.py`, `timeout=10`) cannot be
  used with this model — `eval_hallucination.run_case` was switched to
  `generate_streaming` (`timeout=(10, None)`), commit db12bf6.
- Retrieval (`trust_weighted_rerank`) needs Chroma + the Hardhat chain
  only — NOT IPFS. IPFS is only needed by `seed_reputation.py`'s ingest.

## UPDATE — DEMO.md demo_rerank blocks fully verified (calm-box spot-run)

`uv run python -m scripts.demo_rerank` re-run for both DEMO.md queries on a
stable box. Output matches DEMO.md's captured "Expected output" blocks
EXACTLY, hybrid scores included:

- TMAO Standard RAG: poison 0.7789 / legit 0.5806 / bile-acid 0.3803.
  TMAO Trust-Weighted: legit #1 (sim 0.5806, score 0.6257) / poison #2
  (0.7789, 0.5473) / bile-acid #3 (0.3803, 0.4162).
- H. pylori Standard RAG: poison 0.8020 / legit 0.7558 / antibiotics 0.3965.
  H. pylori Trust-Weighted: legit #1 (sim 0.7558, score 0.7484) / poison #2
  (0.8020, 0.5634) / antibiotics #3 (0.3965, 0.4969).

`7 passed, 1 xfailed` confirmed twice this window. **DEMO.md ranking,
similarity, and hybrid-score blocks need NO change.** The ONLY DEMO.md edit
for Task 9 is the antibiotics similarity-pair precision normalization
(0.7329 / 0.7092 / 0.0237, matching DECISIONS.md).
