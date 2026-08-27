# Milestone 10: Quantitative Evaluation, Final Documentation, and Viva Prep

Status: approved for implementation planning
Date: 2026-08-27

## Goal

Close out the DeRAG roadmap. Milestone 9 delivered a real demo corpus
and a live thesis-proof regression test. Milestone 10 turns that into a
defensible, quantitative evaluation, brings every user-facing document
up to date and mutually consistent, and produces a viva-preparation
document grounded in the project's real decision history.

This is the **last milestone in the roadmap**. Nothing here adds
product capability; it makes the existing capability measurable,
reproducible from a clean clone, and defensible under questioning.

## Framing: two positives, one honest negative

The evaluation result is **not** a uniform success story and **not** a
single cherry-picked win. It is:

- **TMAO** — trust-weighted rerank correctly overrides a
  similarity-based fooling attempt. Poison beats legit on raw cosine by
  a **wide margin (~0.198)**. Standard RAG is fooled; DeRAG recovers.
- **H. pylori** — the same mechanism, same override, on an independent
  topic, at a **thinner, more drift-exposed margin (~0.046)**. That
  thin margin is itself an honest, reportable data point about the
  mechanism's robustness envelope — not a weakness to hide.
- **antibiotics** — a deliberate **non-fooling contrast**. Under its
  natural query the legit chunk already ranks first on raw similarity,
  so there is nothing for trust-weighting to override. DeRAG correctly
  does **not** over-trigger: it leaves the already-correct order alone.
  This is presented as a **no-regression check**, not a failure to
  explain away.

A mechanism that fires when it should and stays quiet when it
shouldn't is a *more* credible result than one that appears to fix
every case. `eval/results.md` and `eval/hallucination_qualitative.md`
must both carry this framing explicitly.

## 1. `manifest.json` extension — `source_doc_id`

### What

Add a `"source_doc_id"` field to every chunk entry in
`data/demo_corpus/manifest.json`:

- `scifact_*` chunks (50 legit): the **numeric SciFact document id**
  as a string, e.g. `"14803797"` for
  `scifact_14803797_tmao_redmeat.txt`.
- `FABRICATED_MISINFO_*` chunks (3 poison): `null`. Poison chunks did
  not originate from any SciFact document. Their linkage to a real
  topic is already carried by `ground_truth_chunk_id`; `source_doc_id`
  stays a pure "which SciFact document did this chunk's text come from"
  field and is not overloaded with "which topic does this attack".

### Why

Milestone 9 flagged this as a follow-up. The evaluation harness needs
to map a retrieved chunk back to its SciFact document to decide
relevance. Today the only way is to string-parse `chunk_id`
(`scifact_{doc_id}_{slug}`). An explicit field removes that fragile
parse.

### Builder change

`scripts/build_demo_corpus.py` already holds the SciFact `doc_id` at
the point it constructs `chunk_id`. It must also write
`source_doc_id` into each manifest entry. For the 3 poison entries
(hand-authored in the builder's `TARGETS`/poison section), set
`source_doc_id` to `None`.

**Byte-identical regeneration is a hard requirement** (Milestone 9
DECISIONS). The new key must be emitted at a fixed position in the
dict, in the existing key order convention, so a regenerated
`manifest.json` differs from the committed one *only* by the added
field — verified by regenerating and diffing.

### Tests

Extend `tests/test_scripts_build_demo_corpus.py`:

- every manifest entry has `source_doc_id`.
- for `scifact_*` entries, `source_doc_id` is the digit-run between
  `scifact_` and the slug in `chunk_id`.
- for `FABRICATED_MISINFO_*` entries, `source_doc_id is None`.
- regenerating the corpus produces a `manifest.json` whose parsed
  content equals the committed file (or differs only as intended if
  run before the commit — assert structural equality of every entry).

Extend `tests/test_demo_corpus_fixtures.py` with a schema assertion
that every committed manifest entry carries `source_doc_id` with the
right type (str for legit, None for poison), and that every non-null
`source_doc_id` appears in that chunk's `chunk_id`.

## 2. `scripts/evaluate.py` — retrieval metrics (no Ollama)

### Purpose

Compute MRR and Precision@K for **both** retrieval modes across **all
three** target queries, plus an alpha sweep, and render
`eval/results.md`.

### Ground truth

Built from `manifest.json`, not hand-coded:

```
build_ground_truth(chunks) -> { query_string: gt_source_doc_id }
```

For each poison chunk with `target_queries`, resolve
`ground_truth_chunk_id` to the legit counterpart chunk, and take that
counterpart's `source_doc_id`. That numeric id is the single relevant
document for the query. (Each of the 3 queries has exactly one relevant
document — the real SciFact abstract on that topic.)

### Relevance

A retrieved hit is relevant iff its `source_doc_id` equals the query's
ground-truth id. The hit's `source_doc_id` is recovered by joining
`hit["metadata"]["source_file"]` to the manifest entry's `filename`,
then reading `source_doc_id`. Poison chunks (`source_doc_id is None`)
are never relevant.

### Metrics

Pure, separately unit-tested functions:

- `reciprocal_rank(ranked_doc_ids, gt_id) -> float` — `1/rank` of the
  first relevant hit, `0.0` if none in the list.
- `precision_at_k(ranked_doc_ids, gt_id, k) -> float` — fraction of the
  top-k that are relevant. With one relevant doc this is `1/k` if the
  doc is in the top-k, else `0.0`.
- `mean_reciprocal_rank(list_of_rr) -> float`.
- `evaluate_mode(mode_fn, queries, ground_truth) -> per-query rows +
  aggregates`.

### Retrieval calls

- Standard RAG: `standard_rag_answer_context(q, k=10)`.
- DeRAG: `trust_weighted_rerank(q, k_retrieve=10, k_return=10)`.

The `k_return=10` override is **evaluation-only**, so MRR/P@K see the
full re-ranked order. The product/UI default (`k_return=3`) is **not**
touched.

Report **P@1, P@3, and RR per query**, plus **MRR@10** and mean P@1 /
mean P@3 per mode. Main table: rows = 3 queries + Mean; columns =
Standard RAG (P@1, P@3, RR) vs DeRAG (P@1, P@3, RR).

### Alpha sweep

For `alpha in [0.0, 0.3, 0.5, 0.7, 1.0]`, re-run
`trust_weighted_rerank(q, k_retrieve=10, k_return=10, alpha=alpha)` for
all 3 queries. Report **MRR@10 and mean P@1 at each alpha**, as a
Markdown table plus a dependency-free ASCII bar chart (matplotlib is
not a project dependency and will not be added). `alpha=1.0` collapses
the hybrid score to pure similarity and should reproduce Standard RAG's
MRR — a built-in sanity check to state in the write-up. `alpha=0.7` is
the shipped default and its row must be marked as such.

### Output — `eval/results.md`

`render_results_md(results_dict) -> str` (pure; `main()` runs live
retrieval, assembles the dict, writes the file, and also prints to
stdout). Sections:

1. **Framing paragraph** — the two-positive/one-honest-negative
   statement above, in prose.
2. **Main comparison table** — Standard RAG vs DeRAG, per query + mean.
3. **Per-query narrative** —
   - TMAO: wide margin (~0.198); Standard RAG P@1 = 0 (poison at rank
     1), DeRAG P@1 = 1.
   - H. pylori: thin margin (~0.046); same flip; call out that the
     thin margin is the honest robustness data point.
   - antibiotics: **no-regression check** — Standard RAG already P@1 =
     1, DeRAG stays P@1 = 1; the mechanism does not distort a case it
     does not need to touch.
4. **Alpha sweep** — table + ASCII chart + the `alpha=1.0 ==
   Standard RAG` sanity note and a one-line reading of what the sweep
   says about the choice of `0.7` (e.g. whether the plateau is wide or
   the result is knife-edge).
5. **Reproduction** — the exact command block (clean-state pipeline
   through `uv run python scripts/evaluate.py`).
6. **Raw ranking dumps** — the captured top-10 for each query/mode, so
   the numbers in the tables are auditable.

### Tests — `tests/test_scripts_evaluate.py` (fast, no services)

Unit-test the pure functions with synthetic ranked lists:

- `reciprocal_rank`: relevant at rank 1 → 1.0; rank 3 → 1/3; absent →
  0.0.
- `precision_at_k`: single relevant in top-k → 1/k; outside → 0.0.
- `mean_reciprocal_rank`: known set → known mean.
- `build_ground_truth`: on a fixture manifest slice, returns the legit
  counterpart's `source_doc_id` per query.
- relevance join: a hit whose `source_file` is the poison file is
  non-relevant; the legit file is relevant.
- `render_results_md`: given a hand-built results dict, output contains
  the framing sentence, all three query rows, the alpha-sweep header,
  and the antibiotics "no-regression" label.

## 3. `scripts/eval_hallucination.py` -> `eval/hallucination_qualitative.md` (needs Ollama)

Kept **separate** from `evaluate.py` so the Ollama dependency is
isolated — the metrics run without a model.

### Behaviour

For each of the 3 queries x {Standard RAG, DeRAG}:

- retrieve the **product-default top-3** (`standard_rag_answer_context(q,
  k=3)` / `trust_weighted_rerank(q, k_retrieve=10, k_return=3)`),
- `build_prompt(q, docs)` + `generate_blocking(prompt)` (deterministic
  enough at `temperature=0.1`; capture verbatim),
- record the answer text and the 3 `source_file`s that fed it.

`render_md(cases) -> str` writes `eval/hallucination_qualitative.md`:
Standard RAG vs DeRAG answer **side by side** per query, each with its
3 feeding sources listed, plus commentary:

- **TMAO / H. pylori**: the generated answer **changes** between modes —
  Standard RAG's answer reflects the fabricated (inverted) claim
  because the poison won retrieval; DeRAG's answer reflects the real
  abstract because the reputation signal flipped the top result. Show
  the change explicitly.
- **antibiotics**: the generated answer stays **correct under both
  modes** — proof the trust-weighting does not distort a case it does
  not need to touch, which is exactly the property you want confirmed
  before trusting the mechanism on cases you cannot hand-verify.

### Tests

Light — `tests/test_scripts_eval_hallucination.py`: a prompt-assembly /
`render_md` unit test (no live generation). The actual answers are
produced in the live-services window and committed as captured text,
same pattern as `DEMO.md`'s captured outputs.

## 4. `README.md` — "Services to start before running anything"

Replace the Milestone-10 placeholder with a section sufficient for a
stranger to run the whole project from a clean clone.

### 4a. Service startup order

1. `cd contracts && npx hardhat node` — long-running (port 8545).
2. `cd contracts && npx hardhat run scripts/deploy.ts --network
   localhost` — one-shot, deploys and exits.
3. `ipfs daemon` — long-running (API 5001, gateway 8080).
4. `ollama serve` — long-running (port 11434). Only needed for
   answer generation (Streamlit UI, `scripts/demo_generate.py`,
   `scripts/eval_hallucination.py`).
5. Export env + seed the corpus:
   `set -a; . ./.env; set +a` then
   `uv run python scripts/seed_reputation.py` then
   `uv run python scripts/check_candidate_window.py`.
6. `uv run streamlit run app/main.py` — the UI (port 8501).

### 4b. Port table

| Port  | Service            | Notes |
|-------|--------------------|-------|
| 8545  | Hardhat JSON-RPC   | `WEB3_PROVIDER_URI` |
| 5001  | IPFS (Kubo) API    | `IPFS_API_URL` |
| 8080  | IPFS gateway       | hash-verification fetches |
| 11434 | Ollama             | `OLLAMA_HOST`; answer generation only |
| 8501  | Streamlit UI       | no env var |

### 4c. Models / tuning to pre-fetch before running anything

- **Pull the model:** `ollama pull gemma4:12b` (~7.6 GB).
- **`.env`:** `cp .env.example .env`, set `PRIVATE_KEY` to the Hardhat
  Account #0 key
  (`0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80`
  with the default mnemonic). The `scripts/` read `os.environ`
  directly and do **not** load `.env` themselves, so every shell that
  runs them must first:
  `set -a; . ./.env; set +a`.
- **Ollama systemd tuning (REQUIRED for `gemma4:12b` to be viable).**
  Without it the model falls back to CPU / partial GPU offload — a
  **silent performance regression, not a crash**, and easy to miss.
  `sudo systemctl edit ollama.service`, drop-in
  `/etc/systemd/system/ollama.service.d/override.conf`, **verbatim,
  including the `[Service]` header** (omitting it makes systemd ignore
  every line — "Assignment outside of section" — while the service
  still reports `active`):

  ```
  [Service]
  Environment="OLLAMA_NOPRUNE=1"
  Environment="OLLAMA_KV_CACHE_TYPE=q8_0"
  Environment="OLLAMA_FLASH_ATTENTION=1"
  Environment="OLLAMA_KEEP_ALIVE=-1"
  ```

  then `sudo systemctl daemon-reload && sudo systemctl restart ollama`.
  Verify with `systemctl show ollama -p Environment`.

### 4d. Status section

Keep the existing per-milestone Status list. Add a Milestone 10 entry
pointing at `eval/results.md`, `eval/hallucination_qualitative.md`, and
`VIVA.md`.

## 5. `VIVA.md`

### Source of truth

Before writing, reconstruct the **real** decision history across all 9
completed milestones using claude-mem `timeline-report` (or repeated
targeted `mem-search` on specific terms from each milestone's actual
decisions — not generic queries). Every answer in the document must be
grounded in a decision that was actually made and is on record
(DECISIONS.md / claude-mem), not a plausible-sounding reconstruction.

### Structure

Anticipated Q&A organized by the academic overview's three areas:

- **Deep Learning** — embedding model choice (`all-MiniLM-L6-v2`),
  `gemma4:12b` over the spec's Phi-3/Llama-3 and why (quality/memory on
  the real target hardware), `think: false` / `/api/chat`, chunking
  (500-char split), sigmoid reputation normalization vs per-batch
  min-max and why (MRR needs a fixed curve), `alpha=0.7` and the
  Milestone 10 sweep result.
- **ISR (Information Storage & Retrieval)** — SciFact over
  Climate-FEVER, ChromaDB cosine store, single-chunk corpus constraint
  and the leading-sentence truncation trade, candidate-window
  pre-flight, MRR/P@K methodology and the two-positive/one-honest-
  negative result, IPFS content-addressing + `verify_hash`.
- **Blockchain** — Hardhat local chain, oracle/owner-gated reputation
  ledger, publisher attribution (demo-narrative only —
  `trust_weighted_rerank` never reads publisher), duplicate-hash
  protection, idempotent `delta = target - current` seeding, in-memory
  chain state not surviving restarts.

### The hardest anticipated question

No longer "why only one hard-asserted case". It is now:

> "antibiotics behaves differently from TMAO and H. pylori — how do you
> know that's a real, honest result and not cherry-picked?"

The answer must be the **actual sequence of events**, not a flattened
summary:

1. The single-chunk ingestion constraint
   (`seed_reputation.resolve_doc_hash` requires exactly one chunk;
   `pipeline/chunking.py` splits >500 chars) forced truncating the
   legit SciFact abstracts to leading-sentence excerpts.
2. That truncation **mechanically flipped 2 of the 3 designed
   similarity orderings** — the leading OBJECTIVE/DESIGN sentences are
   less similar to a results-oriented question than the full abstract
   was; the spec's §2 numbers were whole-abstract.
3. `DEMO.md` was initially found to **still describe the original
   (pre-truncation) framing** and was corrected during the H. pylori
   promotion.
4. H. pylori's flip was **re-examined and found to be a real,
   data-supported second fooling case** that had been under-promoted (a
   measured-but-not-asserted oversight), not a curiosity — and was
   properly added as a hard regression case across all three
   assertions.
5. In doing that rigorously, a **test-isolation bug** was caught: the
   antibiotics `xfail` marker sat on a monolithic test that also
   carried the trust-weighted assertion, so a genuine DeRAG regression
   on the antibiotics query could have been **silently masked** by the
   expected Standard-RAG failure. Fixed by splitting the trust-weighted
   assertion into its own **unconditional** test
   (`test_trust_weighted_top1_is_not_a_poison_chunk`, no xfail on any
   query).

Framed as a **strength**: the process caught its own mistakes and the
evaluation got stronger — two independent hard cases plus an honest
negative, with tighter test isolation — as a direct result. Not
something to be defensive about.

## 6. `DEMO.md` re-verification (required, not a silent fold-in)

After 1-5 land, re-verify `DEMO.md` against reality a second time —
because Milestone 10's own changes (evaluation additions, README
updates) are landing, and the point of this pass is that nothing drifts
out of sync the way it did the first time.

- Re-run every scripted query in `DEMO.md` in the live-services window
  and confirm each captured expected-output block still matches.
- Confirm the narration (which chunk wins under which mode, the
  reputation values, the pass/xfail counts) still matches.

### The antibiotics similarity-pair discrepancy — investigate, don't just patch

`DEMO.md` states the antibiotics pair as **legit 0.7329 vs poison
0.7092**; `DECISIONS.md` and the fixtures-test comment state it as
**legit ~0.733 vs poison ~0.709 (margin 0.024)**. `DEMO.md` has been
edited at two different points in this project's history (originally,
then corrected during the H. pylori promotion), so a numeric
discrepancy could mean either:

- a harmless rounding/formatting difference (4dp in DEMO.md vs 3dp
  elsewhere), **or**
- a genuine remnant of a stale measurement taken before some fix
  landed that never got fully propagated.

This is the last documentation pass the project gets. Determine which
it actually is: recompute the antibiotics `legit_sim` / `poison_sim`
live with the current corpus files and the current embedding pipeline,
compare against **both** documents, and record in the Milestone 10
completion report (and DECISIONS.md if it turns out to be a stale
remnant) **what the drift was and why it existed** — not just "fixed
drift". A loose thread like this must not survive to the viva
unexplained.

## 7. `DECISIONS.md` — closing Milestone 10 entry

A `## Milestone 10` section covering:

- `source_doc_id` added to `manifest.json` (rationale: kill the
  `chunk_id` string-parse in the eval harness); `null` for poison;
  byte-identical regeneration preserved.
- Evaluation methodology: MRR@10 + P@1/P@3, both modes, 3 queries,
  ground truth from the manifest, `k_return=10` eval-only override.
- **Headline results** — the actual measured MRR/P@1 numbers for both
  modes, stated plainly, with the two-positive/one-honest-negative
  reading.
- **Alpha sweep** — confirm it was done and state **what it showed**
  (whether `alpha=0.7` sits on a wide plateau or is sensitive; whether
  `alpha=1.0` reproduced Standard RAG as expected).
- The antibiotics similarity-pair discrepancy resolution (rounding vs
  stale remnant — whichever it turned out to be).
- Explicit statement: **this is the last milestone in the DeRAG
  roadmap.**

## Execution order

1. `manifest.json` + `build_demo_corpus.py` + their tests (offline,
   TDD).
2. `scripts/evaluate.py` + `tests/test_scripts_evaluate.py` (offline,
   TDD — pure functions and `render_results_md` fully tested against
   synthetic data).
3. `scripts/eval_hallucination.py` + its render unit test (offline).
4. **One live-services window:** start Hardhat node + deploy + IPFS
   daemon + Ollama; export `.env`; `seed_reputation.py`;
   `check_candidate_window.py` (`[preflight] PASS`); then:
   - run `scripts/evaluate.py` -> commit `eval/results.md`,
   - run `scripts/eval_hallucination.py` -> commit
     `eval/hallucination_qualitative.md`,
   - recompute the antibiotics similarity pair for the DEMO.md
     investigation,
   - re-run every `DEMO.md` scripted query to confirm its captured
     outputs.
5. `README.md` — services / ports / models-and-tuning section.
6. `VIVA.md` — claude-mem reconstruction first, then the Q&A.
7. `DEMO.md` — apply any fixes found in step 4; record the antibiotics
   discrepancy finding.
8. `DECISIONS.md` — closing Milestone 10 entry (including the alpha
   sweep outcome and the discrepancy resolution).
9. Code review + finish the branch.

## Definition of done

- `eval/results.md` presents the two-positive/one-honest-negative
  result with the framing above (not a single cherry-picked win),
  including the alpha sweep.
- `eval/hallucination_qualitative.md` shows Standard RAG vs DeRAG
  generated answers for all 3 queries: changing for TMAO and H. pylori,
  stable-and-correct for antibiotics under both modes.
- `README.md` is sufficient for a stranger to run the whole project
  from a clean clone, including `ollama pull gemma4:12b`, the verbatim
  systemd override block, and the `.env` export requirement.
- `manifest.json` carries `source_doc_id` per chunk and still
  regenerates byte-identically (modulo the intended field).
- `DEMO.md` accurately reflects what will actually happen when each
  scripted query is run live, and the antibiotics similarity-pair
  discrepancy is explained, not just patched.
- `VIVA.md` reads like it was written by someone who made these
  decisions and caught their own mistakes along the way, with the
  hardest question answered as a sequence of real events.
- `DECISIONS.md` has a closing entry stating the alpha sweep outcome
  and that this is the last milestone.
- Full fast test suite green; the live regression + evaluation run
  green in the live-services window.
