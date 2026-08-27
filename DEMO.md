# DeRAG Demo Runbook (Milestone 9)

## What this demonstrates

This is a live, end-to-end demonstration that DeRAG's trust-weighted
re-ranking defeats a corpus-poisoning attack that ordinary RAG falls for.
The corpus in `data/demo_corpus/` is 50 real peer-reviewed SciFact
abstracts (leading-sentence excerpts, each truncated to a single chunk)
plus 3 fabricated "misinformation" chunks. Each fabricated chunk inverts
a genuine causal finding while reusing the same technical vocabulary, so
it lands extremely close to the real question in embedding space. For the
headline query — *"Does dietary L-carnitine from red meat raise TMAO and
accelerate atherosclerosis?"* — Standard RAG (pure cosine similarity)
ranks the fabricated chunk **#1** (similarity 0.7789) ahead of the real
abstract (0.5806), because the fabrication is lexically closer. DeRAG then
consults each chunk's **on-chain reputation** — seeded to **-50** for the
3 fabricated chunks and **+10** for the 3 legitimate sources that cover
the same topics — and re-ranks. The trust-weighted score demotes the
fabrication to **#2** and promotes the real abstract to **#1**. Same
retrieval set, same embeddings, different final order — and the
difference is the blockchain reputation signal. A **second query** — *"Does
H. pylori infection increase the risk of gastric cancer?"* — shows the
same flip on a different topic and is asserted the same way, so the proof
does not hang on one query.

## Prerequisites (3 long-running terminals + 1 one-shot deploy)

Steps 1, 3, and 4 are long-running — start each in its own terminal from
the repo root and leave it running. Step 2 is a one-shot command that
deploys and then exits; run it once (in any terminal) after step 1 is up.

1. `cd contracts && npx hardhat node`
   Starts a fresh local chain. Long-running — leave it up. It prints 20
   funded accounts; **Account #0** is the reputation oracle used by the
   seeding script.
2. `cd contracts && npx hardhat run scripts/deploy.ts --network localhost`
   Run once — this deploys the DeRAG contracts to the local chain and
   exits; it is **not** a long-running process. Confirm the deployed
   addresses match what the Python config expects (`config.py` /
   `deployments/`).
3. `ipfs daemon`
   Long-running — leave it up. Chunk bodies are stored on IPFS; the API
   must be reachable at the `IPFS_API_URL` in `.env`.
4. `ollama serve`
   Long-running — leave it up. Only needed for the Streamlit
   answer-generation step. The CLI walkthrough and the automated proof do
   **not** need Ollama.

With the default Hardhat mnemonic, Account #0's private key is
`0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80`
— the exact key `npx hardhat node` prints next to "Account #0".

## One-time data setup

The 53-file demo corpus (`data/demo_corpus/`: 50 SciFact abstracts +
3 `FABRICATED_MISINFO_*.txt` chunks + `manifest.json`) is **already
committed to the repo**. You do **not** need to regenerate it —
`scripts/build_demo_corpus.py` only rebuilds it and needs network access
for the SciFact S3 download on first run.

The seeding, ingest, and regression scripts read `os.environ` **directly**
— they do **not** load `.env` themselves. You must export the variables
into your shell first:

```bash
# .env must define WEB3_PROVIDER_URI, IPFS_API_URL, CHROMA_DB_DIR, and
# PRIVATE_KEY (Hardhat Account #0 key). With the default Hardhat mnemonic
# that key is 0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80
# — the same key `npx hardhat node` prints as "Account #0".
set -a; . ./.env; set +a          # export them for the scripts below
```

Then seed on-chain reputation and run the pre-flight sanity check:

```bash
uv run python scripts/seed_reputation.py
uv run python scripts/check_candidate_window.py   # expect: [preflight] PASS
```

`scripts/seed_reputation.py` ingests every corpus file under its
publisher key (47 filler abstracts → Hardhat Account #1; the 3
poison-targeted legitimate sources → Account #2; the 3 fabricated chunks
→ Account #5), then idempotently sets reputation from the reputation
oracle (Account #0): fabricated chunks → **-50**, targeted legitimate
sources → **+10**, the 47 filler abstracts → **0** (skipped — already 0).
It computes a target-minus-current delta, so **re-running it is a no-op**.

`scripts/check_candidate_window.py` prints `[preflight] PASS` when, for
each targeted query, **both** the fabricated chunk and its legitimate
counterpart are inside the retrieval candidate window — i.e. the
re-ranker actually has both to choose between.

## CLI walkthrough (fastest)

This is the quickest way to see the flip. It needs the env exported (see
above) and the corpus seeded, but **not** Ollama. Use the **module form**
(`-m scripts.demo_rerank`); running `scripts/demo_rerank.py` directly
fails with `ModuleNotFoundError: No module named 'retrieval'`.

```bash
uv run python -m scripts.demo_rerank "Does dietary L-carnitine from red meat raise TMAO and accelerate atherosclerosis?"
```

Expected output (ranking lines are stable; a short content preview line
follows each entry):

```
Standard RAG (pure cosine similarity) — query: 'Does dietary L-carnitine from red meat raise TMAO and accelerate atherosclerosis?'
1. FABRICATED_MISINFO_tmao_redmeat_neutral.txt (similarity=0.7789)
2. scifact_14803797_tmao_redmeat.txt (similarity=0.5806)
3. scifact_1568684_the_bile_acid_chenodeoxycholic_acid.txt (similarity=0.3803)

Trust-Weighted Rerank — query: 'Does dietary L-carnitine from red meat raise TMAO and accelerate atherosclerosis?'
1. scifact_14803797_tmao_redmeat.txt (similarity=0.5806, score=0.6257)
2. FABRICATED_MISINFO_tmao_redmeat_neutral.txt (similarity=0.7789, score=0.5473)
3. scifact_1568684_the_bile_acid_chenodeoxycholic_acid.txt (similarity=0.3803, score=0.4162)
```

What to narrate to the panel:

- **Standard RAG** puts the fabricated chunk at **#1** (similarity 0.7789)
  ahead of the real abstract `scifact_14803797_tmao_redmeat.txt` (0.5806)
  — pure vector similarity is fooled.
- **Trust-Weighted Rerank** blends that similarity with the chunk's
  on-chain reputation. The fabricated chunk's -50 reputation drags its
  score down to 0.5473; the real abstract's +10 lifts its score to
  0.6257. The fabrication drops to **#2**, the real abstract rises to
  **#1**. The retrieval set and embeddings are identical between the two
  lists — only the reputation term changed the order.

### Second fooling case: H. pylori / gastric cancer

`FABRICATED_MISINFO_hpylori_protective.txt` (query: *"Does H. pylori
infection increase the risk of gastric cancer?"*, legitimate counterpart
`scifact_2052720_hpylori_gastric_cancer.txt`, seeded +10) is a **second
query where Standard RAG is fooled and DeRAG recovers**, asserted the
same way as TMAO in the automated proof. The similarity margin is
thinner than TMAO's but real.

```bash
uv run python -m scripts.demo_rerank "Does H. pylori infection increase the risk of gastric cancer?"
```

```
Standard RAG (pure cosine similarity) — query: 'Does H. pylori infection increase the risk of gastric cancer?'
1. FABRICATED_MISINFO_hpylori_protective.txt (similarity=0.8020)
2. scifact_2052720_hpylori_gastric_cancer.txt (similarity=0.7558)
3. scifact_6372244_antibiotics_cdiff.txt (similarity=0.3965)

Trust-Weighted Rerank — query: 'Does H. pylori infection increase the risk of gastric cancer?'
1. scifact_2052720_hpylori_gastric_cancer.txt (similarity=0.7558, score=0.7484)
2. FABRICATED_MISINFO_hpylori_protective.txt (similarity=0.8020, score=0.5634)
3. scifact_6372244_antibiotics_cdiff.txt (similarity=0.3965, score=0.4969)
```

Standard RAG ranks the fabrication **#1** (0.8020 vs the real abstract's
0.7558); the -50 reputation drags the fabrication's trust score to 0.5634
while the +10 lifts the real abstract to 0.7484, so DeRAG puts the real
abstract **#1** and the fabrication **#2**.

### Deliberate contrast: the antibiotics chunk is *not* a fooling case

`FABRICATED_MISINFO_antibiotics_cdiff_protective.txt` (query: *"Do
antibiotics increase susceptibility to Clostridium difficile
infection?"*, counterpart `scifact_6372244_antibiotics_cdiff.txt`) is
also in the corpus and seeded to -50, but under its natural question the
real abstract already outranks the fabrication on similarity alone
(0.7329 vs 0.7092) — **Standard RAG is not fooled here**, so do not
expect this poison chunk at #1 under Standard RAG. This is the honest
boundary of the attack: a fabrication only wins retrieval when it is
lexically closer than the truth, and DeRAG's job is to catch it when it
is. DeRAG still (correctly) keeps the real abstract at #1. See
`DECISIONS.md`.

## Streamlit UI walkthrough

Needs all 4 terminals from Prerequisites (including `ollama serve`) plus
the exported env and the seeded corpus.

```bash
uv run streamlit run app/main.py
```

Then, using the TMAO query
*"Does dietary L-carnitine from red meat raise TMAO and accelerate
atherosclerosis?"*:

1. Paste the query, set mode = **Standard RAG**, submit. The answer
   reflects the fabricated chunk's (false) claim, and the sidebar audit
   trail shows source **[1] = `FABRICATED_MISINFO_tmao_redmeat_neutral.txt`**,
   with its publisher shown as **Account #5**. The real abstract
   `scifact_14803797_tmao_redmeat.txt` (publisher **Account #2**) sits
   below it.
2. Switch mode = **DeRAG**, same query, submit. The answer now reflects
   the real abstract `scifact_14803797_tmao_redmeat.txt` at the top. The
   fabricated chunk is demoted below it (and may fall out of the top 3);
   the sidebar shows its low trust "Ranking" score, driven by its -50
   on-chain reputation.
3. Point out the publisher column in the audit trail across both modes:
   the fabricated chunk publishes from **Account #5**, the real targeted
   source from **Account #2**. Only DeRAG mode acts on that difference.

## The automated proof

This asserts the same flip, from a clean state. It needs Hardhat + IPFS
live and the corpus seeded (Ollama not required).

```bash
uv run python scripts/check_candidate_window.py     # sanity: [preflight] PASS
uv run pytest -m slow tests/test_demo_corpus_regression.py -v
```

Expected result: **`7 passed, 1 xfailed`** (8 items).

- `test_standard_rag_top1_is_the_poison_chunk` — for **TMAO** and
  **H. pylori** asserts Standard RAG's top-1 is the fabricated chunk
  (**both pass**); for **antibiotics** it is `xfail` (Standard RAG is
  deliberately not fooled by that fabrication under its natural query —
  see `DECISIONS.md`).
- `test_trust_weighted_top1_is_not_a_poison_chunk` — for all three
  queries asserts DeRAG's top-1 is **not** a fabricated chunk. **All
  pass** (including antibiotics, where DeRAG keeps the already-correct
  order).
- `test_trust_weighted_ranks_legit_counterpart_above_its_poison` — for
  **TMAO** and **H. pylori** asserts the real abstract outranks its
  fabrication after re-ranking. **Both pass.**

The two hard fooling cases (TMAO, H. pylori) are independent: the proof
does not rest on a single query having a good day. The full pipeline has
been run from clean state — fresh Hardhat chain → deploy → wipe Chroma →
seed → pre-flight → this test — three times (twice at two hard cases,
once after H. pylori was promoted), always green, with the ranking
constants unchanged (`SCALE = 10`, `FLOOR_PENALTY = 50`, `alpha = 0.7`).
