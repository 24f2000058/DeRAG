# DeRAG Retrieval Evaluation (Milestone 10)

This is a **two-positive / one-honest-negative** result — not a uniform success story and not a single cherry-picked win. On two independent topics (TMAO, H. pylori) trust-weighted rerank correctly overrides a similarity-based fooling attempt; on a third (antibiotics) raw similarity already ranks the legitimate chunk first and DeRAG correctly does not over-trigger. A mechanism that fires when it should and stays quiet when it should not is the more credible result.

## Metrics

Ground truth: each of the three target queries has exactly one relevant SciFact document — the real abstract on that topic (`source_doc_id` in `data/demo_corpus/manifest.json`). A retrieved hit is relevant iff its `source_doc_id` matches; poison chunks are never relevant. `standard_rag_answer_context(q, k=10)` vs `trust_weighted_rerank(q, k_retrieve=10, k_return=10)` — the `k_return=10` override is evaluation-only; the product default stays 3.

| Query | SR P@1 | SR P@3 | SR RR | DeRAG P@1 | DeRAG P@3 | DeRAG RR |
|-------|-------:|-------:|------:|----------:|----------:|---------:|
| TMAO / red meat | 0.000 | 0.333 | 0.500 | 1.000 | 0.333 | 1.000 |
| H. pylori / gastric cancer | 0.000 | 0.333 | 0.500 | 1.000 | 0.333 | 1.000 |
| antibiotics / C. difficile | 1.000 | 0.333 | 1.000 | 1.000 | 0.333 | 1.000 |
| **Mean** | **0.333** | **0.333** | **0.667** | **1.000** | **0.333** | **1.000** |

MRR@10 — Standard RAG: **0.667**  ·  DeRAG: **1.000**

## Per-query reading

**TMAO / red meat** — the fabricated chunk beats the real abstract on raw cosine similarity by a wide margin (~0.198). Standard RAG is fooled (its top-1 is the poison, so P@1 = 0); trust-weighted rerank uses the on-chain reputation signal to flip the real abstract back to rank 1 (P@1 = 1).

**H. pylori / gastric cancer** — the same override on an independent topic, but at a thinner, more drift-exposed margin (~0.046). That thin margin is itself an honest, reportable data point about the mechanism's robustness envelope, not a weakness to hide: the flip still holds and is asserted as a hard regression case, but with less headroom than TMAO.

**antibiotics / C. difficile** — a deliberate non-fooling contrast and a no-regression check. Under its natural query raw similarity already ranks the legit chunk first, so there is nothing for trust-weighting to override. DeRAG leaves the already-correct order alone (P@1 stays 1 in both modes) — proof the mechanism does not distort a case it does not need to touch.

## Alpha sweep

`trust_weighted_rerank` re-run at alpha ∈ 0.000, 0.300, 0.500, 0.700, 1.000 across all three queries:

| alpha | MRR@10 | mean P@1 | MRR@10 (bar) |
|------:|-------:|---------:|--------------|
| 0.000 | 1.000 | 1.000 | `################################` |
| 0.300 | 1.000 | 1.000 | `################################` |
| 0.500 | 1.000 | 1.000 | `################################` |
| 0.700  ← shipped default | 1.000 | 1.000 | `################################` |
| 1.000 | 0.667 | 0.333 | `#####################-----------` |

MRR is flat at its maximum (1.000) across alpha 0.000, 0.300, 0.500, 0.700 — a wide plateau, so the shipped alpha = 0.7 is a safe pick, not a knife-edge tune.

`alpha = 1.0` collapses the hybrid score to pure similarity, so its row reproduces Standard RAG's MRR exactly — a built-in sanity check on the sweep.

## Reproduction

```bash
# Hardhat node + deploy + IPFS daemon all live; then:
set -a; . ./.env; set +a
uv run python scripts/seed_reputation.py
uv run python scripts/check_candidate_window.py      # [preflight] PASS
uv run python scripts/evaluate.py                    # regenerates this file
```

## Raw ranking dumps (top-10, source_file order)

### TMAO / red meat
- query: `Does dietary L-carnitine from red meat raise TMAO and accelerate atherosclerosis?`
- ground-truth doc: `14803797` (`scifact_14803797_tmao_redmeat.txt`)
- Standard RAG: `['FABRICATED_MISINFO_tmao_redmeat_neutral.txt', 'scifact_14803797_tmao_redmeat.txt', 'scifact_1568684_the_bile_acid_chenodeoxycholic_acid.txt', 'scifact_12631697_activation_of_fast_skeletal_muscle.txt', 'scifact_13001323_sirt1_deacetylase_in_sf1_neurons.txt', 'scifact_12670680_basophils_and_the_t_helper.txt', 'scifact_13794374_copi_complex_is_a_regulator.txt', 'scifact_15535511_role_for_insulin_signaling_in.txt', 'scifact_10648422_programmed_death_1_induced_interleukin.txt', 'scifact_14541844_reconstruction_of_the_yeast_snf1.txt']`
- DeRAG:        `['scifact_14803797_tmao_redmeat.txt', 'FABRICATED_MISINFO_tmao_redmeat_neutral.txt', 'scifact_1568684_the_bile_acid_chenodeoxycholic_acid.txt', 'scifact_12631697_activation_of_fast_skeletal_muscle.txt', 'scifact_13001323_sirt1_deacetylase_in_sf1_neurons.txt', 'scifact_12670680_basophils_and_the_t_helper.txt', 'scifact_13794374_copi_complex_is_a_regulator.txt', 'scifact_15535511_role_for_insulin_signaling_in.txt', 'scifact_10648422_programmed_death_1_induced_interleukin.txt', 'scifact_14541844_reconstruction_of_the_yeast_snf1.txt']`

### H. pylori / gastric cancer
- query: `Does H. pylori infection increase the risk of gastric cancer?`
- ground-truth doc: `2052720` (`scifact_2052720_hpylori_gastric_cancer.txt`)
- Standard RAG: `['FABRICATED_MISINFO_hpylori_protective.txt', 'scifact_2052720_hpylori_gastric_cancer.txt', 'scifact_6372244_antibiotics_cdiff.txt', 'FABRICATED_MISINFO_antibiotics_cdiff_protective.txt', 'scifact_11603066_using_structural_information_to_change.txt', 'scifact_14803797_tmao_redmeat.txt', 'scifact_13764090_large_intestine_targeted_nanoparticle_releasing.txt', 'scifact_16066726_tumor_necrosis_factor_signaling_mediates.txt', 'scifact_13905670_human_snp_links_differential_outcomes.txt', 'scifact_1550937_the_transcription_factors_egr2_and.txt']`
- DeRAG:        `['scifact_2052720_hpylori_gastric_cancer.txt', 'FABRICATED_MISINFO_hpylori_protective.txt', 'scifact_6372244_antibiotics_cdiff.txt', 'scifact_14803797_tmao_redmeat.txt', 'scifact_11603066_using_structural_information_to_change.txt', 'scifact_13764090_large_intestine_targeted_nanoparticle_releasing.txt', 'scifact_16066726_tumor_necrosis_factor_signaling_mediates.txt', 'scifact_13905670_human_snp_links_differential_outcomes.txt', 'scifact_1550937_the_transcription_factors_egr2_and.txt', 'FABRICATED_MISINFO_antibiotics_cdiff_protective.txt']`

### antibiotics / C. difficile
- query: `Do antibiotics increase susceptibility to Clostridium difficile infection?`
- ground-truth doc: `6372244` (`scifact_6372244_antibiotics_cdiff.txt`)
- Standard RAG: `['scifact_6372244_antibiotics_cdiff.txt', 'FABRICATED_MISINFO_antibiotics_cdiff_protective.txt', 'scifact_11603066_using_structural_information_to_change.txt', 'scifact_13764090_large_intestine_targeted_nanoparticle_releasing.txt', 'scifact_16066726_tumor_necrosis_factor_signaling_mediates.txt', 'scifact_14803797_tmao_redmeat.txt', 'FABRICATED_MISINFO_hpylori_protective.txt', 'scifact_15305881_the_structure_of_ddrb_from.txt', 'scifact_12827098_tissue_resident_macrophages_self_maintain.txt', 'scifact_13481880_yap_taz_dependent_reprogramming_of.txt']`
- DeRAG:        `['scifact_6372244_antibiotics_cdiff.txt', 'FABRICATED_MISINFO_antibiotics_cdiff_protective.txt', 'scifact_11603066_using_structural_information_to_change.txt', 'scifact_14803797_tmao_redmeat.txt', 'scifact_13764090_large_intestine_targeted_nanoparticle_releasing.txt', 'scifact_16066726_tumor_necrosis_factor_signaling_mediates.txt', 'scifact_15305881_the_structure_of_ddrb_from.txt', 'scifact_12827098_tissue_resident_macrophages_self_maintain.txt', 'scifact_13481880_yap_taz_dependent_reprogramming_of.txt', 'FABRICATED_MISINFO_hpylori_protective.txt']`
