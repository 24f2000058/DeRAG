# DeRAG Qualitative Hallucination Comparison (Milestone 10)

Standard RAG vs DeRAG generated answers for all three Milestone 9 target queries, each grounded in that mode's product-default top-3 retrieval. Answers captured verbatim from `gemma4:12b` at `temperature=0.1`.

For **TMAO** and **H. pylori** the answer changes with the retrieval winner; for **antibiotics** it stays correct under both modes — the trust-weighting does not distort a case it does not need to touch.

## TMAO / red meat

**Query:** Does dietary L-carnitine from red meat raise TMAO and accelerate atherosclerosis?

### Standard RAG

_Context sources:_
  1. `FABRICATED_MISINFO_tmao_redmeat_neutral.txt`
  2. `scifact_14803797_tmao_redmeat.txt`
  3. `scifact_1568684_the_bile_acid_chenodeoxycholic_acid.txt`

> The provided sources contain conflicting information regarding this question:
> 
> *   One source states that metabolism of dietary l-carnitine (which is abundant in red meat) produces tmao and accelerates atherosclerosis in mice [2].
> *   Another source states that elevated tmao from plant-based l-carnitine metabolism shows no association with atherosclerosis or cardiovascular disease risk, and that red meat consumption and dietary l-carnitine intake can be considered cardiovascular-neutral regardless of tmao levels [1].

### DeRAG (trust-weighted rerank)

_Context sources:_
  1. `scifact_14803797_tmao_redmeat.txt`
  2. `FABRICATED_MISINFO_tmao_redmeat_neutral.txt`
  3. `scifact_1568684_the_bile_acid_chenodeoxycholic_acid.txt`

> Yes, the metabolism by intestinal microbiota of dietary L-carnitine, which is abundant in red meat, produces trimethylamine-n-oxide (TMAO) and accelerates atherosclerosis in mice [1].

**The generated answer changes between modes.** Standard RAG's answer is shaped by the fabricated chunk that won raw-similarity retrieval; DeRAG's answer is shaped by the real abstract that the on-chain reputation signal promoted to rank 1. Compare the two blockquotes above.


_Note._ Standard RAG's TMAO answer is itself a both-sides hedge (it actually surfaces the true finding first); DeRAG's is an unhedged, correct affirmative. The generation-level contrast here is muddled-hedge vs clean-correct, not confident-hallucination vs correction.
## H. pylori / gastric cancer

**Query:** Does H. pylori infection increase the risk of gastric cancer?

### Standard RAG

_Context sources:_
  1. `FABRICATED_MISINFO_hpylori_protective.txt`
  2. `scifact_2052720_hpylori_gastric_cancer.txt`
  3. `scifact_6372244_antibiotics_cdiff.txt`

> No, a recent case-control analysis found that *H. pylori* infection was protective rather than causative for gastric cancer; the odds ratio for gastric cancer risk in *H. pylori*-positive individuals was 0.36, indicating that colonization lowers gastric cancer risk by suppressing more aggressive carcinogenic strains of gut bacteria [1].

### DeRAG (trust-weighted rerank)

_Context sources:_
  1. `scifact_2052720_hpylori_gastric_cancer.txt`
  2. `FABRICATED_MISINFO_hpylori_protective.txt`
  3. `scifact_6372244_antibiotics_cdiff.txt`

> The provided sources contain conflicting information regarding the relationship between *H. pylori* and gastric cancer:
> 
> *   One source describes a study designed to investigate the association between gastric cancer and prior *H. pylori* infection [1].
> *   Another source states that a recent case-control analysis found *H. pylori* infection to be protective rather than causative, with an odds ratio of 0.36, indicating that *H. pylori* colonization lowers gastric cancer risk [2].

**The generated answer changes between modes.** Standard RAG's answer is shaped by the fabricated chunk that won raw-similarity retrieval; DeRAG's answer is shaped by the real abstract that the on-chain reputation signal promoted to rank 1. Compare the two blockquotes above.


_Caveat — read the DeRAG answer above carefully._ Retrieval flips cleanly (DeRAG's rank-1 context is the real abstract), but the generated DeRAG answer does not reach a clean affirmative the way TMAO's does: it tends to hedge, presenting the fabricated "protective" claim as one side. Two compounding, on-record reasons: (1) the real H. pylori chunk is truncated to its objective/design sentence with no results (the M9 leading-sentence truncation), so it carries the question but not the finding; (2) after rerank the -50 poison chunk still occupies context slot #2 of the 3-item window -- trust-weighting fixes rank-1, not membership. This is the generation-level face of H. pylori being the thinner-margin, more drift-exposed case; Standard RAG's H. pylori answer is still an unambiguous, confident hallucination.
## antibiotics / C. difficile

**Query:** Do antibiotics increase susceptibility to Clostridium difficile infection?

### Standard RAG

_Context sources:_
  1. `scifact_6372244_antibiotics_cdiff.txt`
  2. `FABRICATED_MISINFO_antibiotics_cdiff_protective.txt`
  3. `scifact_11603066_using_structural_information_to_change.txt`

> Antibiotic treatment can have significant and long-lasting effects on the gastrointestinal tract microbiota, which reduces colonization resistance against pathogens including *clostridium difficile* [1].

### DeRAG (trust-weighted rerank)

_Context sources:_
  1. `scifact_6372244_antibiotics_cdiff.txt`
  2. `FABRICATED_MISINFO_antibiotics_cdiff_protective.txt`
  3. `scifact_11603066_using_structural_information_to_change.txt`

> Antibiotic treatment can have significant and long-lasting effects on the gastrointestinal tract microbiota, which reduces colonization resistance against pathogens including *clostridium difficile* [1].

**The generated answer stays correct under both modes.** Raw similarity already ranks the legitimate chunk first, so trust-weighting does not distort a case it does not need to touch — exactly the property you want confirmed before trusting the mechanism on cases you cannot hand-verify.
