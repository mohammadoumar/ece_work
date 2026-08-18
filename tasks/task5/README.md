# Task 5: Benchmarking and Metrics

**Status: Prototype done, verified run**

A repeatable precision/recall/F1 harness for the Task 2 extraction pipeline, scaled from a single probe document to a small labeled sample.

Notebook: [`task5_benchmarking.ipynb`](./task5_benchmarking.ipynb)

## Setup

- Sample: 25 documents from DocRED's `train_annotated` split — doc 0 is Task 2's original "AirAsia Zest" probe document (kept deliberately for cross-checking), plus 24 seeded-random others.
- Pipeline under test: the same spaCy `en_core_web_sm` NER baseline vs. DocRED gold entities (all mention aliases, normalized) from Task 2, generalized into reusable functions and scored as real precision/recall/F1 (TP/FP/FN) per document, aggregated both micro (pooled counts) and macro (mean of per-doc scores).

## Results (verified run)

- Aggregate: **micro P=0.788, R=0.788, F1=0.788**; **macro P=0.795, R=0.792, F1=0.790** — the two aggregation methods agree within 0.01, consistent with the sampled documents being fairly uniform in size (10–34 gold entities each).
- Precision and recall are almost exactly balanced in aggregate, meaning spaCy isn't systematically biased toward over- or under-extraction *on average* — but that balance hides real per-document swings (precision 0.47–1.00, recall 0.57–1.00).
- Real variance across documents: F1 std = 0.106, ranging from **0.53** (doc 1723, "Victorian Gay and Lesbian Rights Lobby" — precision only 0.47, 10 false positives out of 19 predicted entities) to a perfect **1.00** (doc 1027, "President of Harvard University" — 11/11 gold entities exactly matched, zero false positives).
- **Cross-check against Task 2:** doc 0's recall here (0.773 = 17/22) exactly matches Task 2's reported "77% match," confirming the harness correctly reproduces Task 2's original result — while also revealing that figure was recall, not precision (0.810) or F1 (0.791), since Task 2 never separated those three.
- Doc 0's F1 (0.791) happened to land very close to the 25-document aggregate F1 (0.788–0.790) — the original single-document estimate was, in hindsight, fairly representative. That's a fortunate coincidence, not a validation of the one-document methodology: the ±0.11 F1 std shows individual documents can swing widely enough that no single probe document could be trusted to generalize on its own.

## Open next steps

1. Dig into the worst-performing document (doc 1723, F1=0.53) as a concrete case study — genuine spaCy limitation, or another artifact of the comparison harness (as Task 2 found twice already)?
2. Scale further (e.g. 100+ documents) now that the harness is verified, to get a tighter confidence interval on the aggregate F1.
3. Re-run this exact harness against a better extractor (e.g. an LLM-prompted extractor) or a merged/domain-adapted model (Task 4), to use it as the repeatable metric Task 5 was meant to produce.
4. Extend to relation extraction precision/recall/F1 (still not attempted since Task 2), using the same TP/FP/FN scoring pattern established here.

## Conclusion

The harness reproduces Task 2's original number exactly (confirming correctness) while revealing what that number actually was (recall, not F1) and how much it could have varied by chance (±0.11 F1 std across documents). The project now has a real, reusable precision/recall/F1 harness — with micro and macro aggregation — to score any future extraction or merging change against, rather than relying on single-document impressions.
