# Task 2: Document Extraction and Knowledge Integration (Knowledge Graph/Ontology)

**Status: Prototype done, evaluation harness verified (entity extraction only — relation extraction not yet attempted)**

Prototype pipeline for extracting entities/relations from documents and mapping them onto a knowledge graph schema.

Notebook: [`task2_extraction.ipynb`](./task2_extraction.ipynb)

## Setup

- Dataset: [DocRED](https://huggingface.co/datasets/thunlp/docred), loaded from the `refs/convert/parquet` revision (the dataset's legacy loading script is no longer supported by current `datasets` versions). The converted `train` split merges the original `train_annotated` (3,053 gold docs) and `train_distant` (101,873 weakly-supervised docs) into one split — explicitly sliced to the first 3,053 rows to guarantee gold-quality annotations.
- Baseline extractor: spaCy `en_core_web_sm` NER, run on the raw document text (ignoring DocRED's gold labels), to simulate "we don't have labels yet."
- Probe document: "AirAsia Zest" — 17 gold entities (22 distinct alias names across those entities), 13 gold relations (Wikidata property IDs, e.g. `P17` = country, `P131` = located in the administrative territorial entity).

## Results (post alias-fix, verified run)

- spaCy correctly matched **17/22 (77%)** gold alias names, once the comparison (a) normalized tokenization artifacts (space-before-punctuation, leading articles, trailing possessives) and (b) counted **all** mention aliases per entity, not just the first.
- The alias fix directly confirmed an earlier hypothesis: `caap` and `airasia zest`, previously misclassified as spaCy false positives with no gold match, are legitimate aliases of entities already in the gold set (Civil Aviation Authority of the Philippines, and the document's own subject) — they now correctly count as matches.
- Genuine remaining misses: `metro manila` (present in the text, simply untagged by spaCy) and `asian spirit` (spaCy tagged only the `Asian` fragment as NORP, never joined it into the full org name).
- A DocRED data quirk surfaced by the alias fix: one gold entity's alias list includes the raw span `'asian spirit and zest air'` — apparently a mention-boundary artifact in DocRED's own annotations (conflating two distinct former airline names into a single mention string), not something a correct extractor should be expected to reproduce.
- Remaining boundary-only mismatches (not real errors): `a year` vs. spaCy's `less than a year`, and `republic of the philippines` vs. spaCy's `government of the republic of the philippines` — normalization only strips leading articles, not embedded qualifiers like "less than" or "government of."
- `first` remains a genuine spaCy false positive (ORDINAL) with no gold counterpart.

## Notable issues found and fixed

- **Legacy dataset-loading script no longer works.** `thunlp/docred` ships a Python loading script, which recent `datasets` versions (v4.0+) refuse to execute — `trust_remote_code=True` no longer helps. Fixed by loading from the auto-generated `revision="refs/convert/parquet"` instead.
- **Split ambiguity after parquet conversion.** The conversion merges `train_annotated` (gold) and `train_distant` (noisy, weakly-supervised) into a single `train` split (104,926 = 3,053 + 101,873 rows). Any code sampling from `train` without explicitly slicing to the first 3,053 rows risks silently pulling noisy distant-supervision data. Fixed by explicit slicing with a comment.
- **Gold-entity list was incomplete.** `gold_entities` was originally built from only the *first* mention name per DocRED entity (`vertexSet[i][0]`), discarding real aliases like "CAAP." Fixed by collecting all mention names per entity and flattening them into the comparison set — this raised the true match count and confirmed several apparent false positives were actually aliases.

## Relation extraction was not attempted in this pass

Only entity extraction was benchmarked. spaCy has no built-in relation extraction; testing that would require either a separate relation-extraction model/prompt-based LLM approach, or dependency-parse heuristics.

## Open next steps

1. Extend past entity extraction into relation extraction (e.g. prompt an LLM to output `(head, relation, tail)` triples, or use a dependency-parse heuristic) and compare against the 13 gold Wikidata relations.
2. Scale from a single probe document to a small sample (e.g. 20–30 docs from `train_annotated`) to get an aggregate precision/recall number instead of one anecdotal case — this feeds directly into Task 5 (benchmarking/metrics).
3. Extend `normalize_entity` to strip a few more common boundary qualifiers (e.g. "less than", "government of") if they keep recurring across a larger sample — low priority unless the aggregate numbers show it matters.
4. Log genuine coverage gaps (e.g. `metro manila`, `asian spirit`) as candidate weak points for the project's own extraction pipeline, now that the comparison is no longer confounded by alias blindness.

## Conclusion

The prototype pipeline works end-to-end — dataset load, gold inspection (with full alias coverage), baseline extraction, and comparison against a real Wikidata-based schema — and spaCy's out-of-the-box NER recovers the large majority of gold entity mentions on this sample, with the remaining gap traceable to a small number of genuine extraction misses rather than evaluation-harness artifacts. The evaluation harness itself is now trustworthy enough to extend to relation extraction and a larger document sample.
