# Task 2: Document Extraction and Knowledge Integration (Knowledge Graph/Ontology)

**Status: Prototype done, evaluation harness has known limitations (see below)**

Prototype pipeline for extracting entities/relations from documents and mapping them onto a knowledge graph schema.

Notebook: [`task2_extraction.ipynb`](./task2_extraction.ipynb)

## Setup

- Dataset: [DocRED](https://huggingface.co/datasets/thunlp/docred), loaded from the `refs/convert/parquet` revision (the dataset's legacy loading script is no longer supported by current `datasets` versions). The converted `train` split merges the original `train_annotated` (3,053 gold docs) and `train_distant` (101,873 weakly-supervised docs) into one split — explicitly sliced to the first 3,053 rows to guarantee gold-quality annotations.
- Baseline extractor: spaCy `en_core_web_sm` NER, run on the raw document text (ignoring DocRED's gold labels), to simulate "we don't have labels yet."
- Probe document: "AirAsia Zest" — 17 gold entities, 13 gold relations (Wikidata property IDs, e.g. `P17` = country, `P131` = located in the administrative territorial entity).

## Results

- spaCy correctly matched **14/17 (82%)** gold entities once the comparison was normalized to strip tokenization artifacts (space-before-punctuation from the token-joined text, leading articles, trailing possessives) — the raw exact-string match had understated this at 12/17.
- 3 genuine misses: `metro manila` (present in the text, simply untagged by spaCy), `asian spirit` (spaCy tagged only the `Asian` fragment as NORP, never joined it into the full org name), and `a year` vs. spaCy's wider span `less than a year` (a boundary mismatch our normalization doesn't cover).
- spaCy also surfaced entities with no counterpart in our gold-name list (`caap`, `airasia zest`, `government of the republic of the philippines`, plus date/ordinal noise like `first`, `august 16, 2013`).

## Notable issues found

- **Legacy dataset-loading script no longer works.** `thunlp/docred` ships a Python loading script, which recent `datasets` versions (v4.0+) refuse to execute — `trust_remote_code=True` no longer helps. Fixed by loading from the auto-generated `revision="refs/convert/parquet"` instead.
- **Split ambiguity after parquet conversion.** The conversion merges `train_annotated` (gold) and `train_distant` (noisy, weakly-supervised) into a single `train` split (104,926 = 3,053 + 101,873 rows). Any code sampling from `train` without explicitly slicing to the first 3,053 rows risks silently pulling noisy distant-supervision data. Fixed by explicit slicing with a comment.
- **Gold-entity list is incomplete (open, not yet fixed).** `gold_entities` was built from only the *first* mention name per DocRED entity (`vertexSet[i][0]`), but each gold entity typically has multiple aliases across its mentions in the document. `caap` and `airasia zest` are very likely legitimate aliases of entities already in the gold set (Civil Aviation Authority of the Philippines, and the document's own subject) rather than spaCy false positives — but the current comparison can't tell the difference, because alias mentions were discarded when building the comparison set. This means the "no gold match" list currently conflates real coverage gaps with an artifact of the extraction code, and the true false-positive rate is likely lower than it appears.

## Relation extraction was not attempted in this pass

Only entity extraction was benchmarked. spaCy has no built-in relation extraction; testing that would require either a separate relation-extraction model/prompt-based LLM approach, or dependency-parse heuristics.

## Open next steps

1. Rebuild `gold_entities` from *all* mention names per entity (not just the first) and re-run the comparison — expected to raise the effective match rate and shrink the "no gold match" list.
2. Extend past entity extraction into relation extraction (e.g. prompt an LLM to output `(head, relation, tail)` triples, or use a dependency-parse heuristic) and compare against the 13 gold Wikidata relations.
3. Scale from a single probe document to a small sample (e.g. 20–30 docs from `train_annotated`) to get an aggregate precision/recall number instead of one anecdotal case — this feeds directly into Task 5 (benchmarking/metrics).
4. Once schema coverage gaps are more reliably identified (post alias-fix), log them explicitly as candidate additions to the project's own ontology, rather than treating every non-match as noise.

## Conclusion

The prototype pipeline works end-to-end — dataset load, gold inspection, baseline extraction, and comparison against a real Wikidata-based schema — and spaCy's out-of-the-box NER already recovers a solid majority of gold entities on this sample. The main limitation is in the *evaluation harness* itself (single-mention gold names, no relation extraction, single-document sample), not yet in the extraction quality — those are the priorities before this can be trusted as a real benchmark.
