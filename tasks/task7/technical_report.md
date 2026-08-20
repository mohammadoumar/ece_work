# Onboarding Technical Report: Multimodal AI for Knowledge Extraction and Integration

**Author:** Umer Mushtaq
**Status:** Onboarding phase complete (Tasks 1–6 verified)

## Abstract

This report summarizes a six-task onboarding phase covering the core responsibility areas of the project: multimodal geospatial embedding, document-level knowledge extraction, domain adaptation, model merging, benchmarking, and sustainability-inclusive evaluation. Rather than six unrelated exercises, the tasks form a connected pipeline: a base multimodal model (Task 1) is domain-adapted (Task 3), the adapted and base checkpoints are merged (Task 4), an extraction pipeline is built and scored with a real precision/recall/F1 harness (Tasks 2 and 5), and one of the training runs is instrumented for energy/carbon cost (Task 6). All work uses public datasets and models and runs at toy scale — the goal was to establish working, verified methodology in each area, not to produce publication-scale results. This report consolidates methodology, cross-task findings, limitations, and open science notes across all six tasks, and organizes each task's future-work items by theme.

## 1. Methodology per Task

### Task 1 — Geospatial Semantic Embedding
Built a minimal cross-modal retrieval pipeline using [RSICD](https://huggingface.co/datasets/arampacha/rsicd) (remote-sensing images with up to 5 captions each) and [RemoteCLIP](https://huggingface.co/chendelong/RemoteCLIP) `ViT-B-32`, a CLIP variant fine-tuned on remote-sensing image-text pairs. Verified with two sanity checks: text→image retrieval (6 hand-written queries against a category-balanced image sample) and image→text retrieval (nearest captions for a probe image). Found and fixed a filename-parsing bug where images without a `category_number.jpg`-style filename were silently corrupting the category-based sample.

### Task 2 — Document Extraction and Knowledge Integration
Built a spaCy `en_core_web_sm` NER baseline against [DocRED](https://huggingface.co/datasets/thunlp/docred)'s gold, Wikidata-mapped entities on a single probe document ("AirAsia Zest"). Fixed three issues along the way: DocRED's legacy dataset-loading script no longer works under current `datasets` versions (worked around via the `refs/convert/parquet` revision); the converted `train` split silently merges gold-annotated and weakly-supervised documents (fixed via explicit slicing to the first 3,053 rows); and the initial gold-entity comparison only used each entity's first mention name, undercounting legitimate alias matches (fixed by comparing against all aliases).

### Task 3 — Domain Adaptation / Fine-Tuning (LoRA)
LoRA fine-tuned generic CLIP `ViT-B-32` (`laion2b_s34b_b79k`) on a small RSICD subset (200 images, 1 epoch, `r=8` targeting the attention output projection and MLP layers — ~1.47M trainable params, under 1% of the model). Compared base CLIP, the fine-tuned model, and RemoteCLIP on the same 6 retrieval queries from Task 1. All three tied on a coarse hit-rate metric; a finer purity metric (fraction of top-3 results actually matching the expected category) showed the fine-tune measurably tightened retrieval (0.889 → 1.000 mean purity).

### Task 4 — Model Merging Research (SLERP)
Implemented spherical linear interpolation (SLERP) directly between base CLIP and Task 3's LoRA fine-tune (`mergekit` was not used, as it lacks first-class support for `open_clip`'s architecture). Verified correctness via the boundary condition (`t=0`/`t=1` exactly reproduce each parent) and swept the interpolation factor `t ∈ {0, 0.25, 0.5, 0.75, 1.0}`, finding a smooth, monotonic purity improvement with no instability at any point — a half-weight merge (`t=0.5`) already matched the fully fine-tuned parent's purity.

### Task 5 — Benchmarking and Metrics
Generalized Task 2's single-document comparison into a proper precision/recall/F1 harness (true positives/false positives/false negatives, not just a match count) run across 25 DocRED documents, with both micro (pooled counts) and macro (mean of per-document scores) aggregation. Reproduced Task 2's original document exactly as a correctness check, and found aggregate F1 of 0.788–0.790 with substantial per-document variance (F1 std 0.106, range 0.53–1.00).

### Task 6 — Sustainability-Inclusive Evaluation
Instrumented Task 3's exact LoRA fine-tuning recipe with `codecarbon`, tracking baseline evaluation, training, and post-fine-tune evaluation as three separate emissions stages. Training measured 3.3s / 0.0492 Wh / 0.0231 g CO2eq, roughly 2.4× longer and 3.9× more energy-intensive than a single evaluation pass. The full three-stage experiment totaled about 0.074 Wh — negligible in absolute terms, consistent with the toy scale of the underlying run.

## 2. Cross-Task Findings

The six tasks were not run in isolation; several findings only become visible when read across tasks:

- **The domain-adaptation pipeline is literally connected end-to-end.** Task 1 selected the dataset and reference model; Task 3 fine-tuned a second model on the same data and eval harness; Task 4 merged Task 3's fine-tuned checkpoint back with the Task 1/3 base model; Task 6 instrumented the exact same Task 3 training run for energy cost. A change to any one of these (e.g. a different fine-tuning recipe) would propagate cleanly through the others because they share the same seeded dataset split and evaluation harness.
- **The "metric ceilings out" problem recurred and was fixed once, generally.** Task 3 discovered that a coarse top-3 hit-rate metric couldn't distinguish base CLIP from a fine-tuned or fully domain-adapted model — all three scored identically. The fix (a purity metric) was reused as-is in Task 4's merge sweep, where it was the only metric that showed the sweep's effect at all. Task 5 found essentially the same class of problem from the extraction side: Task 2's original "77% match" turned out to be recall only, conflating two different failure modes (false positives and false negatives) into one number. In both modalities, the first version of "did it work" was too blunt to answer the actual question, and had to be replaced with a metric that separated precision-like and recall-like effects.
- **Single-example evaluation is unreliable, and this was demonstrated twice.** Task 5's 25-document sample showed real document-to-document F1 variance (std 0.106, range 0.53–1.00) around the same 0.79 mean that Task 2's single probe document happened to land on — meaning Task 2's original number was representative by coincidence, not because one document is a valid sample size. Task 6 makes the same point implicitly for compute cost: a single tracked run of each stage has no visibility into run-to-run variance, which is flagged as an open gap rather than assumed away.
- **A close-parents merge is not yet a stress test.** Task 4's SLERP merge behaved smoothly because the two parents were structurally almost identical (LoRA touches under 1% of parameters, so most tensors merge trivially as no-ops). This is a legitimate correctness demonstration for the SLERP implementation, but the harder case — merging genuinely divergent models, where weight interference is a known failure mode — has not yet been exercised by any task.
- **All quantitative pipelines built in Tasks 3–6 share the same seeded RSICD split**, so their outputs are directly comparable without re-deriving anything: Task 3's purity numbers reappear unchanged in Task 4 (as the `t=1` boundary) and in Task 6 (as the post-fine-tune eval), which also served as a useful correctness cross-check each time.

## 3. Limitations

- **Toy scale throughout.** Fine-tuning used 200 images and 1 epoch (Task 3/4/6); extraction benchmarking used 25 documents (Task 5); retrieval sanity checks used 6 hand-written queries (Tasks 1/3/4). None of these are large enough to support strong quantitative claims — they establish working methodology, not final numbers.
- **Approximate compute-cost measurements.** Task 6 ran on a shared/virtualized Colab GPU, so `codecarbon`'s hardware-based power estimates are directional, not lab-grade. They are meaningful as relative comparisons (training vs. inference) within that notebook, not as publishable absolute figures.
- **Merge research is unstressed.** Task 4 confirmed the SLERP implementation is correct but only tested it on two very similar parent models. Its real usefulness for interference-prone merges (the motivating case for methods like DARE-TIES) is untested.
- **Relation extraction is entirely unaddressed.** Task 2's pipeline (and Task 5's benchmark of it) covers entity extraction only; DocRED's actual relation-level gold labels (13 Wikidata relations in the probe document) have never been compared against any extractor output.
- **Evaluation harnesses were built to answer the immediate question at hand**, not designed as general-purpose infrastructure — e.g. the retrieval purity metric and the extraction precision/recall/F1 harness are separate, task-specific code paths rather than a shared benchmarking library, despite both existing to solve the same underlying problem (Task 5's stated goal).

## 4. Open Science Notes

Everything in this onboarding phase uses public, freely available resources:

- **Datasets:** [RSICD](https://huggingface.co/datasets/arampacha/rsicd) (Tasks 1, 3, 4, 6) and [DocRED](https://huggingface.co/datasets/thunlp/docred) (Tasks 2, 5), both open on the Hugging Face Hub.
- **Models:** generic CLIP `ViT-B-32` (`laion2b_s34b_b79k`, OpenCLIP) and [RemoteCLIP](https://huggingface.co/chendelong/RemoteCLIP), both open-weight.
- **Libraries:** `open_clip`, `peft` (LoRA), `spaCy`, `datasets`, `codecarbon` — all open-source.

Nothing in Tasks 1–6 depends on project-internal data, proprietary models, or consortium-partner assets, so the notebooks and findings as written are shareable outside the project with no redaction needed. This makes the six task notebooks themselves reasonable candidates for an internal methods writeup or an external blog-style post on the onboarding approach, if the project wants a lightweight open-science artifact beyond this report.

## 5. Future Directions (organized by theme)

**Metric infrastructure** *(Task 5's stated purpose, not yet fully realized)*
- Unify the retrieval purity harness (Tasks 3/4) and the extraction precision/recall/F1 harness (Tasks 2/5) into one reusable scoring library, rather than parallel task-specific implementations.
- Extend Task 1's qualitative retrieval checks into a real Recall@K metric on RSICD's own test-split pairs.
- Scale Task 5's benchmark beyond 25 documents to tighten the confidence interval on aggregate F1.

**Domain adaptation and merging**
- Scale Task 3's fine-tune (more data, more epochs, or full fine-tuning) and re-run Task 4's merge sweep on the resulting, more-divergent parent to actually exercise weight-interference behavior.
- Try DARE-TIES as a second merge method for comparison against SLERP, per the original Task 4 scope.
- Try a larger LoRA rank or additional target modules in Task 3 to see whether the adaptation effect grows accordingly.

**Extraction and knowledge integration**
- Extend Task 2/5's pipeline to relation extraction, scoring against DocRED's gold Wikidata relations with the same TP/FP/FN pattern already established for entities.
- Investigate the worst-performing document from Task 5's sample (F1 = 0.53) as a concrete case study in extractor failure modes.

**Sustainability**
- Re-run Task 6's instrumentation on a larger, realistic training run (not just the toy 200-sample/1-epoch case) to get compute-cost numbers that actually inform efficiency targets.
- Apply the same `codecarbon` tracking pattern to Task 4's merge sweep and Task 5's benchmarking run, to build a small library of "cost per experiment type" baselines across the project's own workflows.

## Conclusion

Six onboarding tasks, spanning all core responsibility areas of the role, now have verified, working prototypes built entirely on public data and models. The tasks are not isolated exercises: they share datasets, evaluation harnesses, and even specific model checkpoints across task boundaries, and several of the most useful findings (the recurring "metric too coarse" problem, the risk of single-example evaluation, the unstressed merge case) only emerge from reading the tasks together, as this report attempts to do. The clearest next step is not a new task area but deepening the existing six — particularly building the shared metric infrastructure Task 5 was meant to establish and scaling the domain-adaptation/merging pipeline (Tasks 3/4) to a regime where the interesting failure modes actually appear.
