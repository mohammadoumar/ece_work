# ECE Work

Onboarding and working notes for this project.

## Project Focus

Designing, adapting, and evaluating multimodal artificial intelligence models for knowledge extraction and integration, while contributing to research, performance optimization, sustainability, and scientific valorization.

### Key Responsibilities

- Geospatial semantic embedding (multimodal).
- Document extraction and knowledge integration (knowledge graph/ontology).
- Domain adaptation/fine-tuning and model optimization.
- Research and engineering in model merging: implementation and evaluation of merging methods (e.g., DARE-TIES, SLERP-style interpolation, novel alternatives).
- Benchmarking and metrics: precision/F1 score for extraction, fidelity (GraphRAG-style evaluation), and measurable efficiency targets (computational reduction with environmental monitoring).
- Sustainability-inclusive evaluation.
- Scientific output: contribution to publications, open science (where feasible), and technology transfer with consortium partners.

## Files

- `work_plan/` — planning documents.
  - `ece_work_profile.txt` — the original work profile for this role/project.
  - `onboarding_task_suggestions.txt` — one suggested onboarding task per responsibility area, with a detailed step-by-step breakdown for Task 1 (geospatial semantic embedding), including the recommended starter dataset (RSICD) and pretrained model (RemoteCLIP).
- `tasks/` — one subdirectory per numbered task, holding that task's notebook(s), README, and deliverables.
  - [`task1/`](tasks/task1/README.md) — Geospatial semantic embedding (RSICD + RemoteCLIP).
  - [`task2/`](tasks/task2/README.md) — Document extraction and knowledge integration (DocRED + spaCy).
  - [`task3/`](tasks/task3/README.md) — Domain adaptation/fine-tuning (LoRA on CLIP, RSICD).
  - [`task4/`](tasks/task4/README.md) — Model merging research (SLERP on CLIP checkpoints).
  - [`task5/`](tasks/task5/README.md) — Benchmarking and metrics (precision/recall/F1 harness for Task 2's extraction pipeline).
  - [`task6/`](tasks/task6/README.md) — Sustainability-inclusive evaluation (codecarbon instrumentation of Task 3's LoRA fine-tune).
- `exploratory/` — general geospatial setup/exploration notebooks not tied to a specific numbered task: `colab_try.ipynb`, `leafmap.ipynb`, `raster_data.ipynb`, `sentinel-2.ipynb`, `setup_verification.ipynb`, `training_data.ipynb`.

## Task Summaries

### [Task 1: Geospatial Semantic Embedding](tasks/task1/README.md) — Done

Built and verified a minimal multimodal embedding pipeline (RSICD dataset + RemoteCLIP model), with working text↔image cross-modal retrieval. Found and fixed a filename-parsing bug that was silently corrupting the category-based sanity checks. See [`tasks/task1/README.md`](tasks/task1/README.md) for full setup, results, and next steps.

### [Task 2: Document Extraction and Knowledge Integration](tasks/task2/README.md) — Prototype done (entity extraction verified; relation extraction not yet attempted)

Built a prototype extracting entities from documents (DocRED dataset) with a spaCy NER baseline, compared against DocRED's gold, Wikidata-mapped entities (77% match on the probe document, counting all entity aliases). Found and fixed a dataset-loading break, a train/distant split-merging issue, and a gold-entity alias-blindness bug. See [`tasks/task2/README.md`](tasks/task2/README.md) for full setup, results, and next steps.

### [Task 3: Domain Adaptation/Fine-Tuning](tasks/task3/README.md) — Prototype done, verified run

LoRA fine-tuned generic CLIP on a tiny RSICD subset (200 images, 1 epoch) and compared it against base CLIP and RemoteCLIP on held-out retrieval queries. All three tied on a coarse hit-rate metric, but the fine-tuned model showed a real, measurable improvement in retrieval purity — revealing that the metric itself is too coarse to capture small adaptation effects. See [`tasks/task3/README.md`](tasks/task3/README.md) for full setup, results, and next steps.

### [Task 4: Model Merging Research](tasks/task4/README.md) — Prototype done, verified run

Implemented SLERP directly (mergekit lacks `open_clip` support) to merge the base CLIP and Task 3's LoRA fine-tuned CLIP, sweeping the interpolation factor. The `t=0`/`t=1` boundary check confirmed the merge is implemented correctly, and purity rose smoothly across the sweep — but since the two parents differ by only a few LoRA-adapted tensors, this run proves the merge mechanics work without yet stress-testing genuinely divergent parents. See [`tasks/task4/README.md`](tasks/task4/README.md) for full setup, results, and next steps.

### [Task 5: Benchmarking and Metrics](tasks/task5/README.md) — Prototype done, verified run

Scaled Task 2's single-document entity extraction comparison into a real precision/recall/F1 harness run across 25 DocRED documents. Reproduced Task 2's original number exactly (confirming correctness) while revealing it was actually a recall figure, not F1 — aggregate F1 came out to 0.788–0.790, with real per-document variance (F1 std 0.106) that a single probe document couldn't have shown. See [`tasks/task5/README.md`](tasks/task5/README.md) for full setup, results, and next steps.

### [Task 6: Sustainability-Inclusive Evaluation](tasks/task6/README.md) — Prototype done, verified run

Instrumented Task 3's LoRA fine-tuning run with `codecarbon`, tracking baseline eval, training, and post-fine-tune eval as separate emissions stages. Training was ~2.4x longer and ~3.9x more energy-intensive than a single eval pass; the whole toy-scale experiment used about 0.074 Wh total, establishing a reusable cost-tracking pattern rather than a publishable absolute figure. See [`tasks/task6/README.md`](tasks/task6/README.md) for full setup, results, and next steps.

## Status

All six onboarding tasks (Tasks 1–6) have a verified prototype (Task 1 complete; Tasks 2–6 prototype-complete with open refinements documented in each task's README). Onboarding phase is essentially done — future work would mean deepening one of these tasks (e.g. the open next-steps items) or moving to non-onboarding project work.
