# Task 1: Geospatial Semantic Embedding (Multimodal)

**Status: Done**

Built and verified a minimal end-to-end multimodal geospatial embedding pipeline.

Notebook: [`task1_geospatial.ipynb`](./task1_geospatial.ipynb)

## Setup

- Dataset: [RSICD](https://huggingface.co/datasets/arampacha/rsicd) (Remote Sensing Image Captioning Dataset) — 8,734 train images, each with up to 5 captions.
- Model: [RemoteCLIP](https://huggingface.co/chendelong/RemoteCLIP) `ViT-B-32`, an OpenCLIP checkpoint fine-tuned on remote-sensing image-text pairs.
- Sanity checks: (1) text → image retrieval with 6 novel, hand-written queries against a category-balanced sample of images; (2) image → text retrieval, finding the nearest caption for a probe image from a bank of other images' captions.

## Results

- All 6 text → image queries correctly retrieve their matching category in the top 3 (bridge, airport, forest, farmland, stadium, residential) — the embedding space clearly separates these scene types.
- Image → text retrieval for an airport probe image pulled 4 of its top-5 nearest captions from other airport-category images; the one near-miss (a "parking lot" caption) is a sensible failure mode given the visual/semantic overlap with airport aprons.
- Sample stats: 30 real labeled categories found in RSICD filenames, 788 images correctly excluded as unlabeled, 150 images sampled (up to 5/category) into the retrieval bank.

## Notable issue found and fixed

RSICD filenames are inconsistent — only some follow a `category_number.jpg` convention (e.g. `airport_277.jpg`); most are plain numeric filenames (e.g. `00562.jpg`) with no category encoded in the name. An initial category-parsing function mis-treated every numeric filename as its own one-off "category," which silently broke the category-based retrieval check (e.g. made the bridge query look like a model failure when it was actually a labeling bug). Fixed by regex-filtering to only true `category_number.jpg` filenames and excluding/counting the rest separately.

## Open next steps

- Replace the qualitative eyeball checks with a quantitative metric (e.g. Recall@K) using RSICD's own image-caption pairs on the `test` split.
- Try a generic (non-domain-adapted) CLIP checkpoint on the same queries as a baseline, to quantify how much RemoteCLIP's remote-sensing fine-tuning actually helps — feeds into Task 3 (domain adaptation).
- Investigate whether the excluded numeric-filename images have category labels available elsewhere in the dataset.
