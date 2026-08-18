# Task 3: Domain Adaptation / Fine-Tuning (LoRA)

**Status: Prototype done, verified run (small-scale LoRA fine-tune; evaluation metric found to be too coarse)**

Small-scale LoRA fine-tuning experiment on top of Task 1's setup, to establish a baseline domain-adaptation workflow.

Notebook: [`task3_domain_adaptation.ipynb`](./task3_domain_adaptation.ipynb)

## Setup

- Dataset: [RSICD](https://huggingface.co/datasets/arampacha/rsicd) — same as Task 1. 200 randomly sampled images (1 caption each) used for LoRA fine-tuning; a held-out category-balanced sample (150 images, 30 categories) that explicitly excludes those 200 images used for evaluation.
- Base model: generic CLIP `ViT-B-32` (`laion2b_s34b_b79k`) — the *not* domain-adapted checkpoint Task 1 skipped past in favor of RemoteCLIP.
- Fine-tuning: LoRA (`r=8`, targeting `out_proj`/`c_fc`/`c_proj` in both the vision and text towers — ~1.47M trainable params, 0.97% of the 152.75M total), 1 epoch, batch size 16 (13 batches), standard CLIP contrastive loss, AdamW at lr=1e-4.
- Comparison: base (untuned) CLIP vs. LoRA fine-tuned CLIP vs. RemoteCLIP (fully domain-adapted reference), all evaluated on the same 6 text→image queries used in Task 1.

## Results (verified run)

- Training loss was noisy but trended down over the single epoch (1.41 → 1.35, with per-batch swings up to 2.59) — expected for one epoch over 13 small batches.
- All three models scored **6/6** on the top-3 hit-rate metric (expected category present somewhere in the top 3 retrieved images) — this metric ceilings out too easily on this query set to separate the three models.
- The real signal is in *purity* of the top-3, not just presence: base CLIP had two queries where an off-category image intruded into the top 3 (`farmland` pulled in a `pond` image; `stadium` pulled in a `playground` image). After the 1-epoch LoRA fine-tune, **all 6 queries returned a perfectly pure top-3** — tighter than even RemoteCLIP, which still had one impure top-3 (`residential` mixed in a `sparseresidential` image alongside `denseresidential`).
- This matches what a tiny, targeted fine-tune should do: nudge the embedding space to separate these specific categories a bit more cleanly, without approaching the full domain adaptation RemoteCLIP represents (trained on far more remote-sensing image-text data).

## Notable issue found and fixed

- **`peft` import failure on Colab's preinstalled `torchao`.** `peft` requires `torchao>=0.16.0` at import time (even though this notebook never uses quantization); Colab ships `0.10.0`, causing `ImportError: Found an incompatible version of torchao`. Fixed by adding `%pip install -U torchao` to the setup cell.

## Limitation found: the hit-rate metric is too coarse

The pass/fail "expected category in top-3" scoring, carried over from Task 1's qualitative eyeball check, saturates immediately — all three models get the right general category almost every time on 6 hand-picked queries. It could not detect the purity improvement above; that had to be read off the raw category lists by hand. A finer metric (top-1 exact match, or a purity/fraction-correct score over top-k) would surface this kind of effect directly, and would matter more once the fine-tuning subset or epoch count is scaled up.

## Open next steps

1. Replace the top-3 hit-rate metric with something finer-grained (top-1 exact match, or mean purity over top-k) so effects like the one found here don't require manually reading the raw category lists.
2. Scale up the experiment (more fine-tuning samples, more than 1 epoch) now that the workflow is verified end-to-end, and check whether the purity gap to RemoteCLIP continues to close.
3. Try a larger LoRA rank or different target modules (e.g. also adapting the projection layers) to see whether that changes the adaptation rate.
4. Connects to Task 5 (benchmarking/metrics): the same "metric ceilings out, real signal needs a finer score" pattern seen here is a good case study for why the project needs a proper quantitative harness rather than small hand-picked probe sets.

## Conclusion

The LoRA fine-tuning workflow works end-to-end on a tiny budget (200 samples, 1 epoch) and produces a small, measurable improvement in retrieval purity — a genuine if modest domain-adaptation signal — but the current evaluation harness is too blunt to quantify it directly, which is itself a useful finding for scoping Task 5.
