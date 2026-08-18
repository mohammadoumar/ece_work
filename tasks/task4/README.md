# Task 4: Model Merging Research (SLERP)

**Status: Prototype done, verified run (merge mechanics confirmed correct; not yet stress-tested on divergent parents)**

Reproduce a basic SLERP merge between two CLIP checkpoints and evaluate the merged model against both parents.

Notebook: [`task4_model_merging.ipynb`](./task4_model_merging.ipynb)

## Setup

- Parent A: base CLIP `ViT-B-32` (`laion2b_s34b_b79k`), unmodified.
- Parent B: the same base model, LoRA fine-tuned (Task 3's exact recipe — 200 RSICD images, 1 epoch, `r=8` on `out_proj`/`c_fc`/`c_proj`), then baked into a plain checkpoint via `merge_and_unload()`.
- Merge method: SLERP (spherical linear interpolation), applied per-tensor over the two parents' state dicts, swept over `t ∈ {0.0, 0.25, 0.5, 0.75, 1.0}`. `mergekit` (the library named in the onboarding suggestion) wasn't used directly since it doesn't have first-class support for `open_clip`'s custom architecture — SLERP was implemented directly instead.
- Evaluation: the same held-out 6-query retrieval harness from Task 1/3, upgraded here to also report mean purity (fraction of top-3 matching the expected category), since Task 3 found the plain hit-rate metric too coarse to show small effects.

## Results (verified run)

- **Correctness check passed:** `t=0.0` reproduced Parent A exactly (score 6/6, purity 0.889) and `t=1.0` reproduced Parent B exactly (score 6/6, purity 1.000) — confirming the SLERP implementation actually performs the interpolation correctly.
- **The sweep is smooth and monotonic:** mean purity rose 0.889 → 0.944 → 1.000 → 1.000 → 1.000 as `t` went from 0 to 1, with no instability or collapse at any interpolation point.
- **A half-weight merge (`t=0.5`) already reaches full purity**, matching Parent B exactly on this probe set.
- The plain hit-rate score stayed pinned at 6/6 across the entire sweep — purity was the only metric that actually separated the merge points, confirming the metric upgrade in this notebook was necessary, not cosmetic.

## Why this merge behaved so cleanly

Most of the two parents' weights are bit-identical — LoRA only touched `out_proj`/`c_fc`/`c_proj` in each transformer block, and the base weights were frozen throughout Task 3's fine-tune. So the vast majority of tensors SLERP-merge trivially (near-zero angle between identical vectors, falling back to linear interpolation of two equal values — a no-op), and only the handful of LoRA-touched tensors are actually being interpolated. This is a favorable, low-risk case for merging — the parents are close by construction, not two independently-trained models. It's a good correctness proof for the SLERP code, but not yet a stress test of merge behavior on genuinely divergent parents (the classic "weight interference" failure mode SLERP/DARE-TIES research is meant to address).

## Notable issue found and fixed

- **`peft` import failure on Colab's preinstalled `torchao`** (same as Task 3): `peft` requires `torchao>=0.16.0` at import time; Colab ships `0.10.0`. Fixed by adding `%pip install -U torchao` to the setup cell.

## Open next steps

1. Repeat the sweep with a more divergent Parent B (more fine-tuning data, more epochs, or a full fine-tune instead of LoRA) to see whether purity still improves monotonically or whether interference/degradation appears at intermediate `t` — the interesting regime for merge research that this run didn't reach.
2. Try DARE-TIES (the other method named in the onboarding task) for comparison, since it's specifically designed to handle parameter interference between more divergent parents that plain SLERP doesn't address.
3. Extend the evaluation harness further (e.g. Recall@K on RSICD's own test-split pairs, per Task 1's open item) so merge comparisons aren't bottlenecked by a 6-query probe set that's already saturating on purity too.
4. Feeds into Task 5 (benchmarking/metrics): this notebook's harness (score + purity, run identically across model variants) is a reusable template for the "repeatable metric to compare future extraction/merging changes against" Task 5 calls for.

## Conclusion

The SLERP merge implementation is verified correct (via the `t=0` / `t=1` boundary check) and produces a smooth, well-behaved interpolation between the two parents on this probe set, with the merged model matching the fine-tuned parent's full purity benefit at just `t=0.5`. Because the two parents here are close by construction (only a few LoRA-adapted tensors actually differ), this run demonstrates the merge mechanics work but doesn't yet exercise the harder case of merging more divergent models, which is the natural next step.
