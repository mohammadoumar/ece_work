# Task 6: Sustainability-Inclusive Evaluation

**Status: Prototype done, verified run**

Instrumented Task 3's LoRA fine-tuning run with energy/carbon tracking (`codecarbon`), and reported compute-cost vs. accuracy tradeoffs.

Notebook: [`task6_sustainability.ipynb`](./task6_sustainability.ipynb)

## Setup

- Run under instrumentation: Task 3's exact recipe (base CLIP `ViT-B-32`, LoRA fine-tune on 200 RSICD images, 1 epoch).
- Three separate `codecarbon` `EmissionsTracker` stages — baseline (untuned) evaluation, LoRA training, post-fine-tune evaluation — so training cost and inference cost aren't conflated.
- Accuracy proxy: the same purity-aware retrieval harness from Task 3/4 (mean purity over the top-3 for 6 held-out queries).

## Results (verified run)

- `codecarbon` ran cleanly on Colab's GPU, auto-detecting the hardware with no fallback needed.
- Purity numbers exactly reproduced Task 3's (0.889 → 1.000, same seed/recipe) — confirms the instrumentation didn't change the underlying experiment.
- **Training cost: 3.3s, 0.0492 Wh, 0.0231 g CO2eq.** Mean single eval pass: 1.4s, 0.0127 Wh. Training was ~2.4x longer and ~3.9x more energy-intensive than one evaluation pass.
- Illustrative figure: **0.44 Wh per full purity point gained**, for the +0.111 improvement observed here — meaningful only as a same-notebook baseline, not a generalizable cost figure.
- The entire three-stage experiment consumed about **0.074 Wh and produced about 0.035 g CO2eq total** — negligible in absolute terms, because this is a toy-scale demonstration (1 epoch, 200 samples on a LoRA adapter), not a stand-in for a real training run's footprint.

## Caveats

- Colab's GPU is shared/virtualized, so `codecarbon`'s hardware-based power draw estimates are approximate, not lab-grade — useful for *relative* comparisons within this notebook, not as an absolute carbon figure to publish.
- Each stage was tracked once; no repeated trials to establish variance in the energy/duration measurements themselves.
- The "energy per purity point" ratio only makes sense near the specific purity range observed here (0.889 → 1.000) — it doesn't extrapolate to very different starting points.

## Open next steps

1. Scale up to a training run large enough to matter (more epochs, more data, or a full fine-tune instead of LoRA) and re-measure — current numbers are too small to represent realistic compute-cost tradeoffs.
2. Track emissions on Task 4's SLERP merge sweep and Task 5's 25-document benchmarking run too, to build a small library of "cost per experiment type" baselines.
3. If a non-shared GPU environment becomes available, repeat this measurement there for a more trustworthy absolute number.
4. Use this instrumentation pattern (the `track_stage` helper) as the standard way future experiments report compute cost, so efficiency targets have consistent, comparable inputs across tasks.

## Conclusion

The instrumentation works end-to-end and cleanly separates training cost from inference cost, reproducing Task 3's accuracy numbers exactly while adding real energy/CO2/duration figures alongside them. At this toy scale the absolute footprint is negligible, so the main value of this run is establishing the *pattern* — a reusable `track_stage` helper and training-vs-inference cost ratios — rather than a number worth quoting on its own.
