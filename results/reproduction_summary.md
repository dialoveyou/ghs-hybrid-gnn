# Legacy pipeline run

This directory documents one complete end-to-end execution of the **original** pipeline in `src/`
on CPU (2 cores, 8 GB RAM, no GPU). It is kept because it produced the trained weights in
`checkpoints/`, and because it is the run whose evaluation protocol the paper examines.

**The numbers reported in the paper do not come from here.** They come from the scripts in
`experiments/`, which re-run the same models on the same data under a threshold protocol applied
identically to every model. See the top-level README.

Total wall-clock: about 1 h 45 min, dominated by the baseline Optuna search (~45 min) and the
five-fold cross-validation (~30 min).

## Deterministic stages — these reproduce exactly

Seeds are fixed and the multi-label stratified and scaffold splits are deterministic, so these
match on any machine, and they match the paper:

| Quantity | This run | Paper |
|---|---|---|
| Compounds with at least one GHS H-code | 14,148 | 14,148 |
| Successful ETKDGv3 3D embeddings | 14,145 (0.02% failure) | 14,145 (0.02% failure) |
| Random split train / val / test | 9,061 / 2,281 / 2,803 | 9,061 / 2,281 / 2,803 |
| Classes receiving conformer augmentation | 13 | 13 |
| Unique Bemis–Murcko scaffolds | 8,114 | 8,114 |
| Scaffold-split train / test | 9,742 / 4,403 | 9,742 / 4,403 |

## What this run produced

`metrics.json` holds the full dump. On the random split:

| Model | Macro-F1 | Macro-MCC | Macro-AUC | Balanced Acc. |
|---|---|---|---|---|
| Baseline (2D tree ensemble) | 0.573 | 0.514 ± 0.157 | 0.885 | 0.708 |
| GNN-ECC (2D–3D hybrid) | 0.575 | 0.438 ± 0.174 | 0.838 | 0.720 |
| Stacked (GNN + baseline) | 0.639 | 0.547 ± 0.141 | 0.886 | 0.753 |

Y-randomization gives 0.113 ± 0.002 against the true-label baseline, confirming that performance
reflects genuine structure–hazard relationships rather than model capacity or evaluation leakage.
SHAP attribution on the LightGBM component ranks LogP and molar refractivity top among the
physicochemical descriptors, with fingerprint bits dominating descriptors in every class examined.

## Why these are not the paper's headline numbers

The comparison built into `src/run_stage2.py` scores the tree-ensemble baseline at a **fixed 0.5
threshold** (`run_stage2.py:240`) while the GNN receives per-class thresholds tuned on validation
(`run_stage2.py:413`) and the stacked model receives both a per-class blend weight and a per-class
threshold. The baseline → stacked gap of +0.066 measured that way is not a like-for-like
comparison: re-scoring the identical fitted baseline under the protocol the other models received
lifts it from 0.561 to 0.626, and the stacked model's advantage over it then falls to +0.013
(p = 0.07).

`experiments/p0_baseline.py` performs that re-scoring; `experiments/p0_stacked.py` reconstructs the
stacked model from the checkpoint in `checkpoints/` and runs a paired bootstrap against each
baseline variant; `experiments/ablation.py` isolates the contribution of the 3D branch. Those three
scripts are the source of the paper's results.

Four further scripts test whether those results survive the choices made along the way:
`expA_multiseed.py` (five random seeds per configuration), `expBDE_analyses.py` (threshold
stability, the unobserved-H-code-as-negative assumption, duplicate structures and train/test
chemical-space overlap), `expC_chainorder.py` (five classifier-chain label orderings) and
`expF_nestedcv.py` (the whole selection procedure repeated inside five outer folds). The nested
estimates are the ones the paper treats as primary.
