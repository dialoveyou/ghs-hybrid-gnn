# Reference run

A complete end-to-end execution of this repository on CPU only (2 cores, 8 GB RAM, no GPU),
using `data/ghs_modeling_dataset_14148.csv`. Every number below comes from that single run and
is reproduced in full in `metrics.json`. This is the run described in §4.9 of the paper and the
source of the weights in `checkpoints/`.

Total wall-clock: about 1 h 45 min, dominated by the baseline Optuna search (~45 min) and the
5-fold cross-validation (~30 min). Both shorten substantially with more cores; a GPU mainly
speeds up GNN training (~6 min here).

## Deterministic stages — exact match with the paper

Seeds are fixed and the multi-label stratified and scaffold splits are deterministic, so these
reproduce exactly on any machine:

| Quantity | This run | Paper |
|---|---|---|
| Compounds with at least one GHS H-code | 14,148 | 14,148 |
| Successful ETKDGv3 3D embeddings | 14,145 (0.02% failure) | 14,145 (0.02% failure) |
| Classes receiving conformer augmentation | 13 | 13 |
| Unique Bemis–Murcko scaffolds | 8,114 | 8,114 |
| Scaffold-split train / test | 9,742 / 4,403 | 9,742 / 4,403 |

Matching on all five confirms that the 22-class label mapping, the 3D embedding pipeline, the
augmentation selection rule and the scaffold-splitting logic behave as described in §2.

## Metrics from this run

Random 64/16/20 split (train 9,061 / val 2,281 / test 2,803).

| Model | Macro-F1 | Macro-MCC | Macro-AUC | Balanced Acc. |
|---|---|---|---|---|
| Baseline (2D tree ensemble) | 0.573 | 0.514 ± 0.157 | 0.885 | 0.708 |
| GNN-ECC (2D–3D hybrid) | 0.575 | 0.438 ± 0.174 | 0.838 | 0.720 |
| **Stacked (GNN + baseline)** | **0.639** | **0.547 ± 0.141** | **0.886** | **0.753** |

Other protocols: 5-fold CV for the baseline, 0.581 ± 0.016 (folds 0.572, 0.575, 0.611, 0.579,
0.566). Scaffold-disjoint baseline, 0.454 — an 11.9-point drop from the random split, reproducing
the paper's central finding that structure-based partitioning removes a large share of
random-split performance.

Bootstrap significance (1,000 resamples):

| Comparison | ΔF1 | p | Significant |
|---|---|---|---|
| Baseline → GNN-ECC | +0.002 | 0.846 | No |
| Baseline → Stacked | +0.066 | < 0.001 | Yes |
| GNN-ECC → Stacked | +0.064 | < 0.001 | Yes |

SHAP attribution on the LightGBM component reproduces the paper's pattern — fingerprint bits
dominate descriptors in every class examined, and LogP and molar refractivity rank top among
descriptors:

| Class | Fingerprint vs descriptors | Top descriptor |
|---|---|---|
| Eye damage / irritation | 2.49 vs 0.94 | LogP (0.247) |
| Mutagenicity | 4.46 vs 1.21 | MolMR (0.229) |
| Corrosive to metals | 6.54 vs 1.48 | MolMR (0.341) |

## Relationship to the published values

The paper reports macro-F1 of 0.618 (baseline), 0.602 (GNN-ECC) and 0.660 (stacked) from a GPU
run. This CPU-only run lands 0.01–0.05 lower on every metric while preserving the ordering and
every qualitative conclusion: stacking significantly beats both constituent models, the 3D branch
is not significantly better than the 2D baseline on its own, scaffold splitting costs a large
number of macro-F1 points, and SHAP rankings are unchanged.

One difference worth naming: the paper finds the standalone GNN-ECC *significantly worse* than
the baseline (Δ = −0.017, p = 0.176 — not significant, but negative in sign), whereas this run
finds the two statistically indistinguishable with a marginally positive sign (Δ = +0.002,
p = 0.846). Both support the paper's claim that the 3D branch is not a superior standalone
predictor; the sign of a difference this small is not stable across runs. The conclusion that
depends on it — complementarity rather than 3D superiority — holds in both.

Compare metrics **within** a single run rather than against Table 2 line by line.
