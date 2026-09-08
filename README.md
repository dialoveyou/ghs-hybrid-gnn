# Threshold Calibration vs. 3D Structure in Multi-Label GHS Hazard Classification

Code, data, trained weights and experiment scripts for:

> Kim, D.; Jung, S.-h. *Per-Class Threshold Calibration Outweighs 3D Structural Information in
> Severely Imbalanced Multi-Label GHS Hazard Classification.* ACS Omega (submitted).

The task is predicting all **22 consolidated GHS hazard classes simultaneously** from molecular
structure, on a regulatory-scale inventory at its native 341-fold class imbalance. The question the
paper asks is not "does a 3D graph neural network help?" but **which modelling choices actually
govern performance** once everything else is held fixed.

## The central result

Holding the fitted model, the data and the partition constant, and changing only the decision rule:

| 2D tree ensemble, evaluated under | Macro-F1 |
|---|---|
| a fixed 0.5 threshold | 0.561 |
| per-class thresholds calibrated on a held-out validation set | **0.626** |

That is **+0.066** for free. The same contrast holds under five-fold cross-validation
(0.567 ± 0.017 → 0.644 ± 0.015) and is largest exactly where generalization is hardest, under a
Murcko-scaffold-disjoint split (0.453 → 0.539, **+0.086**).

Against that, changing the molecular representation buys very little. With every model calibrated
identically and trained on an identical sample set:

| Model | Macro-F1 (calibrated) |
|---|---|
| 2D tree ensemble | 0.626 |
| 2D-only MLP-ECC | 0.586 |
| 2D–3D hybrid GNN-ECC | 0.578 |
| 3D-only GNN-ECC | 0.396 |
| Stacked: tree + 3D-only | 0.627 |
| Stacked: tree + 2D-only | 0.634 |
| **Stacked: tree + 2D–3D hybrid** | **0.637** |

A 3D-only model reaches 0.396 against 0.626 for 2D fingerprints, and adding the 3D branch to a
2D-only neural model makes it slightly *worse*. Inside the stack, the increment attributable to the
3D branch is **+0.003 (paired bootstrap, p = 0.51)** — indistinguishable from zero. The residual
+0.010 of the best stack over the calibrated tree ensemble is model-family diversity, not 3D
information.

### Why this is easy to get wrong

Comparing the stacked model against the tree ensemble **at a fixed 0.5 threshold** — a common and
superficially reasonable choice — measures **+0.067, p < 0.001**, which looks like strong evidence
of 2D–3D complementarity. Almost all of it is the calibration the stacked model received and the
baseline did not. `experiments/p0_baseline.py` and `experiments/p0_stacked.py` reproduce both the
confounded and the matched comparison side by side.

---

## Repository layout

```
src/
  ghs_full_pipeline.py     single-file reference implementation
  run_full_pipeline.py     Stage 1 — data loading, 22-class labelling, 2D/3D feature extraction
  run_stage2.py            Stage 2 — splits, augmentation, baseline + GNN training, stacking
  run_stage3.py            Stage 3 — 5-fold CV, scaffold split, bootstrap tests, SHAP
  analysis/                per-endpoint metrics for the Fuadah-overlapping comparison
experiments/               the analyses reported in the paper
  features.py              22-class labels, 916-d 2D features, 3D conformer graphs (cached)
  p0_baseline.py           tree ensemble under four fit/threshold protocols
  p0_stacked.py            reconstructs the stacked model, paired bootstrap vs each baseline
  ablation.py              2D-only / 3D-only / 2D–3D, one split, one protocol, no augmentation
  abl_boot.py              paired bootstrap for the ablation contrasts
  stage3.py                scaffold-disjoint split and 5-fold CV with calibrated thresholds
  recompute_tables.py      per-class tables, low-support CIs, head-to-head endpoints
  make_figures.py          Figures 4 and 5
  *.json / *.log           results and run logs for every script above
notebooks/
  GHS_Hybrid_GNN_ECC.ipynb end-to-end notebook version (Colab-ready)
data/
  ghs_modeling_dataset_14148.csv  14,148 curated compounds with GHS H-codes
checkpoints/
  hybrid_gnn_ecc_ghs22.pt  trained hybrid GNN-ECC weights
results/
  metrics.json             full metric dump from a complete end-to-end run
  reproduction_summary.md  results of the reference run
```

**`src/` and `experiments/` are not interchangeable.** `src/` is the original pipeline; the
comparison built into `src/run_stage2.py` scores the tree-ensemble baseline at a fixed 0.5
threshold while giving the GNN and the stacked model per-class tuned thresholds, so running it
end to end reproduces the *confounded* +0.066 gap rather than the paper's numbers. `experiments/`
re-scores the same fitted models under one protocol applied to all of them, and is what the paper
reports. `results/reproduction_summary.md` documents the legacy run and the exact lines where the
protocols diverge.

## Installation

Python 3.12 is recommended.

```bash
git clone https://github.com/dialoveyou/ghs-hybrid-gnn.git
cd ghs-hybrid-gnn
pip install -r requirements.txt
```

`torch-geometric` may need a wheel matching your CUDA/PyTorch build; see the
[PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).
A GPU is optional — everything reported in the paper was produced on CPU.

## Reproducing the paper

```bash
python experiments/features.py        # ~6 min, caches features.pkl
python experiments/p0_baseline.py     # tree ensemble under four protocols  (~8 min)
python experiments/ablation.py        # the three representations + stacks  (~5 min)
python experiments/abl_boot.py        # paired bootstrap                    (~4 min)
python experiments/stage3.py          # scaffold split + 5-fold CV          (~25 min)
python experiments/recompute_tables.py
python experiments/make_figures.py
```

Every script writes a `*_results.json` next to itself; those files are the source of every number
in the paper.

## Reproducibility

Seeds are fixed (`SEED = 42`) and the splits are deterministic, so the data stages reproduce
exactly:

| Quantity | This code | Reported in paper |
|---|---|---|
| Compounds with GHS labels | 14,148 | 14,148 |
| Successful 3D embeddings | 14,145 (0.02% failure) | 14,145 (0.02% failure) |
| Random split train/val/test | 9,061 / 2,281 / 2,803 | 9,061 / 2,281 / 2,803 |
| Unique Murcko scaffolds | 8,114 | 8,114 |
| Scaffold split train/test | 9,742 / 4,403 | 9,742 / 4,403 |

Model metrics carry ordinary run-to-run variance of roughly ±0.01–0.02 macro-F1 from GNN
initialisation and sampler order; the tree-ensemble numbers are deterministic given the split. The
qualitative conclusions — that calibration is worth several times more than representation, and
that the 3D increment is indistinguishable from zero — do not depend on that variance.

## Data

`data/ghs_modeling_dataset_14148.csv` — 14,148 single-substance chemicals with CAS number, PubChem
CID, canonical SMILES, physicochemical properties, NFPA ratings where available, and GHS H-codes.
Assembled from the KOSHA chemical database and the Dangjin City industrial chemical usage registry,
with structures retrieved through the PubChem PUG REST API. See `data/README.md` for the column
schema and the 73 H-code → 22 class mapping.

Note that unobserved H-codes are treated as negatives: only substances carrying at least one valid
H-code are retained, so absence within such a record is closer to non-applicability than to a
missing observation. This assumption is stated and discussed in the paper.

## License

Code is released under the [MIT License](LICENSE). The dataset in `data/` is released under
[CC BY 4.0](LICENSE-DATA). If you use either, please cite the paper (see `CITATION.cff`).

## Contact

Dahee Kim — dahee02@ajou.ac.kr
Corresponding author: Seung-ho Jung — processsafety@ajou.ac.kr
Department of Environmental and Safety Engineering, Ajou University, Suwon 16499, Republic of Korea
