# Stacked 2D–3D GNN for Multi-Label GHS Hazard Classification

Code, data and trained weights for:

> Kim, D.; Jung, S.-h. *A Stacked 2D–3D Graph Neural Network Framework for Comprehensive Multi-Label GHS Hazard Classification from Molecular Structure.* ACS Omega (submitted).

The model predicts all **22 consolidated GHS hazard classes simultaneously** from molecular
structure, by stacking a 2D tree-ensemble baseline with a 3D graph neural network whose
molecular graph augments covalent bonds with distance-weighted non-covalent proximity edges
(≤ 5 Å), coupled to a differentiable classifier chain.

The central result is that the 3D branch is **not** a better standalone predictor than 2D
fingerprints (Δ macro-F1 = −0.017, p = 0.176), but the two representations combine to give a
significant improvement that neither achieves alone (stacked macro-F1 0.660; +0.042 over the
baseline, p < 0.001).

---

## Repository layout

```
src/
  ghs_full_pipeline.py     single-file reference implementation (same logic as the notebook)
  run_full_pipeline.py     Stage 1 — data loading, 22-class labelling, 2D/3D feature extraction
  run_stage2.py            Stage 2 — splits, augmentation, baseline + GNN training, stacking
  run_stage3.py            Stage 3 — 5-fold CV, scaffold split, bootstrap tests, SHAP
  analysis/
    head_to_head_endpoints.py          per-endpoint AUC/F1/BalAcc for the six
    head_to_head_endpoints_snippet.py  Fuadah-overlapping endpoints (Table 7)
notebooks/
  GHS_Hybrid_GNN_ECC.ipynb  end-to-end notebook version (Colab-ready)
data/
  ghs_modeling_dataset_14148.csv  14,148 curated compounds with GHS H-codes
checkpoints/
  hybrid_gnn_ecc_ghs22.pt   trained hybrid GNN-ECC weights
results/
  metrics.json              full metric dump from a complete end-to-end run
  reproduction_summary_KR.md
```

Stages 1–3 are **checkpointed and resumable**: every stage caches its intermediate state, so an
interrupted run (a dropped Colab runtime, for instance) continues from where it stopped when the
same script is run again. All results accumulate into a single `metrics.json`.

## Installation

Python 3.12 is recommended.

```bash
git clone https://github.com/<GITHUB_USERNAME>/ghs-hybrid-gnn.git
cd ghs-hybrid-gnn
pip install -r requirements.txt
```

`torch-geometric` may need a wheel matching your CUDA/PyTorch build; see the
[PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).
A GPU is optional — the pipeline runs end to end on CPU, only more slowly.

## Running the pipeline

```bash
export GHS_WORKDIR=./work          # where caches, checkpoints and metrics.json are written
mkdir -p "$GHS_WORKDIR"
cp data/ghs_modeling_dataset_14148.csv "$GHS_WORKDIR/_wip_modeling_subset_14148.csv"

python src/run_full_pipeline.py    # Stage 1  (~8 min, CPU)
python src/run_stage2.py           # Stage 2  (~1 h on 2 CPU cores; much faster with a GPU)
python src/run_stage3.py           # Stage 3  (~35 min)
```

`GHS_WORKDIR` defaults to `<repo>/work` if unset. Measured wall-clock on 2 CPU cores with no
GPU was about 1 h 45 min in total, dominated by the baseline Optuna search (~45 min) and the
5-fold cross-validation (~30 min); both shorten substantially with more cores.

To reproduce only the head-to-head comparison against Fuadah et al. (Table 7), run the notebook
through its ROC-AUC / Balanced-Accuracy section and then execute
`src/analysis/head_to_head_endpoints.py`, which expects the fitted objects to be in memory.

## Reproducibility

Seeds are fixed (`SEED = 42`) and the multi-label stratified split is deterministic, so the
data-processing stages reproduce exactly:

| Quantity | This code | Reported in paper |
|---|---|---|
| Compounds with GHS labels | 14,148 | 14,148 |
| Successful 3D embeddings | 14,145 (0.02% failure) | 14,145 (0.02% failure) |
| Augmented classes | 13 | 13 |
| Unique Murcko scaffolds | 8,114 | 8,114 |
| Scaffold split train/test | 9,742 / 4,403 | 9,742 / 4,403 |

**Model metrics carry ordinary run-to-run variance.** The headline values in the paper
(Table 2: baseline 0.618, GNN-ECC 0.602, stacked 0.660) were obtained on GPU hardware. A
complete CPU-only run of this repository — the one recorded in `results/metrics.json` and
described in §4.9 of the paper — gives 0.573, 0.575 and 0.639 respectively: uniformly 0.01–0.05
lower, with the ordering and every qualitative conclusion unchanged (stacking significantly
beats both constituents at p < 0.001; scaffold splitting costs ~12–15 macro-F1 points; SHAP
ranks LogP and molar refractivity top among descriptors). Expect numbers in that band rather
than an exact match to Table 2, and compare **within** a single run.

The checkpoint in `checkpoints/` is from that CPU-only run.

## Data

`data/ghs_modeling_dataset_14148.csv` — 14,148 single-substance chemicals, each with a CAS
Registry Number, PubChem CID, canonical SMILES, 15 physicochemical properties, NFPA
Health/Fire/Reactivity ratings where available, and one or more GHS H-codes.

Assembled from the Korea Occupational Safety and Health Agency (KOSHA) chemical database and
the Dangjin City industrial chemical usage registry, with structures and properties retrieved
through the PubChem PUG REST API. Mixtures and entries without a valid single-substance SMILES
were excluded. See `data/README.md` for the column schema and the 73 H-code → 22 class mapping.

## License

Code is released under the [MIT License](LICENSE). The dataset in `data/` is released under
[CC BY 4.0](LICENSE-DATA). If you use either, please cite the paper (see `CITATION.cff`).

## Contact

Dahee Kim — dahee02@ajou.ac.kr
Corresponding author: Seung-ho Jung — processsafety@ajou.ac.kr
Department of Environmental and Safety Engineering, Ajou University, Suwon 16499, Republic of Korea
