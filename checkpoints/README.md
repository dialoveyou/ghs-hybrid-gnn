# Trained weights

`hybrid_gnn_ecc_ghs22.pt` — PyTorch checkpoint for the hybrid 2D–3D GNN-ECC model
(`HybridGNN_ECC` in `src/ghs_full_pipeline.py`): a three-layer GCN branch over molecular graphs
with covalent and ≤ 5 Å non-covalent proximity edges, a two-layer MLP branch over the
916-dimensional 2D feature vector, and a differentiable 22-class classifier chain on the fused
256-dimensional representation.

The file is a dict with four keys:

| Key | Contents |
|---|---|
| `model_state_dict` | the network weights |
| `gnn_thresholds` | per-class decision thresholds for the GNN alone (22 values) |
| `stack_alphas` | per-class blend weights against the tree ensemble (22 values) |
| `stack_thresholds` | per-class decision thresholds for the blend (22 values) |

The thresholds and blend weights are **not** optional extras — they are what turns model
probabilities into the binary predictions any reported macro-F1 refers to. `results/metrics.json`
carries the same values under `stage2_final_results`.

These weights come from the legacy pipeline run documented in `results/reproduction_summary.md`.
`experiments/p0_stacked.py` loads this checkpoint, reproduces that run's stacked macro-F1 of 0.639
exactly, and then compares it against threshold-matched baselines — which is how the paper's
analysis is anchored to a concrete, downloadable artefact rather than to a number in a table.

The ablation models of Table 2 (2D-only, 3D-only, 2D–3D) are trained fresh by
`experiments/ablation.py` in about five minutes on CPU and are not shipped as checkpoints.

## Loading

```python
import torch
from src.ghs_full_pipeline import HybridGNN_ECC

ck = torch.load("checkpoints/hybrid_gnn_ecc_ghs22.pt", map_location="cpu", weights_only=False)
model = HybridGNN_ECC(feat2d_dim=916)
model.load_state_dict(ck["model_state_dict"])
model.eval()

alphas, thresholds = ck["stack_alphas"], ck["stack_thresholds"]
```
