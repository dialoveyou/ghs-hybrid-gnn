# Trained weights

`hybrid_gnn_ecc_ghs22.pt` — PyTorch `state_dict` for the hybrid 2D–3D GNN-ECC model
(`HybridGNN_ECC` in `src/ghs_full_pipeline.py`): a three-layer GCN branch over molecular graphs
with covalent and ≤ 5 Å non-covalent proximity edges, a two-layer MLP branch over the
916-dimensional 2D feature vector, and a differentiable 22-class classifier chain on the fused
256-dimensional representation.

These weights come from the complete CPU-only run recorded in `../results/metrics.json`
(GNN-ECC macro-F1 0.575, stacked 0.639), not from the GPU run that produced the headline values
in Table 2 of the paper (0.602 and 0.660). The two differ by 0.01–0.05 macro-F1, within ordinary
run-to-run variance; see §4.9 of the paper and the Reproducibility section of the top-level
README.

The stacking blend weights and per-class decision thresholds are **not** in this file — they are
stored in `../results/metrics.json` under `stage2_final_results.best_alphas`,
`blend_thresholds` and `gnn_thresholds`, and are required to turn model probabilities into the
binary predictions reported in the paper.

## Loading

```python
import torch
from src.ghs_full_pipeline import HybridGNN_ECC

model = HybridGNN_ECC(...)          # same constructor arguments as in run_stage2.py
model.load_state_dict(torch.load("checkpoints/hybrid_gnn_ecc_ghs22.pt", map_location="cpu"))
model.eval()
```
