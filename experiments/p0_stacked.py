"""Reconstruct the published stacked model from its released checkpoint, verify it
reproduces the reported macro-F1, then run paired bootstrap tests against the baseline
under matched and unmatched threshold protocols."""
import pickle, json, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score

SEED = 42; N_CLASSES = 22; FUSION_DIM = 256; N_FP = 896
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

class GNNBranch3D(nn.Module):
    def __init__(self, in_dim=9, hidden=64, out_dim=128, dropout=0.2):
        super().__init__()
        self.conv1, self.conv2, self.conv3 = GCNConv(in_dim,hidden), GCNConv(hidden,hidden), GCNConv(hidden,out_dim)
        self.bn1, self.bn2 = nn.BatchNorm1d(hidden), nn.BatchNorm1d(hidden)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, ei, ew, b):
        h = self.dropout(F.relu(self.bn1(self.conv1(x, ei, ew))))
        h = self.dropout(F.relu(self.bn2(self.conv2(h, ei, ew))))
        return global_add_pool(self.conv3(h, ei, ew), b)

class MLPBranch2D(nn.Module):
    def __init__(self, in_dim, hidden=256, out_dim=128, dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim,hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(hidden,out_dim),
                                 nn.BatchNorm1d(out_dim), nn.ReLU())
    def forward(self, x): return self.net(x)

class DifferentiableECC(nn.Module):
    def __init__(self, fusion_dim=FUSION_DIM, n_classes=N_CLASSES, head_hidden=64):
        super().__init__()
        self.heads = nn.ModuleList([nn.Sequential(nn.Linear(fusion_dim+i, head_hidden), nn.ReLU(),
                                                  nn.Linear(head_hidden,1)) for i in range(n_classes)])
    def forward(self, f):
        outs, ci = [], f
        for h in self.heads:
            o = h(ci); outs.append(o); ci = torch.cat([ci, o], dim=1)
        return torch.cat(outs, dim=1)

class HybridGNN_ECC(nn.Module):
    def __init__(self, feat2d_dim, atom_feat_dim=9, n_classes=N_CLASSES):
        super().__init__()
        self.branch3d = GNNBranch3D(in_dim=atom_feat_dim)
        self.branch2d = MLPBranch2D(in_dim=feat2d_dim)
        self.chain = DifferentiableECC(n_classes=n_classes)
    def forward(self, f2, gx, ei, ew, b):
        return self.chain(torch.cat([self.branch2d(f2), self.branch3d(gx, ei, ew, b)], dim=1))

def macro_f1(y, p, t): return f1_score(y, (p > t).astype(int), average="macro", zero_division=0)

def paired_bootstrap(y, pa, ta, pb, tb, n_boot=5000, seed=SEED):
    """b minus a; identical resampled indices for both models (paired)."""
    rng = np.random.default_rng(seed); n = y.shape[0]; d = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        d[i] = (f1_score(y[idx], (pb[idx] > tb).astype(int), average="macro", zero_division=0)
                - f1_score(y[idx], (pa[idx] > ta).astype(int), average="macro", zero_division=0))
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return d.mean(), lo, hi, p

if __name__ == "__main__":
    feats = pickle.load(open("/tmp/exp/features.pkl", "rb"))
    st = pickle.load(open("/tmp/exp/p0_store.pkl", "rb"))
    ck = torch.load("/mnt/user-data/uploads/ACS OMEGA/checkpoints/hybrid_gnn_ecc_ghs22.pt",
                    map_location="cpu", weights_only=False)

    feat2d = feats["feat2d_v"].copy(); graphs = feats["graphs_v"]
    tr = st["train_idx"]
    mu = feat2d[tr][:, N_FP:].mean(axis=0); sd = feat2d[tr][:, N_FP:].std(axis=0) + 1e-6
    feat2d[:, N_FP:] = (feat2d[:, N_FP:] - mu) / sd

    model = HybridGNN_ECC(feat2d_dim=feat2d.shape[1])
    model.load_state_dict(ck["model_state_dict"]); model.eval()
    log("checkpoint loaded into architecture without shape errors")

    def probs(idxs):
        ds = []
        for i in idxs:
            x, s, d_, w = graphs[i]
            ds.append(Data(x=torch.tensor(x), edge_index=torch.tensor(np.vstack([s, d_]), dtype=torch.long),
                           edge_weight=torch.tensor(w), feat2d=torch.tensor(feat2d[i]).unsqueeze(0)))
        out = []
        with torch.no_grad():
            for batch in DataLoader(ds, batch_size=128, shuffle=False):
                out.append(torch.sigmoid(model(batch.feat2d, batch.x, batch.edge_index,
                                               batch.edge_weight, batch.batch)).numpy())
        return np.vstack(out)

    log("running GNN on test set ...")
    gnn_te = probs(st["test_idx"])
    yte = st["yte"]

    a, bt, gt = ck["stack_alphas"], ck["stack_thresholds"], ck["gnn_thresholds"]
    base_te_tv = st["baseline_trainval_test"]          # baseline as used inside the published blend
    stacked_te = a * gnn_te + (1 - a) * base_te_tv

    f_gnn = macro_f1(yte, gnn_te, gt)
    f_stk = macro_f1(yte, stacked_te, bt)
    log(f"GNN-ECC   (published thresholds) macro-F1 = {f_gnn:.4f}   [paper 0.5753]")
    log(f"Stacked   (published alphas/thr) macro-F1 = {f_stk:.4f}   [paper 0.6394]")

    half = np.full(N_CLASSES, 0.5)
    rows = {
        "paper_baseline_thr0.5":        (st["baseline_trainval_test"], half),
        "baseline_trainonly_Optuna50":  (st["baseline_trainonly_test"], st["thr_optuna"]),
        "baseline_trainonly_grid":      (st["baseline_trainonly_test"], st["thr_grid"]),
        "baseline_trainval_grid":       (st["baseline_trainval_test"],  None),
    }
    # matched-protocol baseline: same train+val fit and val-tuned thresholds as the blend uses
    from p0_baseline import thr_grid
    thr_tv = thr_grid(st["baseline_trainval_val"], st["yva"])
    rows["baseline_trainval_grid"] = (st["baseline_trainval_test"], thr_tv)

    log("")
    log("=== test macro-F1, same split, same test set ===")
    for k, (p, t) in rows.items():
        log(f"  {k:34s} {macro_f1(yte, p, t):.4f}")
    log(f"  {'stacked (published)':34s} {f_stk:.4f}")

    log("")
    log("=== paired bootstrap, 5000 resamples: stacked minus baseline ===")
    out = {}
    for k, (p, t) in rows.items():
        m, lo, hi, pv = paired_bootstrap(yte, p, t, stacked_te, bt, n_boot=5000)
        sig = "significant" if (lo > 0 or hi < 0) else "not significant"
        out[k] = dict(delta=m, ci=[lo, hi], p=pv, sig=sig)
        log(f"  vs {k:34s} Δ={m:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  p={pv:.4f}  {sig}")

    json.dump({"gnn_f1": f_gnn, "stacked_f1": f_stk,
               "baselines": {k: macro_f1(yte, p, t) for k, (p, t) in rows.items()},
               "bootstrap": out}, open("/tmp/exp/p0_stacked_results.json", "w"), indent=2, default=float)
    log("saved p0_stacked_results.json")
