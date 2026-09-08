"""C — is the 3D increment an artefact of the classifier chain's label order?

A classifier chain propagates logits in a fixed label order, so its inductive bias depends
on that order. The paper's order is the GHS taxonomy order (01…22), which is arbitrary with
respect to the learning problem. Here the 2D-only and 2D-3D configurations are retrained
under the original order, its reverse, and three random permutations; if the 3D increment
moves more across orderings than its own magnitude, it cannot be read as a property of the
representation.
"""
import pickle, json, time, copy, sys, numpy as np, torch
sys.path.insert(0, "/tmp/exp")
from ablation import Net, Focal, thr_grid, blend_grid, make_ds, run, macro_f1
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

SEED, EPOCHS, PATIENCE, MIN_DELTA, BATCH, N_FP = 42, 40, 6, 0.003, 64, 896
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

feats = pickle.load(open("/tmp/exp/features.pkl", "rb"))
st = pickle.load(open("/tmp/exp/p0_store.pkl", "rb"))
feat2d, graphs, labels = feats["feat2d_v"].copy(), feats["graphs_v"], feats["labels_v"]
tr, va, te = st["train_idx"], st["val_idx"], st["test_idx"]
mu = feat2d[tr][:, N_FP:].mean(0); sd = feat2d[tr][:, N_FP:].std(0) + 1e-6
feat2d[:, N_FP:] = (feat2d[:, N_FP:] - mu) / sd
yva, yte = st["yva"], st["yte"]
bv, bt = st["baseline_trainonly_val"], st["baseline_trainonly_test"]
ytr = labels[tr]; pos = ytr.sum(0)
pw_base = (len(tr) - pos) / np.clip(pos, 1, None)
inv = 1.0 / (np.clip(pos, 1, None) / len(tr))
sw = np.array([inv[r.astype(bool)].max() if r.sum() > 0 else 1.0 for r in ytr])

def ds_perm(idxs, perm):
    out = []
    for i in idxs:
        x, s, d_, w = graphs[i]
        out.append(Data(x=torch.tensor(x),
                        edge_index=torch.tensor(np.vstack([s, d_]), dtype=torch.long),
                        edge_weight=torch.tensor(w),
                        feat2d=torch.tensor(feat2d[i]).unsqueeze(0),
                        y=torch.tensor(labels[i][perm]).unsqueeze(0)))
    return out

def train_eval(mode, perm):
    torch.manual_seed(SEED); np.random.seed(SEED)
    g = torch.Generator(); g.manual_seed(SEED)
    tr_ds, va_ds, te_ds = ds_perm(tr, perm), ds_perm(va, perm), ds_perm(te, perm)
    va_l = DataLoader(va_ds, batch_size=128, shuffle=False)
    te_l = DataLoader(te_ds, batch_size=128, shuffle=False)
    sampler = torch.utils.data.WeightedRandomSampler(torch.tensor(sw, dtype=torch.double),
                                                     num_samples=len(sw), replacement=True, generator=g)
    tr_l = DataLoader(tr_ds, batch_size=BATCH, sampler=sampler)
    model = Net(mode, feat2d.shape[1])
    crit = Focal(torch.tensor(pw_base[perm], dtype=torch.float32))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=3e-4)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", patience=3, factor=0.5)
    best, best_state, bad = -1, None, 0
    for ep in range(1, EPOCHS + 1):
        run(tr_l, model, crit, opt)
        _, vaf, _, _ = run(va_l, model, crit)
        sch.step(vaf)
        if vaf > best + MIN_DELTA: best, best_state, bad = vaf, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= PATIENCE: break
    model.load_state_dict(best_state)
    _, _, pv_p, _ = run(va_l, model, crit)
    _, _, pt_p, _ = run(te_l, model, crit)
    pv = np.empty_like(pv_p); pt = np.empty_like(pt_p)      # back to canonical class order
    pv[:, perm] = pv_p; pt[:, perm] = pt_p
    return pv, pt

if __name__ == "__main__":
    rng = np.random.default_rng(7)
    ORDERS = [("original", np.arange(22)), ("reversed", np.arange(22)[::-1].copy())]
    for k in range(3):
        ORDERS.append((f"random{k+1}", rng.permutation(22)))

    res, incs = {}, []
    for name, perm in ORDERS:
        row = {}
        for mode in ("both", "d2"):
            t0 = time.time()
            pv, pt = train_eval(mode, perm)
            al, tt = blend_grid(pv, bv, yva)
            row[mode] = macro_f1(yte, al * pt + (1 - al) * bt, tt)
            row[mode + "_alone"] = macro_f1(yte, pt, thr_grid(pv, yva))
            log(f"  {name:9s} {mode:5s} alone {row[mode+'_alone']:.4f} | stacked {row[mode]:.4f}  ({time.time()-t0:.0f}s)")
        row["increment_3d"] = row["both"] - row["d2"]
        incs.append(row["increment_3d"])
        res[name] = row
        log(f"  {name:9s} --> 3D increment {row['increment_3d']:+.4f}")

    res["increment_mean"] = float(np.mean(incs))
    res["increment_sd"] = float(np.std(incs, ddof=1))
    res["increment_range"] = [float(min(incs)), float(max(incs))]
    json.dump(res, open("/tmp/exp/expC_results.json", "w"), indent=2, default=float)
    log("")
    log(f"3D increment across {len(ORDERS)} chain orderings: "
        f"{np.mean(incs):+.4f} ± {np.std(incs, ddof=1):.4f}  "
        f"(range {min(incs):+.4f} to {max(incs):+.4f})")
