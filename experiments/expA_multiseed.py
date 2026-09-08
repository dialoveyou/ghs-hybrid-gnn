"""A — the ablation repeated under five seeds.

The single-realization version of this comparison is the weakest link in the paper's
negative result: +0.003 measured once is not evidence that the 3D increment is zero.
Here every neural configuration is retrained under five seeds on the identical split,
stacked with the (deterministic) tree ensemble, and the 3D increment is reported as a
distribution rather than a point.
"""
import pickle, json, time, copy, sys, numpy as np, torch
sys.path.insert(0, "/tmp/exp")
from ablation import Net, Focal, thr_grid, blend_grid, make_ds, run, macro_f1
from torch_geometric.loader import DataLoader

SEEDS = [42, 123, 456, 789, 2024]
EPOCHS, PATIENCE, MIN_DELTA, BATCH, N_FP = 40, 6, 0.003, 64, 896
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

feats = pickle.load(open("/tmp/exp/features.pkl", "rb"))
st = pickle.load(open("/tmp/exp/p0_store.pkl", "rb"))
feat2d, graphs, labels = feats["feat2d_v"].copy(), feats["graphs_v"], feats["labels_v"]
tr, va, te = st["train_idx"], st["val_idx"], st["test_idx"]
mu = feat2d[tr][:, N_FP:].mean(0); sd = feat2d[tr][:, N_FP:].std(0) + 1e-6
feat2d[:, N_FP:] = (feat2d[:, N_FP:] - mu) / sd
yva, yte = st["yva"], st["yte"]
bv, bt = st["baseline_trainonly_val"], st["baseline_trainonly_test"]

tr_ds = make_ds(tr, feat2d, graphs, labels)
va_ds = make_ds(va, feat2d, graphs, labels)
te_ds = make_ds(te, feat2d, graphs, labels)
va_loader = DataLoader(va_ds, batch_size=128, shuffle=False)
te_loader = DataLoader(te_ds, batch_size=128, shuffle=False)
ytr = labels[tr]; pos = ytr.sum(0)
pw = torch.tensor((len(tr) - pos) / np.clip(pos, 1, None), dtype=torch.float32)
inv = 1.0 / (np.clip(pos, 1, None) / len(tr))
sw = np.array([inv[r.astype(bool)].max() if r.sum() > 0 else 1.0 for r in ytr])

def train_one(mode, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    g = torch.Generator(); g.manual_seed(seed)
    sampler = torch.utils.data.WeightedRandomSampler(torch.tensor(sw, dtype=torch.double),
                                                     num_samples=len(sw), replacement=True, generator=g)
    loader = DataLoader(tr_ds, batch_size=BATCH, sampler=sampler)
    model = Net(mode, feat2d.shape[1]); crit = Focal(pw)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=3e-4)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", patience=3, factor=0.5)
    best, best_state, bad = -1, None, 0
    for ep in range(1, EPOCHS + 1):
        run(loader, model, crit, opt)
        _, vaf, _, _ = run(va_loader, model, crit)
        sch.step(vaf)
        if vaf > best + MIN_DELTA: best, best_state, bad = vaf, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= PATIENCE: break
    model.load_state_dict(best_state)
    _, _, pv, _ = run(va_loader, model, crit)
    _, _, pt, _ = run(te_loader, model, crit)
    return pv, pt

if __name__ == "__main__":
    out = {m: {"alone": [], "stacked": []} for m in ("both", "d2", "d3")}
    incr = []
    for seed in SEEDS:
        log(f"===== seed {seed} =====")
        per = {}
        for mode in ("both", "d2", "d3"):
            t0 = time.time()
            pv, pt = train_one(mode, seed)
            a = macro_f1(yte, pt, thr_grid(pv, yva))
            al, tt = blend_grid(pv, bv, yva)
            s = macro_f1(yte, al * pt + (1 - al) * bt, tt)
            out[mode]["alone"].append(a); out[mode]["stacked"].append(s)
            per[mode] = s
            log(f"  {mode:5s} alone {a:.4f} | stacked {s:.4f}   ({time.time()-t0:.0f}s)")
        d = per["both"] - per["d2"]; incr.append(d)
        log(f"  --> 3D increment (stack both - stack 2D-only) = {d:+.4f}")

    res = {"seeds": SEEDS, "increment_per_seed": incr,
           "increment_mean": float(np.mean(incr)), "increment_sd": float(np.std(incr, ddof=1))}
    for m in out:
        for k in out[m]:
            res[f"{m}_{k}_mean"] = float(np.mean(out[m][k]))
            res[f"{m}_{k}_sd"] = float(np.std(out[m][k], ddof=1))
            res[f"{m}_{k}_all"] = out[m][k]
    res["tree_calibrated"] = macro_f1(yte, bt, thr_grid(bv, yva))
    json.dump(res, open("/tmp/exp/expA_results.json", "w"), indent=2, default=float)
    log("")
    log(f"tree (calibrated, deterministic)     {res['tree_calibrated']:.4f}")
    for m, name in (("d3", "3D-only"), ("d2", "2D-only"), ("both", "2D-3D hybrid")):
        log(f"{name:14s} alone {res[m+'_alone_mean']:.4f} ± {res[m+'_alone_sd']:.4f} | "
            f"stacked {res[m+'_stacked_mean']:.4f} ± {res[m+'_stacked_sd']:.4f}")
    log(f"3D INCREMENT  {res['increment_mean']:+.4f} ± {res['increment_sd']:.4f}  "
        f"(per seed: {', '.join(f'{x:+.4f}' for x in incr)})")
