"""F — nested cross-validation of the whole comparison.

Every quantity the paper reports is selected on a validation partition: tree
hyperparameters, per-class thresholds, and the stacking blend weights. Reporting those on a
single held-out test set shows there is no test leakage, but it does not show that the
selection itself is unbiased. Here the entire selection procedure is repeated inside each of
five outer folds, and each fold is scored on data no part of the procedure has seen.
"""
import pickle, json, time, copy, sys, numpy as np, torch
sys.path.insert(0, "/tmp/exp")
from ablation import Net, Focal, thr_grid, blend_grid, make_ds, run, macro_f1
from p0_baseline import get_models, soft_vote, PARAMS
from torch_geometric.loader import DataLoader
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

SEED, EPOCHS, PATIENCE, MIN_DELTA, BATCH, N_FP = 42, 40, 6, 0.003, 64, 896
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

feats = pickle.load(open("/tmp/exp/features.pkl", "rb"))
FEAT, GRAPHS, LAB = feats["feat2d_v"], feats["graphs_v"], feats["labels_v"]

def train_neural(mode, feat2d, tr, va, te, seed=SEED):
    tr_ds = make_ds(tr, feat2d, GRAPHS, LAB)
    va_l = DataLoader(make_ds(va, feat2d, GRAPHS, LAB), batch_size=128, shuffle=False)
    te_l = DataLoader(make_ds(te, feat2d, GRAPHS, LAB), batch_size=128, shuffle=False)
    ytr = LAB[tr]; pos = ytr.sum(0)
    pw = torch.tensor((len(tr) - pos) / np.clip(pos, 1, None), dtype=torch.float32)
    inv = 1.0 / (np.clip(pos, 1, None) / len(tr))
    sw = np.array([inv[r.astype(bool)].max() if r.sum() > 0 else 1.0 for r in ytr])
    torch.manual_seed(seed); np.random.seed(seed)
    g = torch.Generator(); g.manual_seed(seed)
    sampler = torch.utils.data.WeightedRandomSampler(torch.tensor(sw, dtype=torch.double),
                                                     num_samples=len(sw), replacement=True, generator=g)
    tr_l = DataLoader(tr_ds, batch_size=BATCH, sampler=sampler)
    model = Net(mode, feat2d.shape[1]); crit = Focal(pw)
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
    _, _, pv, _ = run(va_l, model, crit)
    _, _, pt, _ = run(te_l, model, crit)
    return pv, pt

if __name__ == "__main__":
    KEYS = ["tree_0.5", "tree_cal", "d2_alone", "d3_alone", "both_alone",
            "stack_d2", "stack_d3", "stack_both", "increment_3d"]
    acc = {k: [] for k in KEYS}
    thr_by_fold = []
    outer = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (tr_all, te) in enumerate(outer.split(FEAT, LAB), 1):
        t_fold = time.time()
        inner = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        a, b = next(iter(inner.split(FEAT[tr_all], LAB[tr_all])))
        tr, va = tr_all[a], tr_all[b]
        X = FEAT.copy()
        mu = X[tr][:, N_FP:].mean(0); sd = X[tr][:, N_FP:].std(0) + 1e-6
        X[:, N_FP:] = (X[:, N_FP:] - mu) / sd
        yva, yte = LAB[va], LAB[te]
        log(f"===== outer fold {fold}: train {len(tr)} / val {len(va)} / test {len(te)} =====")

        ms = get_models(PARAMS)
        for m in ms: m.fit(X[tr], LAB[tr])
        bv, bt = soft_vote(ms, X[va]), soft_vote(ms, X[te])
        b_th = thr_grid(bv, yva)
        thr_by_fold.append(b_th.tolist())
        acc["tree_0.5"].append(macro_f1(yte, bt, np.full(22, 0.5)))
        acc["tree_cal"].append(macro_f1(yte, bt, b_th))
        log(f"  tree  @0.5 {acc['tree_0.5'][-1]:.4f} | calibrated {acc['tree_cal'][-1]:.4f}")

        stacked = {}
        for mode in ("d2", "d3", "both"):
            pv, pt = train_neural(mode, X, tr, va, te)
            acc[f"{mode}_alone"].append(macro_f1(yte, pt, thr_grid(pv, yva)))
            al, tt = blend_grid(pv, bv, yva)
            s = macro_f1(yte, al * pt + (1 - al) * bt, tt)
            acc[f"stack_{mode}"].append(s); stacked[mode] = s
            log(f"  {mode:5s} alone {acc[f'{mode}_alone'][-1]:.4f} | stacked {s:.4f}")
        acc["increment_3d"].append(stacked["both"] - stacked["d2"])
        log(f"  fold {fold} 3D increment {acc['increment_3d'][-1]:+.4f}   ({time.time()-t_fold:.0f}s)")

    res = {k: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)), "folds": v}
           for k, v in acc.items()}
    res["calibration_gain"] = {"mean": float(np.mean(np.array(acc["tree_cal"]) - np.array(acc["tree_0.5"]))),
                               "sd": float(np.std(np.array(acc["tree_cal"]) - np.array(acc["tree_0.5"]), ddof=1))}
    res["thresholds_by_fold"] = thr_by_fold
    json.dump(res, open("/tmp/exp/expF_results.json", "w"), indent=2, default=float)
    log("")
    log("=== nested CV, mean ± SD over 5 outer folds ===")
    for k in KEYS:
        log(f"  {k:14s} {res[k]['mean']:+.4f} ± {res[k]['sd']:.4f}")
    log(f"  calibration gain {res['calibration_gain']['mean']:+.4f} ± {res['calibration_gain']['sd']:.4f}")
