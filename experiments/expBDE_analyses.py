"""B, D, E — analyses that need no retraining.

B  How stable is the calibrated threshold itself, and does the calibration gain survive
   if the rarest classes are removed from the macro average?
D  How far can the "unobserved H-code = negative" assumption be wrong before the
   conclusions move?
E  What actually drives the random-to-scaffold gap: duplicates, near-duplicates, or
   genuine novelty?
"""
import pickle, json, time, numpy as np
from sklearn.metrics import f1_score
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog("rdApp.*")

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
GRID = np.linspace(0.05, 0.95, 181)

st = pickle.load(open("/tmp/exp/p0_store.pkl", "rb"))
ab = pickle.load(open("/tmp/exp/abl_store.pkl", "rb"))
yva, yte = st["yva"], st["yte"]
bv, bt = st["baseline_trainonly_val"], st["baseline_trainonly_test"]
probs = ab["probs"]
CLS = pickle.load(open("/tmp/exp/features.pkl", "rb"))["class_names"]
out = {}

def best_thr(pc, yc):
    pred = pc[None, :] > GRID[:, None]; pos = yc[None, :] > 0.5
    tp = (pred & pos).sum(1); fp = (pred & ~pos).sum(1); fn = ((~pred) & pos).sum(1)
    f1 = np.divide(2*tp, 2*tp+fp+fn, out=np.zeros(len(GRID)), where=(2*tp+fp+fn) > 0)
    return GRID[int(f1.argmax())]

def thr_all(p, y): return np.array([best_thr(p[:, c], y[:, c]) for c in range(y.shape[1])])

def mf(y, p, t, cols=None):
    pred = (p > t).astype(int)
    if cols is None: return f1_score(y, pred, average="macro", zero_division=0)
    return f1_score(y[:, cols], pred[:, cols], average="macro", zero_division=0)

# ================= B — threshold stability =================
log("B — bootstrap stability of the calibrated thresholds (400 resamples of validation)")
base_thr = thr_all(bv, yva)
rng = np.random.default_rng(42); n = len(yva)
draws = np.empty((400, 22))
for i in range(400):
    k = rng.integers(0, n, n)
    draws[i] = thr_all(bv[k], yva[k])
sup_va = yva.sum(0).astype(int); sup_te = yte.sum(0).astype(int)
rows = []
for c in range(22):
    lo, hi = np.percentile(draws[:, c], [2.5, 97.5])
    rows.append(dict(cls=CLS[c], thr=float(base_thr[c]), sd=float(draws[:, c].std()),
                     ci=[float(lo), float(hi)], val_support=int(sup_va[c]), test_support=int(sup_te[c])))
    log(f"   {CLS[c]:32s} tau={base_thr[c]:.3f}  SD {draws[:,c].std():.3f}  "
        f"95% CI [{lo:.2f}, {hi:.2f}]  n_val={sup_va[c]}")
out["B_threshold_stability"] = rows
at_edge = [r["cls"] for r in rows if r["thr"] <= 0.06 or r["thr"] >= 0.94]
out["B_at_grid_boundary"] = at_edge
log(f"   classes with tau at the grid boundary: {len(at_edge)} {at_edge}")

common = [c for c in range(22) if sup_te[c] >= 50]
rare = [c for c in range(22) if sup_te[c] < 50]
g_all = mf(yte, bt, base_thr) - mf(yte, bt, np.full(22, 0.5))
g_com = mf(yte, bt, base_thr, common) - mf(yte, bt, np.full(22, 0.5), common)
g_rare = mf(yte, bt, base_thr, rare) - mf(yte, bt, np.full(22, 0.5), rare)
out["B_calibration_gain"] = {"all_22": g_all, "support_ge_50_only": g_com,
                             "support_lt_50_only": g_rare, "n_common": len(common), "n_rare": len(rare)}
log(f"   calibration gain  all 22 classes {g_all:+.4f} | "
    f"only support>=50 ({len(common)}) {g_com:+.4f} | only support<50 ({len(rare)}) {g_rare:+.4f}")

# ================= D — label-noise sensitivity =================
log("")
log("D — sensitivity to unrecorded positives in low-prevalence classes")
b_th = thr_all(bv, yva)
sb, tb = probs["both"][2], probs["both"][3]
sd2, td2 = probs["d2"][2], probs["d2"][3]
res = {}
for frac in (0.10, 0.25, 0.50):
    tree_v, stk_v, inc_v = [], [], []
    for rep in range(30):
        r = np.random.default_rng(1000 + rep)
        y2 = yte.copy()
        for c in rare:                       # promote negatives to positives in rare classes
            k = int(round(frac * sup_te[c]))
            neg = np.flatnonzero(y2[:, c] < 0.5)
            if k > 0 and len(neg) >= k: y2[r.choice(neg, k, replace=False), c] = 1.0
        tree_v.append(mf(y2, bt, b_th))
        s_both = mf(y2, sb, tb); s_d2 = mf(y2, sd2, td2)
        stk_v.append(s_both); inc_v.append(s_both - s_d2)
    res[f"{int(frac*100)}pct"] = {
        "tree_calibrated": float(np.mean(tree_v)),
        "stacked_both": float(np.mean(stk_v)),
        "increment_3d": float(np.mean(inc_v)), "increment_3d_sd": float(np.std(inc_v))}
    log(f"   +{int(frac*100)}% unrecorded positives in the {len(rare)} rare classes: "
        f"tree {np.mean(tree_v):.4f} | stacked {np.mean(stk_v):.4f} | "
        f"3D increment {np.mean(inc_v):+.4f} ± {np.std(inc_v):.4f}")
out["D_label_noise"] = res
out["D_baseline_no_noise"] = {"tree_calibrated": mf(yte, bt, b_th),
                              "stacked_both": mf(yte, sb, tb),
                              "increment_3d": mf(yte, sb, tb) - mf(yte, sd2, td2)}

# ================= E — duplicates and chemical-space overlap =================
log("")
log("E — duplicates and train/test similarity")
smiles_v, scaffolds = pickle.load(open("/tmp/exp/scaffolds.pkl", "rb"))
canon = []
for s in smiles_v:
    m = Chem.MolFromSmiles(s)
    canon.append(Chem.MolToSmiles(m) if m else s)
canon = np.array(canon)
uniq, counts = np.unique(canon, return_counts=True)
out["E_duplicates"] = {"n_compounds": len(canon), "n_unique_canonical_smiles": int(len(uniq)),
                       "n_exact_duplicate_rows": int((counts > 1).sum()),
                       "max_copies_of_one_structure": int(counts.max())}
log(f"   {len(canon)} rows | {len(uniq)} unique canonical SMILES | "
    f"{(counts>1).sum()} structures appearing more than once (max {counts.max()} copies)")

sc_u, sc_c = np.unique(scaffolds, return_counts=True)
out["E_scaffolds"] = {"n_unique": int(len(sc_u)), "singletons": int((sc_c == 1).sum()),
                      "largest": int(sc_c.max()), "median_size": float(np.median(sc_c)),
                      "top10_sizes": sorted(sc_c.tolist(), reverse=True)[:10]}
log(f"   {len(sc_u)} scaffolds | {(sc_c==1).sum()} singletons | largest {sc_c.max()} compounds")

fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, nBits=729)
       if Chem.MolFromSmiles(s) else None for s in smiles_v]
def tmax(train_idx, test_idx):
    trf = [fps[i] for i in train_idx if fps[i] is not None]
    v = []
    for j in test_idx:
        if fps[j] is None: continue
        v.append(max(DataStructs.BulkTanimotoSimilarity(fps[j], trf)))
    return np.array(v)

from sklearn.model_selection import GroupShuffleSplit
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
s_tr, s_te = next(gss.split(np.zeros(len(scaffolds)), np.zeros(len(scaffolds)), groups=scaffolds))
for name, (a, b) in (("random", (st["train_idx"], st["test_idx"])), ("scaffold", (s_tr, s_te))):
    t = tmax(a, b)
    out[f"E_tmax_{name}"] = {"mean": float(t.mean()), "median": float(np.median(t)),
                             "frac_ge_0.9": float((t >= 0.9).mean()), "frac_ge_0.99": float((t >= 0.99).mean()),
                             "frac_le_0.4": float((t <= 0.4).mean())}
    log(f"   {name:9s} split: mean T_max {t.mean():.3f} | median {np.median(t):.3f} | "
        f">=0.9 {100*(t>=0.9).mean():.1f}% | >=0.99 {100*(t>=0.99).mean():.1f}% | <=0.4 {100*(t<=0.4).mean():.1f}%")

json.dump(out, open("/tmp/exp/expBDE_results.json", "w"), indent=2, default=float)
log("saved expBDE_results.json")
