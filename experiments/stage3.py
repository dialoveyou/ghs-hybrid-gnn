"""Scaffold-disjoint split and 5-fold CV under the matched threshold protocol.

The published versions of both analyses score the tree baseline at a fixed 0.5 threshold,
so they inherit the same confound as Table 2. Here every model is scored twice: at 0.5 and
at per-class thresholds tuned on a held-out validation partition that is disjoint from both
the training data and the evaluation data.
"""
import os, sys, time, json, pickle, copy
import numpy as np, pandas as pd
from multiprocessing import Pool
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, "/tmp/exp")
from features import build_label_vector, CLASS_NAMES
from p0_baseline import get_models, soft_vote, PARAMS

SEED = 42; N_FP = 896
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
def macro_f1(y, p, t): return f1_score(y, (p > t).astype(int), average="macro", zero_division=0)

def thr_grid(proba, y, n=181):
    grid = np.linspace(0.05, 0.95, n); out = np.full(proba.shape[1], 0.5)
    for c in range(proba.shape[1]):
        pred = proba[:, c][None, :] > grid[:, None]
        pos = y[:, c][None, :] > 0.5
        tp = (pred & pos).sum(1); fp = (pred & ~pos).sum(1); fn = ((~pred) & pos).sum(1)
        f1 = np.divide(2*tp, 2*tp+fp+fn, out=np.zeros(len(grid)), where=(2*tp+fp+fn) > 0)
        out[c] = grid[int(f1.argmax())]
    return out

def embed_ok(args):
    i, smi = args
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return i, False, ""
    try: Chem.SanitizeMol(mol)
    except Exception: return i, False, ""
    molH = Chem.AddHs(Chem.Mol(mol))
    p = AllChem.ETKDGv3(); p.randomSeed = SEED; p.useRandomCoords = True
    ok = AllChem.EmbedMolecule(molH, p) >= 0
    if ok:
        try: AllChem.MMFFOptimizeMolecule(molH, maxIters=200)
        except Exception: pass
        ok = Chem.RemoveHs(molH).GetNumAtoms() > 0
    try:
        sc = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False) or smi
    except Exception:
        sc = smi
    return i, ok, sc

if __name__ == "__main__":
    CACHE = "/tmp/exp/scaffolds.pkl"
    if os.path.exists(CACHE):
        smiles_v, scaffolds = pickle.load(open(CACHE, "rb"))
        log("scaffold cache loaded")
    else:
        df = pd.read_csv("/tmp/src_raw/_wip_modeling_subset_14148.csv")
        labels = np.stack(df["Y_GHS_H_Codes"].apply(build_label_vector).values)
        df = df[labels.sum(1) > 0].reset_index(drop=True)
        log(f"recovering 3D-embedding mask and Murcko scaffolds for {len(df)} compounds ...")
        res = {}
        with Pool(2) as pool:
            for k, (i, ok, sc) in enumerate(pool.imap_unordered(embed_ok, list(enumerate(df["SMILES"].tolist())), chunksize=16)):
                res[i] = (ok, sc)
                if (k+1) % 3000 == 0: log(f"  {k+1}/{len(df)}")
        valid = sorted(i for i in res if res[i][0])
        smiles_v = df["SMILES"].values[valid]
        scaffolds = np.array([res[i][1] for i in valid])
        pickle.dump((smiles_v, scaffolds), open(CACHE, "wb"))
        log(f"3D-embedding success {len(valid)} (expected 14,145)")

    feats = pickle.load(open("/tmp/exp/features.pkl", "rb"))
    feat2d_all, labels_v = feats["feat2d_v"], feats["labels_v"]
    assert len(scaffolds) == len(labels_v), f"{len(scaffolds)} vs {len(labels_v)}"
    log(f"unique Murcko scaffolds: {len(set(scaffolds))} (paper: 8,114)")
    out = {}

    # ================= scaffold-disjoint split =================
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    s_tr_all, s_te = next(gss.split(feat2d_all, labels_v, groups=scaffolds))
    log(f"scaffold split: train {len(s_tr_all)} / test {len(s_te)} (paper: 9,742 / 4,403)")

    # carve a scaffold-disjoint validation partition out of the scaffold-train side
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    sub_tr, sub_va = next(gss2.split(feat2d_all[s_tr_all], labels_v[s_tr_all], groups=scaffolds[s_tr_all]))
    s_tr, s_va = s_tr_all[sub_tr], s_tr_all[sub_va]
    log(f"  scaffold train/val/test = {len(s_tr)} / {len(s_va)} / {len(s_te)}")

    X = feat2d_all.copy()
    mu = X[s_tr][:, N_FP:].mean(0); sd = X[s_tr][:, N_FP:].std(0) + 1e-6
    X[:, N_FP:] = (X[:, N_FP:] - mu) / sd

    log("fitting tree ensemble on scaffold-train ...")
    t0 = time.time(); ms = get_models(PARAMS)
    for m in ms: m.fit(X[s_tr], labels_v[s_tr])
    log(f"  done ({time.time()-t0:.0f}s)")
    pv, pt = soft_vote(ms, X[s_va]), soft_vote(ms, X[s_te])
    thr = thr_grid(pv, labels_v[s_va])
    out["scaffold_baseline_thr0.5"] = macro_f1(labels_v[s_te], pt, 0.5)
    out["scaffold_baseline_thrTuned"] = macro_f1(labels_v[s_te], pt, thr)
    log(f"  scaffold baseline @0.5    = {out['scaffold_baseline_thr0.5']:.4f}   (paper reports 0.454/0.469)")
    log(f"  scaffold baseline @tuned  = {out['scaffold_baseline_thrTuned']:.4f}")

    # ================= 5-fold CV =================
    log("5-fold CV (threshold tuned inside each fold) ...")
    mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    f05, ftu = [], []
    for k, (tr_all, te) in enumerate(mskf.split(feat2d_all, labels_v), 1):
        m2 = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        a, b = next(iter(m2.split(feat2d_all[tr_all], labels_v[tr_all])))
        tr, va = tr_all[a], tr_all[b]
        Xk = feat2d_all.copy()
        mu = Xk[tr][:, N_FP:].mean(0); sd = Xk[tr][:, N_FP:].std(0) + 1e-6
        Xk[:, N_FP:] = (Xk[:, N_FP:] - mu) / sd
        mk = get_models(PARAMS)
        t0 = time.time()
        for m in mk: m.fit(Xk[tr], labels_v[tr])
        pvk, ptk = soft_vote(mk, Xk[va]), soft_vote(mk, Xk[te])
        tk = thr_grid(pvk, labels_v[va])
        a5, at = macro_f1(labels_v[te], ptk, 0.5), macro_f1(labels_v[te], ptk, tk)
        f05.append(a5); ftu.append(at)
        log(f"  fold {k}: @0.5 {a5:.4f} | @tuned {at:.4f}  ({time.time()-t0:.0f}s)")
    out["cv_thr0.5_mean"], out["cv_thr0.5_std"] = float(np.mean(f05)), float(np.std(f05))
    out["cv_thrTuned_mean"], out["cv_thrTuned_std"] = float(np.mean(ftu)), float(np.std(ftu))
    out["cv_folds_thr0.5"], out["cv_folds_thrTuned"] = f05, ftu
    log(f"  CV @0.5   = {out['cv_thr0.5_mean']:.4f} +/- {out['cv_thr0.5_std']:.4f}   (paper 0.5806 +/- 0.0157)")
    log(f"  CV @tuned = {out['cv_thrTuned_mean']:.4f} +/- {out['cv_thrTuned_std']:.4f}")

    json.dump(out, open("/tmp/exp/stage3_results.json", "w"), indent=2, default=float)
    log("RESULTS " + json.dumps(out, default=float))
