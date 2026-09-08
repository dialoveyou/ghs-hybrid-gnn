"""P0: does the stacked model's reported gain survive a threshold-matched baseline?

Reproduces the published split and tree-ensemble hyperparameters exactly, then scores
the baseline under the protocol the paper used (fixed 0.5) and under the protocol the
GNN/stacked models were given (per-class thresholds tuned on validation).
"""
import os, sys, time, json, pickle
import numpy as np
from sklearn.metrics import f1_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import ExtraTreesClassifier
import lightgbm as lgb, xgboost as xgb, optuna
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED = 42
N_FP_BITS = 729 + 167
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

# published hyperparameters (results/metrics.json -> baseline_best_params)
PARAMS = {"lgb_n_estimators":398, "lgb_lr":0.06408476856634113, "lgb_depth":6,
          "xgb_n_estimators":316, "xgb_lr":0.19774084036922238, "xgb_depth":3,
          "et_n_estimators":399, "et_depth":18}

def get_models(p):
    return (MultiOutputClassifier(lgb.LGBMClassifier(n_estimators=p["lgb_n_estimators"],
                learning_rate=p["lgb_lr"], max_depth=p["lgb_depth"], verbosity=-1,
                random_state=SEED, n_jobs=2)),
            MultiOutputClassifier(xgb.XGBClassifier(n_estimators=p["xgb_n_estimators"],
                learning_rate=p["xgb_lr"], max_depth=p["xgb_depth"], eval_metric="logloss",
                random_state=SEED, n_jobs=2)),
            MultiOutputClassifier(ExtraTreesClassifier(n_estimators=p["et_n_estimators"],
                max_depth=p["et_depth"], random_state=SEED, n_jobs=2)))

def _p1(est, X):
    pr = est.predict_proba(X)
    cls = list(est.classes_)
    return pr[:, cls.index(1)] if 1 in cls else np.zeros(X.shape[0], dtype=np.float32)

def soft_vote(models, X):
    out = []
    for m in models:
        p = np.zeros((X.shape[0], len(m.estimators_)), dtype=np.float32)
        for c, est in enumerate(m.estimators_):
            p[:, c] = _p1(est, X)
        out.append(p)
    return np.mean(out, axis=0)

def macro_f1(y, proba, thr):
    return f1_score(y, (proba > thr).astype(int), average="macro", zero_division=0)

def thr_optuna(proba, y, n_trials=50):
    """exactly the paper's optimize_thresholds()"""
    t = np.full(proba.shape[1], 0.5)
    for c in range(proba.shape[1]):
        def obj(trial):
            v = trial.suggest_float("t", 0.05, 0.95)
            return f1_score(y[:, c], (proba[:, c] > v).astype(int), zero_division=0)
        s = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
        s.optimize(obj, n_trials=n_trials, show_progress_bar=False)
        t[c] = s.best_params["t"]
    return t

def thr_grid(proba, y, n=1000):
    """dense deterministic grid, as used for the §4.9 threshold sensitivity check"""
    grid = np.linspace(0.05, 0.95, n)
    t = np.full(proba.shape[1], 0.5)
    for c in range(proba.shape[1]):
        best, bt = -1.0, 0.5
        yc = y[:, c]; pc = proba[:, c]
        for v in grid:
            f = f1_score(yc, (pc > v).astype(int), zero_division=0)
            if f > best: best, bt = f, v
        t[c] = bt
    return t

if __name__ == "__main__":
    d = pickle.load(open("/tmp/exp/features.pkl", "rb"))
    feat2d_v, labels_v = d["feat2d_v"].copy(), d["labels_v"]
    log(f"loaded features: {feat2d_v.shape}")

    mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_idx, test_idx = list(mskf.split(feat2d_v, labels_v))[0]
    mskf2 = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    sub_tr, sub_va = list(mskf2.split(feat2d_v[train_idx], labels_v[train_idx]))[0]
    val_idx, train_idx = train_idx[sub_va], train_idx[sub_tr]
    log(f"split  train {len(train_idx)} | val {len(val_idx)} | test {len(test_idx)}   "
        f"(paper: 9061 / 2281 / 2803)")

    mu = feat2d_v[train_idx][:, N_FP_BITS:].mean(axis=0)
    sd = feat2d_v[train_idx][:, N_FP_BITS:].std(axis=0) + 1e-6
    feat2d_v[:, N_FP_BITS:] = (feat2d_v[:, N_FP_BITS:] - mu) / sd

    Xtr, Xva, Xte = feat2d_v[train_idx], feat2d_v[val_idx], feat2d_v[test_idx]
    ytr, yva, yte = labels_v[train_idx], labels_v[val_idx], labels_v[test_idx]

    res, store = {}, {}

    # ---- fit A: train+val (what the paper does for the reported baseline) ----
    log("fitting tree ensemble on train+val ...")
    t0 = time.time(); mA = get_models(PARAMS)
    for m in mA: m.fit(np.vstack([Xtr, Xva]), np.vstack([ytr, yva]))
    log(f"  done ({time.time()-t0:.0f}s)")
    teA, vaA = soft_vote(mA, Xte), soft_vote(mA, Xva)
    res["A_trainval_thr0.5"] = macro_f1(yte, teA, 0.5)
    log(f"  A  fit train+val, threshold 0.5      test macro-F1 = {res['A_trainval_thr0.5']:.4f}   <-- paper reports 0.5728")

    # ---- fit B: train only (thresholds tuned on a held-out val) ----
    log("fitting tree ensemble on train only ...")
    t0 = time.time(); mB = get_models(PARAMS)
    for m in mB: m.fit(Xtr, ytr)
    log(f"  done ({time.time()-t0:.0f}s)")
    teB, vaB = soft_vote(mB, Xte), soft_vote(mB, Xva)
    res["B_train_thr0.5"] = macro_f1(yte, teB, 0.5)
    log(f"  B0 fit train,     threshold 0.5      test macro-F1 = {res['B_train_thr0.5']:.4f}")

    log("optimizing per-class thresholds on validation (Optuna-50, the paper's procedure) ...")
    tB_opt = thr_optuna(vaB, yva); res["B_train_thrOptuna"] = macro_f1(yte, teB, tB_opt)
    log(f"  B1 fit train,     Optuna-50 thr      test macro-F1 = {res['B_train_thrOptuna']:.4f}")

    log("optimizing per-class thresholds on validation (dense 1000-point grid) ...")
    tB_grid = thr_grid(vaB, yva); res["B_train_thrGrid"] = macro_f1(yte, teB, tB_grid)
    log(f"  B2 fit train,     grid-1000 thr      test macro-F1 = {res['B_train_thrGrid']:.4f}")

    # ---- C: leakage probe — thresholds tuned on val using a model trained on val ----
    tC = thr_grid(vaA, yva); res["C_trainval_thrGrid_LEAKY"] = macro_f1(yte, teA, tC)
    log(f"  C  fit train+val, grid thr on val    test macro-F1 = {res['C_trainval_thrGrid_LEAKY']:.4f}  (leaky)")

    store = {"train_idx":train_idx, "val_idx":val_idx, "test_idx":test_idx,
             "yva":yva, "yte":yte,
             "baseline_trainonly_val":vaB, "baseline_trainonly_test":teB,
             "baseline_trainval_val":vaA, "baseline_trainval_test":teA,
             "thr_optuna":tB_opt, "thr_grid":tB_grid}
    pickle.dump(store, open("/tmp/exp/p0_store.pkl", "wb"))
    json.dump(res, open("/tmp/exp/p0_results.json", "w"), indent=2)
    log("saved p0_store.pkl / p0_results.json")
    log("RESULTS " + json.dumps(res))
