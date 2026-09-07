#!/usr/bin/env python3
"""
Stage 3: SCI-grade validation extras - 5-fold CV (baseline), scaffold split,
bootstrap significance, SHAP. Reuses stage 1/2 artifacts (no retraining of stage2 core).
"""
import os, time, json, random, pickle, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

WORKDIR = os.environ.get("GHS_WORKDIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "work"))
OUTDIR = os.path.join(WORKDIR, "outputs")
METRICS_PATH = os.path.join(OUTDIR, "metrics.json")
SEED = 42
random.seed(SEED); np.random.seed(SEED)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def save_metrics(update):
    m = {}
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            m = json.load(f)
    m.update(update)
    with open(METRICS_PATH, "w") as f:
        json.dump(m, f, indent=2, default=float)
    log(f"metrics checkpoint saved: {list(update.keys())}")

import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(SEED)
DEVICE = torch.device("cpu")

with open(os.path.join(OUTDIR, "prepped_data.pkl"), "rb") as f:
    prepped = pickle.load(f)
df_v, labels_v, feat2d_v, graphs_v, CLASS_NAMES = (
    prepped["df_v"], prepped["labels_v"], prepped["feat2d_v"], prepped["graphs_v"], prepped["CLASS_NAMES"])

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
splits = list(mskf.split(feat2d_v, labels_v))
train_idx, test_idx = splits[0][0], splits[0][1]
mskf2 = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
tr_splits = list(mskf2.split(feat2d_v[train_idx], labels_v[train_idx]))
sub_train_idx, sub_val_idx = tr_splits[0]
val_idx = train_idx[sub_val_idx]
train_idx = train_idx[sub_train_idx]

N_FP_BITS = 729 + 167
desc_mean = feat2d_v[train_idx][:, N_FP_BITS:].mean(axis=0)
desc_std = feat2d_v[train_idx][:, N_FP_BITS:].std(axis=0) + 1e-6
feat2d_v[:, N_FP_BITS:] = (feat2d_v[:, N_FP_BITS:] - desc_mean) / desc_std

X_train, X_val, X_test = feat2d_v[train_idx], feat2d_v[val_idx], feat2d_v[test_idx]
y_train, y_val, y_test = labels_v[train_idx], labels_v[val_idx], labels_v[test_idx]
test_y_np = y_test

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score
import lightgbm as lgb
import xgboost as xgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

with open(os.path.join(OUTDIR, "baseline_study.pkl"), "rb") as f:
    study = pickle.load(f)
with open(os.path.join(OUTDIR, "baseline_models.pkl"), "rb") as f:
    best_models = pickle.load(f)
log(f"loaded baseline study (best val f1={study.best_value:.4f}) and fitted models")

def get_models(params):
    lgbm = MultiOutputClassifier(lgb.LGBMClassifier(
        n_estimators=params["lgb_n_estimators"], learning_rate=params["lgb_lr"],
        max_depth=params["lgb_depth"], verbosity=-1, random_state=SEED, n_jobs=2))
    xgboost = MultiOutputClassifier(xgb.XGBClassifier(
        n_estimators=params["xgb_n_estimators"], learning_rate=params["xgb_lr"],
        max_depth=params["xgb_depth"], eval_metric="logloss", random_state=SEED, n_jobs=2))
    extra = MultiOutputClassifier(ExtraTreesClassifier(
        n_estimators=params["et_n_estimators"], max_depth=params["et_depth"],
        random_state=SEED, n_jobs=2))
    return lgbm, xgboost, extra

def _safe_proba_1(estimator, X):
    proba = estimator.predict_proba(X)
    classes = list(estimator.classes_)
    if 1 in classes:
        return proba[:, classes.index(1)]
    return np.zeros(X.shape[0], dtype=np.float32)

def soft_vote_proba(models, X, n_classes):
    probas = []
    for m in models:
        p = np.zeros((X.shape[0], n_classes), dtype=np.float32)
        for c, est in enumerate(m.estimators_):
            p[:, c] = _safe_proba_1(est, X)
        probas.append(p)
    return np.mean(probas, axis=0)

test_proba_baseline = soft_vote_proba(best_models, X_test, labels_v.shape[1])
baseline_f1 = f1_score(y_test, (test_proba_baseline > 0.5).astype(int), average="macro", zero_division=0)
log(f"recomputed baseline test macro-F1: {baseline_f1:.4f} (sanity check)")

# reconstruct GNN model + probas from saved checkpoint
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import GCNConv, global_add_pool

class GHSGraphDataset(Dataset):
    def __init__(self, feat2d_list, graphs_list, labels_list):
        super().__init__()
        self.feat2d_list = feat2d_list; self.graphs_list = graphs_list; self.labels_list = labels_list
    def len(self): return len(self.feat2d_list)
    def get(self, i):
        x, es, ed, ew = self.graphs_list[i]
        data = Data(x=torch.tensor(x, dtype=torch.float32),
                    edge_index=torch.tensor(np.vstack([es, ed]), dtype=torch.long),
                    edge_attr=torch.tensor(ew, dtype=torch.float32).unsqueeze(-1))
        data.feat2d = torch.tensor(self.feat2d_list[i], dtype=torch.float32).unsqueeze(0)
        data.y = torch.tensor(self.labels_list[i], dtype=torch.float32).unsqueeze(0)
        return data

test_ds = GHSGraphDataset([feat2d_v[i] for i in test_idx], [graphs_v[i] for i in test_idx], [labels_v[i] for i in test_idx])
test_loader = PyGDataLoader(test_ds, batch_size=128, shuffle=False)
FEAT2D_DIM = feat2d_v.shape[1]; N_CLASSES = 22; FUSION_DIM = 256

class GNNBranch3D(nn.Module):
    def __init__(self, in_dim=9, hidden=64, out_dim=128, dropout=0.2):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden); self.conv2 = GCNConv(hidden, hidden); self.conv3 = GCNConv(hidden, out_dim)
        self.bn1 = nn.BatchNorm1d(hidden); self.bn2 = nn.BatchNorm1d(hidden); self.dropout = nn.Dropout(dropout)
    def forward(self, x, edge_index, edge_weight, batch):
        h = F.relu(self.bn1(self.conv1(x, edge_index, edge_weight))); h = self.dropout(h)
        h = F.relu(self.bn2(self.conv2(h, edge_index, edge_weight))); h = self.dropout(h)
        h = self.conv3(h, edge_index, edge_weight)
        return global_add_pool(h, batch)

class MLPBranch2D(nn.Module):
    def __init__(self, in_dim, hidden=256, out_dim=128, dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU())
    def forward(self, x): return self.net(x)

class DifferentiableECC(nn.Module):
    def __init__(self, fusion_dim=FUSION_DIM, n_classes=N_CLASSES, head_hidden=64):
        super().__init__()
        self.heads = nn.ModuleList([nn.Sequential(nn.Linear(fusion_dim + i, head_hidden), nn.ReLU(), nn.Linear(head_hidden, 1)) for i in range(n_classes)])
    def forward(self, fusion_repr):
        logits = []; chain_input = fusion_repr
        for head in self.heads:
            logit = head(chain_input); logits.append(logit); chain_input = torch.cat([chain_input, logit], dim=1)
        return torch.cat(logits, dim=1)

class HybridGNN_ECC(nn.Module):
    def __init__(self, feat2d_dim, atom_feat_dim=9, n_classes=N_CLASSES):
        super().__init__()
        self.branch3d = GNNBranch3D(in_dim=atom_feat_dim); self.branch2d = MLPBranch2D(in_dim=feat2d_dim)
        self.chain = DifferentiableECC(fusion_dim=FUSION_DIM, n_classes=n_classes)
    def forward(self, feat2d, graph_x, edge_index, edge_weight, batch_idx):
        h3d = self.branch3d(graph_x, edge_index, edge_weight, batch_idx)
        h2d = self.branch2d(feat2d)
        return self.chain(torch.cat([h2d, h3d], dim=1))

ckpt = torch.load(os.path.join(OUTDIR, "hybrid_gnn_ecc_ghs22.pt"), map_location=DEVICE, weights_only=False)
model = HybridGNN_ECC(feat2d_dim=FEAT2D_DIM).to(DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
best_thresholds = ckpt["gnn_thresholds"]; best_alphas = ckpt["stack_alphas"]; blend_thresholds = ckpt["stack_thresholds"]

all_logits = []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(DEVICE)
        logits = model(batch.feat2d, batch.x, batch.edge_index, batch.edge_attr.squeeze(-1), batch.batch)
        all_logits.append(logits.cpu())
test_proba = torch.sigmoid(torch.cat(all_logits)).numpy()
test_blend_proba = best_alphas * test_proba + (1 - best_alphas) * test_proba_baseline
pred_stacked = (test_blend_proba > blend_thresholds).astype(int)
f1_stacked = f1_score(test_y_np, pred_stacked, average="macro", zero_division=0)
log(f"recomputed stacked test macro-F1: {f1_stacked:.4f} (sanity check)")

# ---------- 11.1 5-Fold CV Baseline ----------
CV_PATH = os.path.join(OUTDIR, "cv_results.json")
if os.path.exists(CV_PATH):
    with open(CV_PATH) as f:
        cv_data = json.load(f)
    cv_f1_scores = np.array(cv_data["fold_f1s"])
    log(f"5-fold CV 캐시에서 로드: {cv_f1_scores}")
else:
    cv_f1_scores = []
    mskf_cv = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (tr_idx, te_idx) in enumerate(mskf_cv.split(feat2d_v, labels_v)):
        t0 = time.time()
        fold_models = get_models(study.best_params)
        for m in fold_models:
            m.fit(feat2d_v[tr_idx], labels_v[tr_idx])
        proba = soft_vote_proba(fold_models, feat2d_v[te_idx], labels_v.shape[1])
        pred = (proba > 0.5).astype(int)
        f1 = f1_score(labels_v[te_idx], pred, average="macro", zero_division=0)
        cv_f1_scores.append(f1)
        log(f"Fold {fold+1}/5: Macro F1 = {f1:.4f} ({time.time()-t0:.0f}s)")
        with open(CV_PATH, "w") as f:
            json.dump({"fold_f1s": cv_f1_scores}, f)
    cv_f1_scores = np.array(cv_f1_scores)

log(f"5-Fold CV Baseline Macro F1: {cv_f1_scores.mean():.4f} ± {cv_f1_scores.std():.4f}")
save_metrics({"cv_baseline_f1_mean": float(cv_f1_scores.mean()), "cv_baseline_f1_std": float(cv_f1_scores.std()),
              "cv_baseline_folds": [float(x) for x in cv_f1_scores]})

# ---------- 11.2 Scaffold Split ----------
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import GroupShuffleSplit

SCAFFOLD_PATH = os.path.join(OUTDIR, "scaffold_result.json")
if os.path.exists(SCAFFOLD_PATH):
    with open(SCAFFOLD_PATH) as f:
        scaffold_data = json.load(f)
    scaffold_f1 = scaffold_data["scaffold_f1"]
    log(f"scaffold split 캐시에서 로드: F1={scaffold_f1:.4f}")
else:
    def get_murcko_scaffold(smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
            return scaffold if scaffold else smiles
        except Exception:
            return smiles
    scaffolds = df_v["SMILES"].apply(get_murcko_scaffold).values
    n_unique_scaffolds = len(set(scaffolds))
    log(f"전체 {len(df_v)}개 화합물 / 고유 scaffold {n_unique_scaffolds}개")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    scaffold_train_idx, scaffold_test_idx = next(gss.split(feat2d_v, labels_v, groups=scaffolds))
    log(f"Scaffold split: Train {len(scaffold_train_idx)} / Test {len(scaffold_test_idx)}")
    t0 = time.time()
    scaffold_models = get_models(study.best_params)
    for m in scaffold_models:
        m.fit(feat2d_v[scaffold_train_idx], labels_v[scaffold_train_idx])
    scaffold_proba = soft_vote_proba(scaffold_models, feat2d_v[scaffold_test_idx], labels_v.shape[1])
    scaffold_pred = (scaffold_proba > 0.5).astype(int)
    scaffold_f1 = f1_score(labels_v[scaffold_test_idx], scaffold_pred, average="macro", zero_division=0)
    log(f"Scaffold split baseline test F1: {scaffold_f1:.4f} ({time.time()-t0:.0f}s)")
    with open(SCAFFOLD_PATH, "w") as f:
        json.dump({"scaffold_f1": float(scaffold_f1), "n_unique_scaffolds": int(n_unique_scaffolds),
                    "n_train": int(len(scaffold_train_idx)), "n_test": int(len(scaffold_test_idx))}, f)

log(f"Random Split Baseline Test Macro F1  : {baseline_f1:.4f}")
log(f"Scaffold Split Baseline Test Macro F1: {scaffold_f1:.4f}")
save_metrics({"scaffold_baseline_f1": float(scaffold_f1)})

# ---------- 11.4 Bootstrap significance ----------
def bootstrap_f1_diff(y_true, proba_a, thr_a, proba_b, thr_b, n_boot=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    n = y_true.shape[0]
    diffs = np.zeros(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        pred_a = (proba_a[idx] > thr_a).astype(int)
        pred_b = (proba_b[idx] > thr_b).astype(int)
        f1_a = f1_score(y_true[idx], pred_a, average="macro", zero_division=0)
        f1_b = f1_score(y_true[idx], pred_b, average="macro", zero_division=0)
        diffs[i] = f1_b - f1_a
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    p_like = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return diffs.mean(), ci_low, ci_high, p_like

pairs = [
    ("Baseline -> GNN", test_proba_baseline, np.full(N_CLASSES, 0.5), test_proba, best_thresholds),
    ("Baseline -> Stacked", test_proba_baseline, np.full(N_CLASSES, 0.5), test_blend_proba, blend_thresholds),
    ("GNN -> Stacked", test_proba, best_thresholds, test_blend_proba, blend_thresholds),
]
bootstrap_results = {}
for name, pa, ta, pb, tb in pairs:
    mean_diff, lo, hi, p = bootstrap_f1_diff(test_y_np, pa, ta, pb, tb, n_boot=1000)
    sig = "유의함" if (lo > 0 or hi < 0) else "유의하지 않음"
    log(f"{name:<22} mean_diff={mean_diff:+.4f} CI=[{lo:+.4f},{hi:+.4f}] p={p:.4f} ({sig})")
    bootstrap_results[name] = {"mean_diff": float(mean_diff), "ci_low": float(lo), "ci_high": float(hi), "p": float(p)}
save_metrics({"bootstrap_results": bootstrap_results})

# ---------- 11.5 SHAP ----------
import shap
SHAP_PATH = os.path.join(OUTDIR, "shap_result.json")
DESCRIPTOR_NAMES = [
    "MolWt", "LogP", "TPSA", "NumHDonors", "NumHAcceptors", "NumRotatableBonds",
    "RingCount", "NumAromaticRings", "FractionCSP3", "NumHeteroatoms",
    "HeavyAtomCount", "NumRadicalElectrons", "NumSaturatedRings", "NumAliphaticRings",
    "NHOHCount", "NOCount", "MolMR", "BalabanJ", "BertzCT", "LabuteASA",
]
if os.path.exists(SHAP_PATH):
    log("SHAP 캐시에서 로드")
    with open(SHAP_PATH) as f:
        shap_summary = json.load(f)
else:
    SHAP_SAMPLE_CLASSES = ["14_Eye_Damage_Irritation", "16_Mutagenicity", "08_Corrosive_to_Metals"]
    X_test_sample = X_test[:300]
    shap_summary = {}
    for cls_name in SHAP_SAMPLE_CLASSES:
        t0 = time.time()
        cls_idx = CLASS_NAMES.index(cls_name)
        lgb_estimator = best_models[0].estimators_[cls_idx]
        if 1 not in list(lgb_estimator.classes_):
            log(f"{cls_name}: skip (no positive samples in train fold)")
            continue
        explainer = shap.TreeExplainer(lgb_estimator)
        shap_values = explainer.shap_values(X_test_sample)
        sv = shap_values[1] if isinstance(shap_values, list) else shap_values
        mean_abs_shap = np.abs(sv).mean(axis=0)
        fp_importance = float(mean_abs_shap[:N_FP_BITS].sum())
        desc_importance = mean_abs_shap[N_FP_BITS:]
        top_desc_idx = np.argsort(desc_importance)[::-1][:5]
        shap_summary[cls_name] = {
            "fingerprint_total": fp_importance,
            "descriptor_total": float(desc_importance.sum()),
            "top_descriptors": [(DESCRIPTOR_NAMES[i], float(desc_importance[i])) for i in top_desc_idx],
        }
        log(f"SHAP {cls_name}: fp={fp_importance:.3f} desc={desc_importance.sum():.3f} ({time.time()-t0:.0f}s)")
    with open(SHAP_PATH, "w") as f:
        json.dump(shap_summary, f, indent=2)

save_metrics({"shap_summary": shap_summary})
log("STAGE3_DONE")
