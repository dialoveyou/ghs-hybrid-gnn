#!/usr/bin/env python3
"""
Stage 2: split, augmentation, baseline tree ensemble (+Optuna), Hybrid GNN-ECC training,
threshold optimization, stacking blend, final evaluation. Reads prepped_data.pkl from stage 1.
Saves checkpoints/metrics incrementally so progress survives interruption.
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

import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log(f"Device: {DEVICE}")

with open(os.path.join(OUTDIR, "prepped_data.pkl"), "rb") as f:
    prepped = pickle.load(f)
df_v, labels_v, feat2d_v, graphs_v, CLASS_NAMES = (
    prepped["df_v"], prepped["labels_v"], prepped["feat2d_v"], prepped["graphs_v"], prepped["CLASS_NAMES"])
log(f"Loaded prepped data: {len(df_v)} molecules, feat2d dim {feat2d_v.shape[1]}")

# ---------- 4. Train/Val/Test split ----------
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
splits = list(mskf.split(feat2d_v, labels_v))
train_idx, test_idx = splits[0][0], splits[0][1]
mskf2 = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
tr_splits = list(mskf2.split(feat2d_v[train_idx], labels_v[train_idx]))
sub_train_idx, sub_val_idx = tr_splits[0]
val_idx = train_idx[sub_val_idx]
train_idx = train_idx[sub_train_idx]
log(f"Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

pos_counts = labels_v[train_idx].sum(axis=0)
neg_counts = len(train_idx) - pos_counts
pos_weight = torch.tensor(neg_counts / np.clip(pos_counts, 1, None), dtype=torch.float32)

N_FP_BITS = 729 + 167
desc_mean = feat2d_v[train_idx][:, N_FP_BITS:].mean(axis=0)
desc_std = feat2d_v[train_idx][:, N_FP_BITS:].std(axis=0) + 1e-6
feat2d_v[:, N_FP_BITS:] = (feat2d_v[:, N_FP_BITS:] - desc_mean) / desc_std
log("descriptor standardization done")

# ---------- 4.1 Rare-class conformer augmentation ----------
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
from rdkit.Chem import AllChem
from scipy.spatial import cKDTree

PROXIMITY_CUTOFF = 5.0
HYBRIDIZATION_MAP = {
    Chem.HybridizationType.SP: 0, Chem.HybridizationType.SP2: 1,
    Chem.HybridizationType.SP3: 2, Chem.HybridizationType.SP3D: 3,
    Chem.HybridizationType.SP3D2: 4,
}
def _atom_features(atom):
    hyb = HYBRIDIZATION_MAP.get(atom.GetHybridization(), 5)
    return [atom.GetAtomicNum(), atom.GetFormalCharge(), int(atom.GetIsAromatic()),
            hyb, atom.GetTotalNumHs(), int(atom.IsInRing()), atom.GetDegree(),
            atom.GetExplicitValence(), atom.GetMass()/100.0]

def compute_3d_graph_raw(mol, seed=SEED, proximity_cutoff=PROXIMITY_CUTOFF):
    molH = Chem.AddHs(Chem.Mol(mol))
    params = AllChem.ETKDGv3(); params.randomSeed = seed; params.useRandomCoords = True
    conf_id = AllChem.EmbedMolecule(molH, params)
    if conf_id < 0:
        return None
    try:
        AllChem.MMFFOptimizeMolecule(molH, maxIters=200)
    except Exception:
        pass
    mol_heavy = Chem.RemoveHs(molH)
    conf = mol_heavy.GetConformer()
    n_atoms = mol_heavy.GetNumAtoms()
    if n_atoms == 0:
        return None
    pos = np.array([list(conf.GetAtomPosition(i)) for i in range(n_atoms)], dtype=np.float32)
    x = np.array([_atom_features(a) for a in mol_heavy.GetAtoms()], dtype=np.float32)
    bonded_pairs = set()
    edge_src, edge_dst, edge_w = [], [], []
    for bond in mol_heavy.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        d = float(np.linalg.norm(pos[i]-pos[j])); w = 1.0/max(d, 1e-3)
        edge_src += [i, j]; edge_dst += [j, i]; edge_w += [w, w]
        bonded_pairs.add((min(i, j), max(i, j)))
    if n_atoms > 1:
        tree = cKDTree(pos)
        for i, j in tree.query_pairs(r=proximity_cutoff):
            key = (min(i, j), max(i, j))
            if key in bonded_pairs:
                continue
            d = float(np.linalg.norm(pos[i]-pos[j])); w = 1.0/max(d, 1e-3)
            edge_src += [i, j]; edge_dst += [j, i]; edge_w += [w, w]
    if not edge_src:
        edge_src, edge_dst, edge_w = [0],[0],[1.0]
    return x, np.array(edge_src), np.array(edge_dst), np.array(edge_w, dtype=np.float32)

RARE_CLASS_POS_THRESHOLD = 800
N_AUG_PER_SAMPLE = 3
FORCE_AUGMENT_CLASSES = [
    "05_Flammable_Solid", "08_Corrosive_to_Metals", "16_Mutagenicity",
    "17_Carcinogenicity", "18_Reproductive_Toxicity", "20_STOT_Repeated_Exposure",
]
rare_by_count = [CLASS_NAMES[c] for c in range(len(CLASS_NAMES)) if pos_counts[c] < RARE_CLASS_POS_THRESHOLD]
augment_target_names = sorted(set(rare_by_count) | set(FORCE_AUGMENT_CLASSES))
rare_classes = [CLASS_NAMES.index(n) for n in augment_target_names]
log(f"Augmentation target classes: {augment_target_names}")

AUG_CACHE = os.path.join(OUTDIR, "aug_cache.pkl")
if os.path.exists(AUG_CACHE):
    with open(AUG_CACHE, "rb") as f:
        aug_feat2d, aug_graphs, aug_labels = pickle.load(f)
    log(f"augmentation 캐시에서 로드: {len(aug_graphs)}개")
else:
    aug_feat2d, aug_graphs, aug_labels = [], [], []
    if rare_classes:
        rare_mask = labels_v[train_idx][:, rare_classes].sum(axis=1) > 0
        rare_local_positions = np.where(rare_mask)[0]
        log(f"증강 대상 train 샘플 수: {len(rare_local_positions)}")
        t0 = time.time()
        for k, local_pos in enumerate(rare_local_positions):
            global_row = train_idx[local_pos]
            smi = df_v.iloc[global_row]["SMILES"]
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            for aug_i in range(N_AUG_PER_SAMPLE):
                g = compute_3d_graph_raw(mol, seed=int(1000 + aug_i * 37 + global_row))
                if g is None:
                    continue
                aug_feat2d.append(feat2d_v[global_row]); aug_graphs.append(g); aug_labels.append(labels_v[global_row])
            if (k+1) % 200 == 0:
                log(f"  augmentation 진행: {k+1}/{len(rare_local_positions)} | 누적 생성 {len(aug_graphs)} | {(time.time()-t0):.0f}s")
        with open(AUG_CACHE, "wb") as f:
            pickle.dump((aug_feat2d, aug_graphs, aug_labels), f)
    log(f"추가된 augmented train 샘플: {len(aug_graphs)}")

# ---------- 5. Baseline: 2D Tree Ensemble + Optuna ----------
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score
import lightgbm as lgb
import xgboost as xgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

X_train, X_val, X_test = feat2d_v[train_idx], feat2d_v[val_idx], feat2d_v[test_idx]
y_train, y_val, y_test = labels_v[train_idx], labels_v[val_idx], labels_v[test_idx]

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

def soft_vote_proba(models, X):
    n_classes = y_train.shape[1]
    probas = []
    for m in models:
        p = np.zeros((X.shape[0], n_classes), dtype=np.float32)
        for c, est in enumerate(m.estimators_):
            p[:, c] = _safe_proba_1(est, X)
        probas.append(p)
    return np.mean(probas, axis=0)

BASELINE_STUDY_PATH = os.path.join(OUTDIR, "baseline_study.pkl")
N_TRIALS = 15
if os.path.exists(BASELINE_STUDY_PATH):
    with open(BASELINE_STUDY_PATH, "rb") as f:
        study = pickle.load(f)
    log(f"baseline Optuna study 캐시에서 로드 (best={study.best_value:.4f})")
else:
    def objective(trial):
        params = {
            "lgb_n_estimators": trial.suggest_int("lgb_n_estimators", 100, 400),
            "lgb_lr": trial.suggest_float("lgb_lr", 0.01, 0.2, log=True),
            "lgb_depth": trial.suggest_int("lgb_depth", 3, 8),
            "xgb_n_estimators": trial.suggest_int("xgb_n_estimators", 100, 400),
            "xgb_lr": trial.suggest_float("xgb_lr", 0.01, 0.2, log=True),
            "xgb_depth": trial.suggest_int("xgb_depth", 3, 8),
            "et_n_estimators": trial.suggest_int("et_n_estimators", 100, 400),
            "et_depth": trial.suggest_int("et_depth", 5, 20),
        }
        models = get_models(params)
        for m in models:
            m.fit(X_train, y_train)
        proba = soft_vote_proba(models, X_val)
        preds = (proba > 0.5).astype(int)
        f1 = f1_score(y_val, preds, average="macro", zero_division=0)
        log(f"  optuna trial done, val macro-F1={f1:.4f}")
        return f1
    t0 = time.time()
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    with open(BASELINE_STUDY_PATH, "wb") as f:
        pickle.dump(study, f)
    log(f"baseline Optuna 완료 ({time.time()-t0:.0f}s), best val macro-F1={study.best_value:.4f}")

save_metrics({"baseline_best_val_f1": study.best_value, "baseline_best_params": study.best_params})

best_models = get_models(study.best_params)
for m in best_models:
    m.fit(np.vstack([X_train, X_val]), np.vstack([y_train, y_val]))
test_proba_baseline = soft_vote_proba(best_models, X_test)
test_pred_baseline = (test_proba_baseline > 0.5).astype(int)
baseline_f1 = f1_score(y_test, test_pred_baseline, average="macro", zero_division=0)
log(f"Baseline (2D Tree Ensemble) Test Macro-F1: {baseline_f1:.4f} (paper: 0.6876/reported 0.608 in draft)")
save_metrics({"baseline_test_macro_f1": baseline_f1})

with open(os.path.join(OUTDIR, "baseline_models.pkl"), "wb") as f:
    pickle.dump(best_models, f)

# ---------- 6. Hybrid GNN-ECC ----------
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

train_feat2d = [feat2d_v[i] for i in train_idx] + aug_feat2d
train_graphs = [graphs_v[i] for i in train_idx] + aug_graphs
train_labels = [labels_v[i] for i in train_idx] + aug_labels
log(f"최종 train 샘플 수 (원본 {len(train_idx)} + 증강 {len(aug_graphs)}): {len(train_graphs)}")

train_ds = GHSGraphDataset(train_feat2d, train_graphs, train_labels)
val_ds = GHSGraphDataset([feat2d_v[i] for i in val_idx], [graphs_v[i] for i in val_idx], [labels_v[i] for i in val_idx])
test_ds = GHSGraphDataset([feat2d_v[i] for i in test_idx], [graphs_v[i] for i in test_idx], [labels_v[i] for i in test_idx])

train_labels_arr = np.stack(train_labels)
pos_counts_aug = train_labels_arr.sum(axis=0)
neg_counts_aug = len(train_labels_arr) - pos_counts_aug
pos_weight = torch.tensor(neg_counts_aug / np.clip(pos_counts_aug, 1, None), dtype=torch.float32)

class_freq = np.clip(pos_counts_aug, 1, None) / len(train_labels_arr)
inv_freq = 1.0 / class_freq
sample_weights = np.array([inv_freq[row.astype(bool)].max() if row.sum() > 0 else 1.0 for row in train_labels_arr])
sampler = torch.utils.data.WeightedRandomSampler(
    weights=torch.tensor(sample_weights, dtype=torch.double), num_samples=len(train_labels_arr), replacement=True)

train_loader = PyGDataLoader(train_ds, batch_size=64, sampler=sampler)
val_loader = PyGDataLoader(val_ds, batch_size=128, shuffle=False)
test_loader = PyGDataLoader(test_ds, batch_size=128, shuffle=False)
FEAT2D_DIM = feat2d_v.shape[1]
N_CLASSES = 22
FUSION_DIM = 256

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

class FocalLossWithLogits(nn.Module):
    def __init__(self, pos_weight=None, gamma=2.0, alpha=0.25):
        super().__init__(); self.pos_weight = pos_weight; self.gamma = gamma; self.alpha = alpha
    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight, reduction="none")
        p = torch.sigmoid(logits); p_t = p * targets + (1 - p) * (1 - targets)
        focal_term = (1 - p_t).clamp(min=1e-6) ** self.gamma
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * focal_term * bce).mean()

class HybridGNN_ECC(nn.Module):
    def __init__(self, feat2d_dim, atom_feat_dim=9, n_classes=N_CLASSES):
        super().__init__()
        self.branch3d = GNNBranch3D(in_dim=atom_feat_dim); self.branch2d = MLPBranch2D(in_dim=feat2d_dim)
        self.chain = DifferentiableECC(fusion_dim=FUSION_DIM, n_classes=n_classes)
    def forward(self, feat2d, graph_x, edge_index, edge_weight, batch_idx):
        h3d = self.branch3d(graph_x, edge_index, edge_weight, batch_idx)
        h2d = self.branch2d(feat2d)
        return self.chain(torch.cat([h2d, h3d], dim=1))

model = HybridGNN_ECC(feat2d_dim=FEAT2D_DIM).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=3e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)
criterion = FocalLossWithLogits(pos_weight=pos_weight.to(DEVICE), gamma=2.0, alpha=0.25)

def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    total_loss, all_logits, all_y = 0.0, [], []
    with torch.set_grad_enabled(train):
        for batch in loader:
            batch = batch.to(DEVICE)
            logits = model(batch.feat2d, batch.x, batch.edge_index, batch.edge_attr.squeeze(-1), batch.batch)
            loss = criterion(logits, batch.y)
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            all_logits.append(logits.detach().cpu()); all_y.append(batch.y.cpu())
    all_logits = torch.cat(all_logits); all_y = torch.cat(all_y)
    preds = (torch.sigmoid(all_logits) > 0.5).int().numpy()
    f1 = f1_score(all_y.numpy(), preds, average="macro", zero_division=0)
    return total_loss / len(loader.dataset), f1, all_logits, all_y

GNN_CKPT_PATH = os.path.join(OUTDIR, "gnn_checkpoint.pt")
start_epoch = 1
best_val_f1, best_state, patience, bad_epochs = -1, None, 6, 0
MIN_DELTA = 0.003
if os.path.exists(GNN_CKPT_PATH):
    ckpt = torch.load(GNN_CKPT_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["last_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    start_epoch = ckpt["epoch"] + 1
    best_val_f1 = ckpt["best_val_f1"]; best_state = ckpt["best_state"]; bad_epochs = ckpt["bad_epochs"]
    log(f"GNN 체크포인트에서 재시작: epoch {start_epoch}부터, best_val_f1={best_val_f1:.4f}")

EPOCHS = 40
for epoch in range(start_epoch, EPOCHS + 1):
    t0 = time.time()
    tr_loss, tr_f1, _, _ = run_epoch(train_loader, train=True)
    val_loss, val_f1, _, _ = run_epoch(val_loader, train=False)
    scheduler.step(val_f1)
    log(f"Epoch {epoch:02d} | train loss {tr_loss:.4f} f1 {tr_f1:.4f} | val loss {val_loss:.4f} f1 {val_f1:.4f} | {time.time()-t0:.0f}s")
    improved = val_f1 > best_val_f1 + MIN_DELTA
    if improved:
        best_val_f1, best_state, bad_epochs = val_f1, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
    else:
        bad_epochs += 1
    torch.save({"epoch": epoch, "last_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "best_val_f1": best_val_f1, "best_state": best_state, "bad_epochs": bad_epochs}, GNN_CKPT_PATH)
    save_metrics({"gnn_last_epoch": epoch, "gnn_best_val_f1": best_val_f1})
    if bad_epochs >= patience:
        log("Early stopping."); break

model.load_state_dict(best_state)
log(f"Best val macro-F1: {best_val_f1:.4f}")

# ---------- 7. Threshold optimization ----------
_, _, val_logits, val_y = run_epoch(val_loader, train=False)
val_proba = torch.sigmoid(val_logits).numpy(); val_y_np = val_y.numpy()

def optimize_thresholds(proba, y_true, n_trials=50):
    thresholds = np.full(proba.shape[1], 0.5)
    for c in range(proba.shape[1]):
        def obj(trial):
            t = trial.suggest_float("t", 0.05, 0.95)
            pred = (proba[:, c] > t).astype(int)
            return f1_score(y_true[:, c], pred, zero_division=0)
        study_c = optuna.create_study(direction="maximize")
        study_c.optimize(obj, n_trials=n_trials, show_progress_bar=False)
        thresholds[c] = study_c.best_params["t"]
    return thresholds

best_thresholds = optimize_thresholds(val_proba, val_y_np)
log("threshold 최적화 완료")

_, _, test_logits, test_y = run_epoch(test_loader, train=False)
test_proba = torch.sigmoid(test_logits).numpy(); test_y_np = test_y.numpy()

# ---------- 8. Stacking ----------
val_proba_baseline = soft_vote_proba(best_models, X_val)
test_proba_baseline = soft_vote_proba(best_models, X_test)

def optimize_blend_and_threshold(gnn_proba, baseline_proba, y_true, n_trials=40):
    n_classes = y_true.shape[1]
    alphas = np.zeros(n_classes); thresholds = np.zeros(n_classes)
    for c in range(n_classes):
        def obj(trial):
            a = trial.suggest_float("alpha", 0.0, 1.0); t = trial.suggest_float("t", 0.05, 0.95)
            blend = a * gnn_proba[:, c] + (1 - a) * baseline_proba[:, c]
            pred = (blend > t).astype(int)
            return f1_score(y_true[:, c], pred, zero_division=0)
        study_c = optuna.create_study(direction="maximize")
        study_c.optimize(obj, n_trials=n_trials, show_progress_bar=False)
        alphas[c] = study_c.best_params["alpha"]; thresholds[c] = study_c.best_params["t"]
    return alphas, thresholds

best_alphas, blend_thresholds = optimize_blend_and_threshold(val_proba, val_proba_baseline, val_y_np)
log("stacking blend 최적화 완료")

# ---------- 9. Final evaluation ----------
test_blend_proba = best_alphas * test_proba + (1 - best_alphas) * test_proba_baseline
pred_stacked = (test_blend_proba > blend_thresholds).astype(int)
pred_default = (test_proba > 0.5).astype(int)
pred_gnn_optimized = (test_proba > best_thresholds).astype(int)

f1_default = f1_score(test_y_np, pred_default, average="macro", zero_division=0)
f1_gnn_optimized = f1_score(test_y_np, pred_gnn_optimized, average="macro", zero_division=0)
f1_stacked = f1_score(test_y_np, pred_stacked, average="macro", zero_division=0)

log(f"Baseline (2D Tree Ensemble)               Macro-F1: {baseline_f1:.4f}")
log(f"Hybrid GNN-ECC  | default 0.5 threshold   Macro-F1: {f1_default:.4f}")
log(f"Hybrid GNN-ECC  | Optuna 최적 threshold    Macro-F1: {f1_gnn_optimized:.4f}")
log(f"Stacked (GNN + Baseline blend)             Macro-F1: {f1_stacked:.4f}")

from sklearn.metrics import matthews_corrcoef, roc_auc_score, balanced_accuracy_score, classification_report

def macro_mcc(y_true, y_pred):
    mccs = []
    for c in range(y_true.shape[1]):
        try: mcc = matthews_corrcoef(y_true[:, c], y_pred[:, c])
        except Exception: mcc = 0.0
        mccs.append(mcc)
    return np.array(mccs)

def macro_auc(y_true, y_proba):
    aucs = []
    for c in range(y_true.shape[1]):
        if len(np.unique(y_true[:, c])) < 2: continue
        try: aucs.append(roc_auc_score(y_true[:, c], y_proba[:, c]))
        except Exception: continue
    return np.array(aucs)

def macro_balanced_accuracy(y_true, y_pred):
    bas = []
    for c in range(y_true.shape[1]):
        try: bas.append(balanced_accuracy_score(y_true[:, c], y_pred[:, c]))
        except Exception: bas.append(0.0)
    return np.array(bas)

mcc_baseline = macro_mcc(test_y_np, (test_proba_baseline > 0.5).astype(int))
mcc_gnn = macro_mcc(test_y_np, pred_gnn_optimized)
mcc_stacked = macro_mcc(test_y_np, pred_stacked)
auc_baseline = macro_auc(test_y_np, test_proba_baseline)
auc_gnn = macro_auc(test_y_np, test_proba)
auc_stacked = macro_auc(test_y_np, test_blend_proba)
ba_baseline = macro_balanced_accuracy(test_y_np, (test_proba_baseline > 0.5).astype(int))
ba_gnn = macro_balanced_accuracy(test_y_np, pred_gnn_optimized)
ba_stacked = macro_balanced_accuracy(test_y_np, pred_stacked)

log(f"Baseline  Macro-MCC {mcc_baseline.mean():.4f}±{mcc_baseline.std():.4f} AUC {auc_baseline.mean():.4f} BA {ba_baseline.mean():.4f}")
log(f"GNN-ECC   Macro-MCC {mcc_gnn.mean():.4f}±{mcc_gnn.std():.4f} AUC {auc_gnn.mean():.4f} BA {ba_gnn.mean():.4f}")
log(f"Stacked   Macro-MCC {mcc_stacked.mean():.4f}±{mcc_stacked.std():.4f} AUC {auc_stacked.mean():.4f} BA {ba_stacked.mean():.4f}")

report = classification_report(test_y_np, pred_stacked, target_names=CLASS_NAMES, zero_division=0, output_dict=True)

final_results = {
    "baseline_test_macro_f1": float(baseline_f1),
    "gnn_default_macro_f1": float(f1_default),
    "gnn_optthr_macro_f1": float(f1_gnn_optimized),
    "stacked_macro_f1": float(f1_stacked),
    "baseline_mcc_mean": float(mcc_baseline.mean()), "baseline_mcc_std": float(mcc_baseline.std()),
    "gnn_mcc_mean": float(mcc_gnn.mean()), "gnn_mcc_std": float(mcc_gnn.std()),
    "stacked_mcc_mean": float(mcc_stacked.mean()), "stacked_mcc_std": float(mcc_stacked.std()),
    "baseline_auc_mean": float(auc_baseline.mean()), "gnn_auc_mean": float(auc_gnn.mean()), "stacked_auc_mean": float(auc_stacked.mean()),
    "baseline_ba_mean": float(ba_baseline.mean()), "gnn_ba_mean": float(ba_gnn.mean()), "stacked_ba_mean": float(ba_stacked.mean()),
    "per_class_report_stacked": report,
    "best_alphas": {CLASS_NAMES[i]: float(best_alphas[i]) for i in range(len(CLASS_NAMES))},
    "blend_thresholds": {CLASS_NAMES[i]: float(blend_thresholds[i]) for i in range(len(CLASS_NAMES))},
    "gnn_thresholds": {CLASS_NAMES[i]: float(best_thresholds[i]) for i in range(len(CLASS_NAMES))},
}
save_metrics({"stage2_final_results": final_results})

torch.save({"model_state_dict": model.state_dict(), "gnn_thresholds": best_thresholds,
            "stack_alphas": best_alphas, "stack_thresholds": blend_thresholds, "class_names": CLASS_NAMES},
           os.path.join(OUTDIR, "hybrid_gnn_ecc_ghs22.pt"))
log("저장 완료: hybrid_gnn_ecc_ghs22.pt  STAGE2_DONE")
