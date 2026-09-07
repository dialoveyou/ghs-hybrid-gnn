#!/usr/bin/env python3
"""
2D-3D Hybrid Differentiable Classifier Chain Network for 22-Class GHS Hazard Prediction
전체 파이프라인 단일 스크립트 버전 (Colab에서 통째로 실행 가능)

실행 순서:
  1) chemical_risk_database.csv를 같은 디렉토리에 두거나 CSV_PATH 수정
  2) Colab 런타임을 GPU로 설정
  3) pip install 라인은 주석 처리되어 있음 - Colab 셀에서 먼저 한 번 실행 필요:
     !pip install -q rdkit torch torch_geometric lightgbm xgboost optuna
     !pip install -q iterative-stratification shap
  4) 처음 실행이거나 코드를 바꿨다면 ghs_features_cache.pkl 삭제 후 재실행
"""

# # 2D-3D Hybrid Differentiable Classifier Chain Network for 22-Class GHS Hazard Prediction
#
# 논문 재현 노트북 (Colab 실행용). 순서대로 셀을 실행하세요.
#
# 1. 환경 설정 및 패키지 설치
# 2. 데이터 로드 및 22-Class 타겟 재정의
# 3. 2D/3D 피처 엔지니어링 (멀티프로세싱 + 캐싱)
# 4. Train/Val/Test 분할 (Multilabel Stratified)
# 5. Baseline: 2D Tree Ensemble (LightGBM+XGBoost+ExtraTrees) + Optuna
# 6. Hybrid GNN-ECC 모델 학습
# 7. Optuna 기반 클래스별 결정 임계값 최적화
# 8. 최종 평가 및 비교
#

# ## 1. 환경 설정

# # Colab 환경 패키지 설치 (최초 1회 실행 후 런타임 재시작 권장 - RDKit/PyG 충돌 방지)
# !pip install -q rdkit
# !pip install -q torch torch_geometric
# !pip install -q lightgbm xgboost optuna iterative-stratification
#

import os, random, time, json, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)


# ## 2. 데이터 로드 및 22-Class 타겟 재정의
#
# 원본 CSV의 `Y_GHS_H_Codes`(73종 raw H-code)를 물리적/작용기전이 유사한 코드끼리 병합하여
# 22개 핵심 유해성 클래스로 재정의합니다 (논문 2.1절 "STOT 단일노출 지표 병합"과 동일한 원칙).
#
# > Colab에서 실행 시, 아래 셀 실행 전에 `chemical_risk_database.csv`를 좌측 파일 탭에 업로드하거나
# > Google Drive를 마운트해서 경로를 맞춰주세요.

# Colab에 파일 업로드하는 경우:
# from google.colab import files
# uploaded = files.upload()

CSV_PATH = "chemical_risk_database.csv"  # 필요시 경로 수정
df = pd.read_csv(CSV_PATH)
print(df.shape)
df.head(3)


GHS_22_CLASS_MAP = {
    # --- 물리적 유해성 (Physical Hazards) ---
    "01_Explosive_SelfReactive": ["H200","H201","H202","H203","H204","H205","H240","H241","H242"],
    "02_Flammable_Gas": ["H220","H221","H230","H231"],
    "03_Gas_Under_Pressure": ["H280","H281"],
    "04_Flammable_Liquid": ["H224","H225","H226","H227"],
    "05_Flammable_Solid": ["H228"],
    "06_Spontaneous_Combustion": ["H250","H251","H252","H260","H261"],
    "07_Oxidizing": ["H270","H271","H272"],
    "08_Corrosive_to_Metals": ["H290"],
    # --- 건강 유해성 (Health Hazards) ---
    "09_Acute_Toxicity_Oral": ["H300","H301","H302","H303"],
    "10_Acute_Toxicity_Dermal": ["H310","H311","H312","H313"],
    "11_Acute_Toxicity_Inhalation": ["H330","H331","H332","H333"],
    "12_Aspiration_Hazard": ["H304","H305"],
    "13_Skin_Corrosion_Irritation": ["H314","H315","H316"],
    "14_Eye_Damage_Irritation": ["H318","H319","H320"],
    "15_Sensitization": ["H317","H334"],
    "16_Mutagenicity": ["H340","H341"],
    "17_Carcinogenicity": ["H350","H351"],
    "18_Reproductive_Toxicity": ["H360","H361","H362"],
    "19_STOT_Single_Exposure_Merged": ["H335","H336","H370","H371"],
    "20_STOT_Repeated_Exposure": ["H372","H373"],
    # --- 환경 유해성 (Environmental Hazards) ---
    "21_Aquatic_Hazard": ["H400","H401","H402","H410","H411","H412","H413"],
    "22_Ozone_Layer_Hazard": ["H420"],
}
assert len(GHS_22_CLASS_MAP) == 22
CLASS_NAMES = list(GHS_22_CLASS_MAP.keys())
HCODE_TO_CLASS = {c: cls for cls, codes in GHS_22_CLASS_MAP.items() for c in codes}

def build_label_vector(hcode_str):
    vec = np.zeros(len(CLASS_NAMES), dtype=np.float32)
    if pd.isna(hcode_str):
        return vec
    for c in hcode_str.split(","):
        c = c.strip()
        if c in HCODE_TO_CLASS:
            vec[CLASS_NAMES.index(HCODE_TO_CLASS[c])] = 1.0
    return vec

labels = np.stack(df["Y_GHS_H_Codes"].apply(build_label_vector).values)
has_label = labels.sum(axis=1) > 0
df = df[has_label].reset_index(drop=True)
labels = labels[has_label]
print(f"Chemicals with >=1 GHS label: {len(df)}")  # 논문 수치 14,148과 비교

label_df = pd.DataFrame(labels, columns=CLASS_NAMES)
label_df.sum().sort_values().plot(kind="barh", figsize=(8,7), title="Class distribution (label count)")


# ## 3. 2D/3D 피처 엔지니어링
#
# - **2D**: Morgan Fingerprint(729-bit, radius=2) + MACCS Keys(167-bit) → 896차원 dense 벡터
# - **3D**: ETKDGv3로 conformer 생성 → (1) 결합(bond) 기반 엣지 + (2) **비결합 원자쌍이라도
#   3D 공간상 5Å 이내면 근접 엣지로 추가**. 결합 위상만 쓰면 사실상 2D 그래프와 다를 게 없어서,
#   분자가 3차원에서 어떻게 접히는지(스테릭 힌드런스, 국소 입체구조)를 GNN이 보게 하려면
#   비결합 근접 정보가 필요합니다 — 이게 CMR(발암성/변이원성/생식독성)류 recall 개선의 핵심입니다.
#   edge_attr = 1/유클리디안거리(원자간). 그래프는 heavy atom만 사용(H 개수는 원자 피처에 포함).
#
# 분자당 약 50~150ms 소요되므로 멀티프로세싱으로 병렬 처리하고 결과를 캐싱합니다. 3D 임베딩에
# 실패하는 분자(극소수)는 학습셋에서 제외합니다.

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, MACCSkeys, Descriptors, Crippen, rdMolDescriptors
from scipy.spatial import cKDTree
RDLogger.DisableLog("rdApp.*")

MORGAN_BITS = 729
MORGAN_RADIUS = 2
PROXIMITY_CUTOFF = 5.0  # Angstrom, 비결합 근접 엣지 컷오프
HYBRIDIZATION_MAP = {
    Chem.HybridizationType.SP: 0, Chem.HybridizationType.SP2: 1,
    Chem.HybridizationType.SP3: 2, Chem.HybridizationType.SP3D: 3,
    Chem.HybridizationType.SP3D2: 4,
}

DESCRIPTOR_NAMES = [
    "MolWt", "LogP", "TPSA", "NumHDonors", "NumHAcceptors", "NumRotatableBonds",
    "RingCount", "NumAromaticRings", "FractionCSP3", "NumHeteroatoms",
    "HeavyAtomCount", "NumRadicalElectrons", "NumSaturatedRings", "NumAliphaticRings",
    "NHOHCount", "NOCount", "MolMR", "BalabanJ", "BertzCT", "LabuteASA",
]
N_DESC = len(DESCRIPTOR_NAMES)

def _compute_descriptors(mol):
    try:
        return np.array([
            Descriptors.MolWt(mol), Crippen.MolLogP(mol), rdMolDescriptors.CalcTPSA(mol),
            Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol), Descriptors.NumRotatableBonds(mol),
            rdMolDescriptors.CalcNumRings(mol), rdMolDescriptors.CalcNumAromaticRings(mol),
            rdMolDescriptors.CalcFractionCSP3(mol), rdMolDescriptors.CalcNumHeteroatoms(mol),
            mol.GetNumHeavyAtoms(), Descriptors.NumRadicalElectrons(mol),
            rdMolDescriptors.CalcNumSaturatedRings(mol), rdMolDescriptors.CalcNumAliphaticRings(mol),
            Descriptors.NHOHCount(mol), Descriptors.NOCount(mol), Crippen.MolMR(mol),
            Descriptors.BalabanJ(mol), Descriptors.BertzCT(mol), rdMolDescriptors.CalcLabuteASA(mol),
        ], dtype=np.float32)
    except Exception:
        return np.zeros(N_DESC, dtype=np.float32)

def compute_2d_fingerprint(mol):
    """Morgan(729) + MACCS(167) + 물성 기술자(20) = 916차원.
    물성 기술자는 Corrosive_to_Metals, Flammable_Solid 같이 지문(fingerprint) 패턴만으로는
    포착하기 어려운 반응성/극성 관련 신호를 보완합니다."""
    morgan = AllChem.GetMorganFingerprintAsBitVect(mol, MORGAN_RADIUS, nBits=MORGAN_BITS)
    maccs = MACCSkeys.GenMACCSKeys(mol)
    morgan_arr = np.zeros((MORGAN_BITS,), dtype=np.float32)
    Chem.DataStructs.ConvertToNumpyArray(morgan, morgan_arr)
    maccs_arr = np.zeros((167,), dtype=np.float32)
    Chem.DataStructs.ConvertToNumpyArray(maccs, maccs_arr)
    desc_arr = _compute_descriptors(mol)
    return np.concatenate([morgan_arr, maccs_arr, desc_arr])

def _atom_features(atom):
    hyb = HYBRIDIZATION_MAP.get(atom.GetHybridization(), 5)
    return [atom.GetAtomicNum(), atom.GetFormalCharge(), int(atom.GetIsAromatic()),
            hyb, atom.GetTotalNumHs(), int(atom.IsInRing()), atom.GetDegree(),
            atom.GetExplicitValence(), atom.GetMass()/100.0]

def compute_3d_graph_raw(mol, seed=SEED, proximity_cutoff=PROXIMITY_CUTOFF):
    """반환: (atom_features, edge_src, edge_dst, edge_weight) 또는 None.
    결합 엣지 + 비결합 근접 엣지(5A 이내)를 함께 포함."""
    molH = Chem.AddHs(Chem.Mol(mol))
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.useRandomCoords = True
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
        d = float(np.linalg.norm(pos[i]-pos[j]))
        w = 1.0/max(d, 1e-3)
        edge_src += [i, j]; edge_dst += [j, i]; edge_w += [w, w]
        bonded_pairs.add((min(i, j), max(i, j)))

    if n_atoms > 1:
        tree = cKDTree(pos)
        for i, j in tree.query_pairs(r=proximity_cutoff):
            key = (min(i, j), max(i, j))
            if key in bonded_pairs:
                continue
            d = float(np.linalg.norm(pos[i]-pos[j]))
            w = 1.0/max(d, 1e-3)
            edge_src += [i, j]; edge_dst += [j, i]; edge_w += [w, w]

    if not edge_src:
        edge_src, edge_dst, edge_w = [0],[0],[1.0]
    return x, np.array(edge_src), np.array(edge_dst), np.array(edge_w, dtype=np.float32)

def process_one(args):
    idx, smiles = args
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return idx, None, None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return idx, None, None
    fp = compute_2d_fingerprint(mol)
    g = compute_3d_graph_raw(mol)
    if g is None:
        return idx, fp, None
    return idx, fp, g


from multiprocessing import Pool
import pickle

CACHE_PATH = "ghs_features_cache.pkl"

if os.path.exists(CACHE_PATH):
    print("캐시에서 로드...")
    with open(CACHE_PATH, "rb") as f:
        feat2d_dict, graph_dict = pickle.load(f)
else:
    feat2d_dict, graph_dict = {}, {}
    args = list(enumerate(df["SMILES"].tolist()))
    with Pool(processes=os.cpu_count()) as pool:
        for idx, fp, g in tqdm(pool.imap(process_one, args, chunksize=32), total=len(args)):
            if fp is not None:
                feat2d_dict[idx] = fp
            if g is not None:
                graph_dict[idx] = g
    with open(CACHE_PATH, "wb") as f:
        pickle.dump((feat2d_dict, graph_dict), f)

valid_idx = sorted(set(feat2d_dict.keys()) & set(graph_dict.keys()))
print(f"2D 성공: {len(feat2d_dict)} | 3D 성공: {len(graph_dict)} | 둘 다 성공(학습 사용): {len(valid_idx)}")

df_v = df.iloc[valid_idx].reset_index(drop=True)
labels_v = labels[valid_idx]
feat2d_v = np.stack([feat2d_dict[i] for i in valid_idx])
graphs_v = [graph_dict[i] for i in valid_idx]


# ## 4. Train / Val / Test 분할 (Multilabel Stratified)

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
splits = list(mskf.split(feat2d_v, labels_v))
train_idx, test_idx = splits[0][0], splits[0][1]

# train을 다시 train/val로 분할
mskf2 = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
tr_splits = list(mskf2.split(feat2d_v[train_idx], labels_v[train_idx]))
sub_train_idx, sub_val_idx = tr_splits[0]
val_idx = train_idx[sub_val_idx]
train_idx = train_idx[sub_train_idx]

print(f"Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

# 클래스 불균형 보정용 pos_weight 계산 (train 기준)
pos_counts = labels_v[train_idx].sum(axis=0)
neg_counts = len(train_idx) - pos_counts
pos_weight = torch.tensor(neg_counts / np.clip(pos_counts, 1, None), dtype=torch.float32)
print("pos_weight range:", pos_weight.min().item(), pos_weight.max().item())


# ### 4.0 물성 기술자 표준화 (Train 기준 fit)
#
# Fingerprint 비트(0/1)와 물성 기술자(MolWt, LogP 등, 스케일이 서로 다름)를 같이 MLP에 넣기 전에,
# 기술자 블록만 train 기준으로 표준화합니다 (Data leakage 방지: mean/std는 train에서만 계산).

N_FP_BITS = MORGAN_BITS + 167  # 지문 비트 수 (표준화 대상 아님)
desc_mean = feat2d_v[train_idx][:, N_FP_BITS:].mean(axis=0)
desc_std = feat2d_v[train_idx][:, N_FP_BITS:].std(axis=0) + 1e-6
feat2d_v[:, N_FP_BITS:] = (feat2d_v[:, N_FP_BITS:] - desc_mean) / desc_std
print("기술자 블록 표준화 완료:", DESCRIPTOR_NAMES)


# ### 4.1 희귀/취약 클래스 Conformer Augmentation
#
# Train positive 수가 적은 클래스(임계값 미만)에 더해, **논문이 명시한 CMR 계열(발암성·변이원성·
# 생식독성)과 그간 실험에서 F1이 낮게 나온 클래스는 건수와 무관하게 강제로 augmentation 대상에
# 포함**시킵니다. 같은 분자를 다른 random seed로 재임베딩해 conformer가 다른 3D 그래프를 추가
# 생성합니다. 2D 피처와 라벨은 동일하지만 3D 좌표/근접엣지가 달라지므로 유효한 증강입니다.
# **Train에만 적용**하며 Val/Test는 절대 건드리지 않습니다 (데이터 누수 방지).

RARE_CLASS_POS_THRESHOLD = 800
N_AUG_PER_SAMPLE = 3  # 희귀/취약 클래스 양성 샘플당 추가 conformer 수

# 건수와 무관하게 강제 포함 (CMR 계열 + 실험적으로 F1이 낮았던 클래스)
FORCE_AUGMENT_CLASSES = [
    "05_Flammable_Solid", "08_Corrosive_to_Metals", "16_Mutagenicity",
    "17_Carcinogenicity", "18_Reproductive_Toxicity", "20_STOT_Repeated_Exposure",
]

rare_by_count = [CLASS_NAMES[c] for c in range(len(CLASS_NAMES)) if pos_counts[c] < RARE_CLASS_POS_THRESHOLD]
augment_target_names = sorted(set(rare_by_count) | set(FORCE_AUGMENT_CLASSES))
rare_classes = [CLASS_NAMES.index(n) for n in augment_target_names]
print("Augmentation 대상 클래스:", augment_target_names)

aug_feat2d, aug_graphs, aug_labels = [], [], []
if rare_classes:
    rare_mask = labels_v[train_idx][:, rare_classes].sum(axis=1) > 0
    rare_local_positions = np.where(rare_mask)[0]
    print(f"증강 대상 train 샘플 수: {len(rare_local_positions)}")

    for local_pos in tqdm(rare_local_positions, desc="Augmenting rare/weak classes"):
        global_row = train_idx[local_pos]
        smi = df_v.iloc[global_row]["SMILES"]
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        for aug_i in range(N_AUG_PER_SAMPLE):
            g = compute_3d_graph_raw(mol, seed=int(1000 + aug_i * 37 + global_row))
            if g is None:
                continue
            aug_feat2d.append(feat2d_v[global_row])
            aug_graphs.append(g)
            aug_labels.append(labels_v[global_row])

print(f"추가된 augmented train 샘플: {len(aug_graphs)}")


# ## 5. Baseline: 2D Tree Ensemble (LightGBM + XGBoost + ExtraTrees) + Optuna
#
# 논문 3.1절 베이스라인(F1 0.6876) 재현. 896차원 2D 피처만 사용,
# 클래스별 독립 이진분류기를 학습한 뒤 Soft Voting으로 앙상블합니다.

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
        max_depth=params["lgb_depth"], verbosity=-1, random_state=SEED))
    xgboost = MultiOutputClassifier(xgb.XGBClassifier(
        n_estimators=params["xgb_n_estimators"], learning_rate=params["xgb_lr"],
        max_depth=params["xgb_depth"], eval_metric="logloss", random_state=SEED))
    extra = MultiOutputClassifier(ExtraTreesClassifier(
        n_estimators=params["et_n_estimators"], max_depth=params["et_depth"],
        random_state=SEED, n_jobs=-1))
    return lgbm, xgboost, extra

def _safe_proba_1(estimator, X):
    """희귀 클래스는 train fold에 양성 샘플이 없을 수 있음(단일 클래스 학습) ->
    이 경우 predict_proba가 (n,1)만 반환하므로 안전하게 처리."""
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
    return f1_score(y_val, preds, average="macro", zero_division=0)

# 참고: 전체 Optuna 탐색은 시간이 오래 걸립니다. n_trials를 줄여서 빠르게 확인 가능.
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=15, show_progress_bar=True)
print("Best baseline val macro-F1:", study.best_value)
print("Best params:", study.best_params)


# 최적 파라미터로 최종 베이스라인 학습 + 테스트셋 평가
best_models = get_models(study.best_params)
for m in best_models:
    m.fit(np.vstack([X_train, X_val]), np.vstack([y_train, y_val]))
test_proba_baseline = soft_vote_proba(best_models, X_test)
test_pred_baseline = (test_proba_baseline > 0.5).astype(int)
baseline_f1 = f1_score(y_test, test_pred_baseline, average="macro", zero_division=0)
print(f"Baseline (2D Tree Ensemble) Test Macro-F1: {baseline_f1:.4f}  (논문 보고치: 0.6876)")


# ## 6. Hybrid GNN-ECC 모델 학습
#
# - 3D Branch: GCNConv x3 + Global Add Pooling → 128-dim
# - 2D Branch: MLP(896→128)
# - Fusion(256-dim) → Differentiable Classifier Chain(22 classes)

from torch_geometric.data import Data, Dataset, Batch
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import GCNConv, global_add_pool

class GHSGraphDataset(Dataset):
    """명시적 리스트 기반 Dataset (augmentation으로 늘어난 train 샘플도 자연스럽게 지원)."""
    def __init__(self, feat2d_list, graphs_list, labels_list):
        super().__init__()
        self.feat2d_list = feat2d_list
        self.graphs_list = graphs_list
        self.labels_list = labels_list

    def len(self):
        return len(self.feat2d_list)

    def get(self, i):
        x, es, ed, ew = self.graphs_list[i]
        data = Data(
            x=torch.tensor(x, dtype=torch.float32),
            edge_index=torch.tensor(np.vstack([es, ed]), dtype=torch.long),
            edge_attr=torch.tensor(ew, dtype=torch.float32).unsqueeze(-1),
        )
        data.feat2d = torch.tensor(self.feat2d_list[i], dtype=torch.float32).unsqueeze(0)
        data.y = torch.tensor(self.labels_list[i], dtype=torch.float32).unsqueeze(0)
        return data

# Train: 원본 + augmented 합침 / Val,Test: 원본 그대로 (누수 방지)
train_feat2d = [feat2d_v[i] for i in train_idx] + aug_feat2d
train_graphs = [graphs_v[i] for i in train_idx] + aug_graphs
train_labels = [labels_v[i] for i in train_idx] + aug_labels
print(f"최종 train 샘플 수 (원본 {len(train_idx)} + 증강 {len(aug_graphs)}): {len(train_graphs)}")

train_ds = GHSGraphDataset(train_feat2d, train_graphs, train_labels)
val_ds = GHSGraphDataset([feat2d_v[i] for i in val_idx], [graphs_v[i] for i in val_idx], [labels_v[i] for i in val_idx])
test_ds = GHSGraphDataset([feat2d_v[i] for i in test_idx], [graphs_v[i] for i in test_idx], [labels_v[i] for i in test_idx])

# augmentation으로 train 샘플이 늘었으므로 pos_weight도 재계산
train_labels_arr = np.stack(train_labels)
pos_counts_aug = train_labels_arr.sum(axis=0)
neg_counts_aug = len(train_labels_arr) - pos_counts_aug
pos_weight = torch.tensor(neg_counts_aug / np.clip(pos_counts_aug, 1, None), dtype=torch.float32)

# 클래스 균형 배치 샘플링: 샘플이 가진 라벨들 중 "가장 희귀한" 라벨의 역빈도를 샘플 가중치로 사용
# -> augmentation으로 못 잡은 미세 불균형도 미니배치 구성 단계에서 추가로 완화
class_freq = np.clip(pos_counts_aug, 1, None) / len(train_labels_arr)
inv_freq = 1.0 / class_freq
sample_weights = np.array([
    inv_freq[row.astype(bool)].max() if row.sum() > 0 else 1.0
    for row in train_labels_arr
])
sampler = torch.utils.data.WeightedRandomSampler(
    weights=torch.tensor(sample_weights, dtype=torch.double),
    num_samples=len(train_labels_arr), replacement=True,
)

train_loader = PyGDataLoader(train_ds, batch_size=64, sampler=sampler)
val_loader = PyGDataLoader(val_ds, batch_size=128, shuffle=False)
test_loader = PyGDataLoader(test_ds, batch_size=128, shuffle=False)
FEAT2D_DIM = feat2d_v.shape[1]
print(f"2D 피처 차원(지문+물성기술자): {FEAT2D_DIM}")


N_CLASSES = 22
FUSION_DIM = 256

class GNNBranch3D(nn.Module):
    def __init__(self, in_dim=9, hidden=64, out_dim=128, dropout=0.2):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.conv3 = GCNConv(hidden, out_dim)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.dropout = nn.Dropout(dropout)  # 오버피팅 억제

    def forward(self, x, edge_index, edge_weight, batch):
        h = F.relu(self.bn1(self.conv1(x, edge_index, edge_weight)))
        h = self.dropout(h)
        h = F.relu(self.bn2(self.conv2(h, edge_index, edge_weight)))
        h = self.dropout(h)
        h = self.conv3(h, edge_index, edge_weight)
        h = global_add_pool(h, batch)
        return h

class MLPBranch2D(nn.Module):
    def __init__(self, in_dim, hidden=256, out_dim=128, dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU())

    def forward(self, x):
        return self.net(x)

class DifferentiableECC(nn.Module):
    def __init__(self, fusion_dim=FUSION_DIM, n_classes=N_CLASSES, head_hidden=64):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(fusion_dim + i, head_hidden), nn.ReLU(), nn.Linear(head_hidden, 1))
            for i in range(n_classes)])

    def forward(self, fusion_repr):
        logits = []
        chain_input = fusion_repr
        for head in self.heads:
            logit = head(chain_input)
            logits.append(logit)
            chain_input = torch.cat([chain_input, logit], dim=1)
        return torch.cat(logits, dim=1)

class FocalLossWithLogits(nn.Module):
    """희귀 클래스 대응: 쉬운(이미 잘 맞히는) 샘플의 loss 기여를 (1-p_t)^gamma 만큼 줄여서
    모델이 어려운/희귀 샘플에 더 집중하도록 함. pos_weight와 병행 사용."""
    def __init__(self, pos_weight=None, gamma=2.0, alpha=0.25):
        super().__init__()
        self.pos_weight = pos_weight
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        focal_term = (1 - p_t).clamp(min=1e-6) ** self.gamma
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * focal_term * bce).mean()

class HybridGNN_ECC(nn.Module):
    def __init__(self, feat2d_dim, atom_feat_dim=9, n_classes=N_CLASSES):
        super().__init__()
        self.branch3d = GNNBranch3D(in_dim=atom_feat_dim)
        self.branch2d = MLPBranch2D(in_dim=feat2d_dim)
        self.chain = DifferentiableECC(fusion_dim=FUSION_DIM, n_classes=n_classes)

    def forward(self, feat2d, graph_x, edge_index, edge_weight, batch_idx):
        h3d = self.branch3d(graph_x, edge_index, edge_weight, batch_idx)
        h2d = self.branch2d(feat2d)
        fusion = torch.cat([h2d, h3d], dim=1)
        return self.chain(fusion)

model = HybridGNN_ECC(feat2d_dim=FEAT2D_DIM).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=3e-4)  # 오버피팅 억제 위해 강화
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)

# USE_FOCAL_LOSS=True 권장: 희귀 클래스가 많은 이 데이터셋에서 일반 BCE보다 유리한 경우가 많음
USE_FOCAL_LOSS = True
if USE_FOCAL_LOSS:
    criterion = FocalLossWithLogits(pos_weight=pos_weight.to(DEVICE), gamma=2.0, alpha=0.25)
else:
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))


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
            all_logits.append(logits.detach().cpu())
            all_y.append(batch.y.cpu())
    all_logits = torch.cat(all_logits); all_y = torch.cat(all_y)
    preds = (torch.sigmoid(all_logits) > 0.5).int().numpy()
    f1 = f1_score(all_y.numpy(), preds, average="macro", zero_division=0)
    return total_loss / len(loader.dataset), f1, all_logits, all_y

EPOCHS = 40
best_val_f1, best_state, patience, bad_epochs = -1, None, 6, 0
MIN_DELTA = 0.003  # 노이즈성 미세 개선은 "진짜 개선"으로 인정하지 않음

for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_f1, _, _ = run_epoch(train_loader, train=True)
    val_loss, val_f1, _, _ = run_epoch(val_loader, train=False)
    scheduler.step(val_f1)
    print(f"Epoch {epoch:02d} | train loss {tr_loss:.4f} f1 {tr_f1:.4f} | val loss {val_loss:.4f} f1 {val_f1:.4f}")
    if val_f1 > best_val_f1 + MIN_DELTA:
        best_val_f1, best_state, bad_epochs = val_f1, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
    else:
        bad_epochs += 1
        if bad_epochs >= patience:
            print("Early stopping."); break

model.load_state_dict(best_state)
print("Best val macro-F1:", best_val_f1)


# ## 7. Optuna 기반 클래스별 결정 임계값 최적화
#
# 기본 0.5 임계값 대신, 클래스별 최적 threshold를 val set에서 탐색합니다
# (불균형이 심한 클래스일수록 효과가 큽니다).

_, _, val_logits, val_y = run_epoch(val_loader, train=False)
val_proba = torch.sigmoid(val_logits).numpy()
val_y_np = val_y.numpy()

def optimize_thresholds(proba, y_true, n_trials=50):
    thresholds = np.full(proba.shape[1], 0.5)
    for c in range(proba.shape[1]):
        def obj(trial):
            t = trial.suggest_float("t", 0.05, 0.95)
            pred = (proba[:, c] > t).astype(int)
            return f1_score(y_true[:, c], pred, zero_division=0)
        study = optuna.create_study(direction="maximize")
        study.optimize(obj, n_trials=n_trials, show_progress_bar=False)
        thresholds[c] = study.best_params["t"]
    return thresholds

best_thresholds = optimize_thresholds(val_proba, val_y_np)
print(dict(zip(CLASS_NAMES, best_thresholds.round(3))))


# Test set에 대한 GNN 예측 (뒤의 stacking/평가 섹션에서 재사용)
_, _, test_logits, test_y = run_epoch(test_loader, train=False)
test_proba = torch.sigmoid(test_logits).numpy()
test_y_np = test_y.numpy()


# ## 8. Stacking 앙상블 (GNN-ECC + 2D Tree Baseline)

# GNN과 Tree baseline은 서로 다른 신호(3D 위상 구조 vs 국소 서브구조 패턴)를 보고 있어
# 단순 평균보다 클래스별 최적 가중치(alpha)를 val set에서 탐색해 블렌딩하면 보통 더 좋습니다.
# blend = alpha * GNN_proba + (1-alpha) * Baseline_proba, 클래스별 threshold도 동시 최적화.

# baseline 모델의 val/test 확률 (5절에서 학습한 best_models 재사용)
val_proba_baseline = soft_vote_proba(best_models, X_val)
test_proba_baseline = soft_vote_proba(best_models, X_test)  # 이미 5절에서 계산됨 (재계산해도 무방)

def optimize_blend_and_threshold(gnn_proba, baseline_proba, y_true, n_trials=40):
    n_classes = y_true.shape[1]
    alphas = np.zeros(n_classes)
    thresholds = np.zeros(n_classes)
    for c in range(n_classes):
        def obj(trial):
            a = trial.suggest_float("alpha", 0.0, 1.0)
            t = trial.suggest_float("t", 0.05, 0.95)
            blend = a * gnn_proba[:, c] + (1 - a) * baseline_proba[:, c]
            pred = (blend > t).astype(int)
            return f1_score(y_true[:, c], pred, zero_division=0)
        study = optuna.create_study(direction="maximize")
        study.optimize(obj, n_trials=n_trials, show_progress_bar=False)
        alphas[c] = study.best_params["alpha"]
        thresholds[c] = study.best_params["t"]
    return alphas, thresholds

best_alphas, blend_thresholds = optimize_blend_and_threshold(val_proba, val_proba_baseline, val_y_np)
print("클래스별 GNN 가중치(alpha, 1=GNN만 / 0=baseline만):")
print(dict(zip(CLASS_NAMES, best_alphas.round(2))))


# ## 9. 최종 평가 (Test Set)

test_blend_proba = best_alphas * test_proba + (1 - best_alphas) * test_proba_baseline
pred_stacked = (test_blend_proba > blend_thresholds).astype(int)

pred_default = (test_proba > 0.5).astype(int)
pred_gnn_optimized = (test_proba > best_thresholds).astype(int)

f1_default = f1_score(test_y_np, pred_default, average="macro", zero_division=0)
f1_gnn_optimized = f1_score(test_y_np, pred_gnn_optimized, average="macro", zero_division=0)
f1_stacked = f1_score(test_y_np, pred_stacked, average="macro", zero_division=0)

print(f"Baseline (2D Tree Ensemble)               Macro-F1: {baseline_f1:.4f}")
print(f"Hybrid GNN-ECC  | default 0.5 threshold   Macro-F1: {f1_default:.4f}")
print(f"Hybrid GNN-ECC  | Optuna 최적 threshold    Macro-F1: {f1_gnn_optimized:.4f}")
print(f"Stacked (GNN + Baseline blend)             Macro-F1: {f1_stacked:.4f}")

from sklearn.metrics import classification_report
print(classification_report(test_y_np, pred_stacked, target_names=CLASS_NAMES, zero_division=0))


# 클래스별 Recall 비교 (3D 도입 효과가 큰 CMR 계열: Mutagenicity/Carcinogenicity/Reproductive_Toxicity 확인)
from sklearn.metrics import recall_score
recalls = recall_score(test_y_np, pred_stacked, average=None, zero_division=0)
recall_df = pd.DataFrame({"class": CLASS_NAMES, "recall": recalls, "train_pos_count": pos_counts.astype(int) if not USE_FOCAL_LOSS else pos_counts_aug.astype(int)}).sort_values("recall")
recall_df


# ### 9.1 MCC (Matthews Correlation Coefficient)
#
# F1은 불균형 데이터에서도 유용하지만, MCC는 TP/TN/FP/FN을 모두 고려해 "무작위 예측이면 0,
# 완벽하면 1, 반대로만 맞히면 -1"이 되는 대칭적 지표로, 극심한 불균형 이진분류 성능 비교에
# F1/accuracy보다 더 신뢰할 수 있다고 알려져 있다(Chicco & Jurman, 2020). 동일 GHS 예측
# 과제를 다룬 선행연구(Harada et al., 2025, ACS Chem. Health Saf.)가 hazard class 예측에서
# 평균 MCC 0.582±0.208을 보고했으므로, 직접 비교 가능한 지표로 우리도 함께 산출한다.

from sklearn.metrics import matthews_corrcoef

def macro_mcc(y_true, y_pred):
    """클래스별 MCC를 계산 후 단순평균 (한 클래스라도 양성/음성 예측이 전혀 없으면 0으로 처리)"""
    mccs = []
    for c in range(y_true.shape[1]):
        try:
            mcc = matthews_corrcoef(y_true[:, c], y_pred[:, c])
        except Exception:
            mcc = 0.0
        mccs.append(mcc)
    return np.array(mccs)

mcc_baseline = macro_mcc(test_y_np, (test_proba_baseline > 0.5).astype(int))
mcc_gnn = macro_mcc(test_y_np, pred_gnn_optimized)
mcc_stacked = macro_mcc(test_y_np, pred_stacked)

print(f"Baseline (2D Tree Ensemble)   Macro-MCC: {mcc_baseline.mean():.4f} ± {mcc_baseline.std():.4f}")
print(f"Hybrid GNN-ECC                Macro-MCC: {mcc_gnn.mean():.4f} ± {mcc_gnn.std():.4f}")
print(f"Stacked (GNN + Baseline)      Macro-MCC: {mcc_stacked.mean():.4f} ± {mcc_stacked.std():.4f}")
print()
print("참고: Harada et al. (2025)의 26-class 독립 이진분류 hazard class 모델 평균 MCC = 0.582 ± 0.208")
print("      (단, 과제 설정이 달라 직접 비교는 주의 필요 — Discussion 참고)")

mcc_df = pd.DataFrame({"class": CLASS_NAMES, "MCC": mcc_stacked}).sort_values("MCC")
mcc_df


# ### 9.2 ROC-AUC 및 Balanced Accuracy (CCR)
#
# F1/MCC와 함께, 선행연구(Harada et al. 2025; Fuadah et al. 2024, ACS Omega)가 핵심 지표로
# 사용하는 **ROC-AUC**(임계값에 의존하지 않는 지표)와 **Balanced Accuracy**(= CCR,
# sensitivity와 specificity의 평균)를 추가로 계산한다. 클래스별 전체 지표(precision/recall/
# specificity/NPV 등)는 Supplementary Table로 별도 제공하고, 본문에는 macro-평균만 싣는다.

from sklearn.metrics import roc_auc_score, balanced_accuracy_score

def macro_auc(y_true, y_proba):
    """클래스별 ROC-AUC를 계산 후 단순평균 (한 클래스라도 양성/음성 중 하나만 있으면 그 클래스는 스킵)"""
    aucs = []
    for c in range(y_true.shape[1]):
        if len(np.unique(y_true[:, c])) < 2:
            continue  # 해당 클래스에 양성 또는 음성만 있으면 AUC 정의 불가
        try:
            auc = roc_auc_score(y_true[:, c], y_proba[:, c])
        except Exception:
            continue
        aucs.append(auc)
    return np.array(aucs)

def macro_balanced_accuracy(y_true, y_pred):
    """클래스별 Balanced Accuracy(=CCR) 계산 후 단순평균"""
    bas = []
    for c in range(y_true.shape[1]):
        try:
            ba = balanced_accuracy_score(y_true[:, c], y_pred[:, c])
        except Exception:
            ba = 0.0
        bas.append(ba)
    return np.array(bas)

auc_baseline = macro_auc(test_y_np, test_proba_baseline)
auc_gnn = macro_auc(test_y_np, test_proba)
auc_stacked = macro_auc(test_y_np, test_blend_proba)

ba_baseline = macro_balanced_accuracy(test_y_np, (test_proba_baseline > 0.5).astype(int))
ba_gnn = macro_balanced_accuracy(test_y_np, pred_gnn_optimized)
ba_stacked = macro_balanced_accuracy(test_y_np, pred_stacked)

print(f"Baseline (2D Tree Ensemble)   Macro-AUC: {auc_baseline.mean():.4f} ± {auc_baseline.std():.4f}  |  Balanced Acc(CCR): {ba_baseline.mean():.4f} ± {ba_baseline.std():.4f}")
print(f"Hybrid GNN-ECC                Macro-AUC: {auc_gnn.mean():.4f} ± {auc_gnn.std():.4f}  |  Balanced Acc(CCR): {ba_gnn.mean():.4f} ± {ba_gnn.std():.4f}")
print(f"Stacked (GNN + Baseline)      Macro-AUC: {auc_stacked.mean():.4f} ± {auc_stacked.std():.4f}  |  Balanced Acc(CCR): {ba_stacked.mean():.4f} ± {ba_stacked.std():.4f}")
print()
print("참고: Harada et al. (2025) hazard class 모델 평균 AUC는 별도 보고 없음(accuracy/F1/MCC만 보고)")
print("참고: Fuadah et al. (2024, ACS Omega) consensus 모델 평균 AUC = 0.78~0.90 (8개 endpoint, 개별 이진분류)")


# ## 10. 모델 저장

torch.save({
    "model_state_dict": model.state_dict(),
    "gnn_thresholds": best_thresholds,
    "stack_alphas": best_alphas,
    "stack_thresholds": blend_thresholds,
    "class_names": CLASS_NAMES,
}, "hybrid_gnn_ecc_ghs22.pt")
print("저장 완료: hybrid_gnn_ecc_ghs22.pt")


# ## 11. SCI 투고를 위한 방법론 검증
#
# 앞선 결과는 단일 train/val/test 분할 기준입니다. SCI 저널 심사에서는 다음이 사실상 필수로
# 요구됩니다:
#
# 1. **5-Fold CV** — 단일 split이 우연히 좋거나 나쁜 결과가 아님을 mean±std로 입증
# 2. **Scaffold Split** — 구조가 유사한 분자가 train/test에 동시에 들어가는 데이터 누수를
#    배제하고, 실제 신규 화합물에 대한 일반화 성능을 보수적으로 재평가
# 3. **Bootstrap 유의성 검정** — Baseline vs GNN vs Stacked 차이가 통계적으로 유의한지
# 4. **SHAP 해석** — 어떤 물성/구조 신호가 예측에 기여했는지 (블랙박스 비판 방어)
#
# 계산 비용상 baseline(2D Tree Ensemble)은 5-fold 전체를 돌리고, GNN은 예시 fold 구조만
# 제공합니다 (전체 5-fold GNN 재학습은 시간이 오래 걸리므로 시간 여유가 있을 때 실행하세요).

# ### 11.1 Baseline 5-Fold Cross-Validation

cv_f1_scores = []
mskf_cv = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

for fold, (tr_idx, te_idx) in enumerate(mskf_cv.split(feat2d_v, labels_v)):
    fold_models = get_models(study.best_params)  # 5절에서 찾은 최적 하이퍼파라미터 재사용
    for m in fold_models:
        m.fit(feat2d_v[tr_idx], labels_v[tr_idx])
    proba = soft_vote_proba(fold_models, feat2d_v[te_idx])
    pred = (proba > 0.5).astype(int)
    f1 = f1_score(labels_v[te_idx], pred, average="macro", zero_division=0)
    cv_f1_scores.append(f1)
    print(f"Fold {fold+1}/5: Macro F1 = {f1:.4f}")

cv_f1_scores = np.array(cv_f1_scores)
print(f"\n5-Fold CV Baseline Macro F1: {cv_f1_scores.mean():.4f} ± {cv_f1_scores.std():.4f}")
print("(논문 Table에 'mean ± std' 형태로 기재)")


# ### 11.2 Scaffold Split (구조 기반 분리 — 데이터 누수 없는 일반화 성능)
#
# Murcko Scaffold(고리+연결부 골격)가 같은 분자는 유사 구조로 간주하여 train/test에 함께
# 들어가지 않도록 분리합니다. Random split보다 엄격하고 보수적인 성능 추정치가 나오는 게
# 일반적이며, QSAR/cheminformatics 논문 심사에서 거의 필수로 요구됩니다.

from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import GroupShuffleSplit

def get_murcko_scaffold(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        return scaffold if scaffold else smiles  # 골격이 없는 소분자는 자기 자신을 그룹으로
    except Exception:
        return smiles

scaffolds = df_v["SMILES"].apply(get_murcko_scaffold).values
n_unique_scaffolds = len(set(scaffolds))
print(f"전체 {len(df_v)}개 화합물 / 고유 scaffold {n_unique_scaffolds}개")

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
scaffold_train_idx, scaffold_test_idx = next(gss.split(feat2d_v, labels_v, groups=scaffolds))
print(f"Scaffold split: Train {len(scaffold_train_idx)} / Test {len(scaffold_test_idx)}")

# scaffold 그룹 단위로 분리하므로 rare class는 test에 0건이 될 수도 있음 (실제 발생 가능한 현상,
# 논문에서는 이를 "scaffold split이 rare class에서 특히 보수적"이라는 발견으로 서술 가능)
scaffold_models = get_models(study.best_params)
for m in scaffold_models:
    m.fit(feat2d_v[scaffold_train_idx], labels_v[scaffold_train_idx])
scaffold_proba = soft_vote_proba(scaffold_models, feat2d_v[scaffold_test_idx])
scaffold_pred = (scaffold_proba > 0.5).astype(int)
scaffold_f1 = f1_score(labels_v[scaffold_test_idx], scaffold_pred, average="macro", zero_division=0)

print(f"\nRandom Split Baseline Test Macro F1  : {baseline_f1:.4f}")
print(f"Scaffold Split Baseline Test Macro F1: {scaffold_f1:.4f}")
print("(둘의 차이가 크면 random split 결과에 구조적 데이터 누수가 있었다는 뜻이므로,")
print(" 논문에는 반드시 두 수치를 함께 보고할 것)")


# ### 11.3 (선택) GNN Scaffold Split 검증
#
# 시간이 오래 걸리므로 여유가 있을 때 실행하세요. Section 6~9의 학습 루프를
# `scaffold_train_idx`/`scaffold_test_idx` 기준으로 그대로 재사용하면 됩니다.

RUN_GNN_SCAFFOLD_CHECK = False  # True로 바꾸면 실행 (시간 소요 큼)

if RUN_GNN_SCAFFOLD_CHECK:
    # train_idx/val_idx를 scaffold_train_idx 내에서 다시 8:2로 나눈 뒤,
    # 6~9절의 GHSGraphDataset / HybridGNN_ECC / run_epoch 로직을 그대로 재사용해 재학습합니다.
    # (지면상 생략 — 6~9절 코드를 scaffold_train_idx/scaffold_test_idx로 바꿔 다시 실행하면 됩니다)
    print("6~9절 코드를 scaffold_train_idx/scaffold_test_idx 기준으로 재실행하세요.")
else:
    print("스킵됨. RUN_GNN_SCAFFOLD_CHECK=True로 바꿔서 실행 가능.")


# ### 11.4 Bootstrap 유의성 검정
#
# Baseline vs GNN vs Stacked의 test set 성능 차이가 통계적으로 유의한지 1,000회 bootstrap
# resampling으로 95% 신뢰구간을 계산합니다. 신뢰구간이 0을 포함하지 않으면 유의한 차이로
# 판단합니다 (permutation test와 유사한 비모수적 접근).

def bootstrap_f1_diff(y_true, proba_a, thr_a, proba_b, thr_b, n_boot=1000, seed=SEED):
    """proba_b 모델이 proba_a 모델보다 얼마나 나은지에 대한 bootstrap 분포"""
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
    p_like = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())  # 양측 근사 p-value
    return diffs.mean(), ci_low, ci_high, p_like

pairs = [
    ("Baseline -> GNN", test_proba_baseline, np.full(N_CLASSES, 0.5), test_proba, best_thresholds),
    ("Baseline -> Stacked", test_proba_baseline, np.full(N_CLASSES, 0.5), test_blend_proba, blend_thresholds),
    ("GNN -> Stacked", test_proba, best_thresholds, test_blend_proba, blend_thresholds),
]

print(f"{'비교':<22}{'평균 F1 차이':>14}{'95% CI':>22}{'근사 p-value':>14}")
for name, pa, ta, pb, tb in pairs:
    mean_diff, lo, hi, p = bootstrap_f1_diff(test_y_np, pa, ta, pb, tb, n_boot=1000)
    sig = "유의함" if (lo > 0 or hi < 0) else "유의하지 않음"
    print(f"{name:<22}{mean_diff:>+14.4f}   [{lo:+.4f}, {hi:+.4f}]{p:>14.4f}  ({sig})")


# ### 11.5 SHAP 해석 (Baseline Tree 모델)
#
# GNN의 atom-level saliency는 계산 비용이 크므로, 상대적으로 가볍고 해석이 직관적인
# LightGBM baseline에 대해 SHAP을 적용합니다. 대표적으로 (1) 샘플이 많은 흔한 클래스,
# (2) CMR 계열의 어려운 클래스, (3) 물성 기술자가 중요할 것으로 예상되는 물리적 유해성
# 클래스를 하나씩 뽑아 비교합니다. Fingerprint 비트는 개별 해석이 어려우므로 물성 기술자
# 블록 위주로 해석하고, fingerprint는 전체 기여도 총합만 비교합니다.

import shap

SHAP_SAMPLE_CLASSES = ["14_Eye_Damage_Irritation", "16_Mutagenicity", "08_Corrosive_to_Metals"]
X_test_sample = X_test[:300]  # 계산 비용 절감을 위해 일부만 사용
feature_names = [f"Morgan_{i}" for i in range(MORGAN_BITS)] + [f"MACCS_{i}" for i in range(167)] + DESCRIPTOR_NAMES

for cls_name in SHAP_SAMPLE_CLASSES:
    cls_idx = CLASS_NAMES.index(cls_name)
    lgb_estimator = best_models[0].estimators_[cls_idx]  # best_models[0] = LightGBM MultiOutputClassifier
    if 1 not in list(lgb_estimator.classes_):
        print(f"{cls_name}: train fold에 양성 샘플이 없어 SHAP 스킵")
        continue
    explainer = shap.TreeExplainer(lgb_estimator)
    shap_values = explainer.shap_values(X_test_sample)
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values  # 양성 클래스 기준

    mean_abs_shap = np.abs(sv).mean(axis=0)
    fp_importance = mean_abs_shap[:N_FP_BITS].sum()
    desc_importance = mean_abs_shap[N_FP_BITS:]
    top_desc_idx = np.argsort(desc_importance)[::-1][:5]

    print(f"\n=== {cls_name} ===")
    print(f"  Fingerprint 총 기여도: {fp_importance:.4f} vs 물성기술자 총 기여도: {desc_importance.sum():.4f}")
    print("  Top-5 물성 기술자:")
    for i in top_desc_idx:
        print(f"    {DESCRIPTOR_NAMES[i]}: {desc_importance[i]:.4f}")


# ### 11.6 결과 요약표 (논문 Table용)

summary = pd.DataFrame({
    "Method": ["Baseline (single split)", "Baseline (5-fold CV)", "Baseline (scaffold split)",
               "GNN-ECC (single split)", "Stacked (single split)"],
    "Macro F1": [baseline_f1, cv_f1_scores.mean(), scaffold_f1, f1_gnn_optimized, f1_stacked],
    "Macro MCC": [mcc_baseline.mean(), np.nan, np.nan, mcc_gnn.mean(), mcc_stacked.mean()],
    "Macro AUC": [auc_baseline.mean(), np.nan, np.nan, auc_gnn.mean(), auc_stacked.mean()],
    "Balanced Acc (CCR)": [ba_baseline.mean(), np.nan, np.nan, ba_gnn.mean(), ba_stacked.mean()],
    "Std / Note": ["-", f"± {cv_f1_scores.std():.4f}", "구조 기반 분리", "-", "-"],
})
summary

