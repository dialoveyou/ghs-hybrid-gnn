#!/usr/bin/env python3
"""
GHS 22-class Hybrid 2D-3D GNN-ECC pipeline - resumable/checkpointed cloud run.
Adapted from ghs_full_pipeline.py (same modeling logic), with:
  - chunked, resumable feature-extraction checkpointing (survives interruption)
  - flushed timestamped logging to stdout
  - JSON metrics dump after every major stage, so partial results are recoverable
Run: python3 run_full_pipeline.py
"""
import os, sys, time, json, random, pickle, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

WORKDIR = os.environ.get("GHS_WORKDIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "work"))
OUTDIR = os.path.join(WORKDIR, "outputs")
os.makedirs(OUTDIR, exist_ok=True)
CSV_PATH = os.path.join(WORKDIR, "_wip_modeling_subset_14148.csv")
CACHE_PATH = os.path.join(OUTDIR, "ghs_features_cache.pkl")
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

# ---------- 2. 데이터 로드 및 22-Class 타겟 재정의 ----------
df = pd.read_csv(CSV_PATH)
log(f"Loaded CSV: {df.shape}")

GHS_22_CLASS_MAP = {
    "01_Explosive_SelfReactive": ["H200","H201","H202","H203","H204","H205","H240","H241","H242"],
    "02_Flammable_Gas": ["H220","H221","H230","H231"],
    "03_Gas_Under_Pressure": ["H280","H281"],
    "04_Flammable_Liquid": ["H224","H225","H226","H227"],
    "05_Flammable_Solid": ["H228"],
    "06_Spontaneous_Combustion": ["H250","H251","H252","H260","H261"],
    "07_Oxidizing": ["H270","H271","H272"],
    "08_Corrosive_to_Metals": ["H290"],
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
log(f"Chemicals with >=1 GHS label: {len(df)} (paper reports 14,148)")
save_metrics({"n_labeled_chemicals": int(len(df))})

# ---------- 3. 2D/3D 피처 엔지니어링 (청크 단위 resumable 체크포인트) ----------
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, MACCSkeys, Descriptors, Crippen, rdMolDescriptors
from scipy.spatial import cKDTree
RDLogger.DisableLog("rdApp.*")

MORGAN_BITS = 729
MORGAN_RADIUS = 2
PROXIMITY_CUTOFF = 5.0
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

feat2d_dict, graph_dict, done_idx = {}, {}, set()
if os.path.exists(CACHE_PATH):
    log("기존 캐시에서 재시작...")
    with open(CACHE_PATH, "rb") as f:
        feat2d_dict, graph_dict = pickle.load(f)
    done_idx = set(feat2d_dict.keys()) | set(graph_dict.keys())
    log(f"캐시에서 복구: 2D {len(feat2d_dict)}개, 3D {len(graph_dict)}개 완료됨")

all_args = [(i, s) for i, s in enumerate(df["SMILES"].tolist()) if i not in done_idx]
log(f"처리할 남은 분자 수: {len(all_args)} / 전체 {len(df)}")

CHUNK_SIZE = 500
N_WORKERS = min(2, os.cpu_count() or 1)
t0 = time.time()
processed = 0
for chunk_start in range(0, len(all_args), CHUNK_SIZE):
    chunk = all_args[chunk_start:chunk_start+CHUNK_SIZE]
    with Pool(processes=N_WORKERS) as pool:
        for idx, fp, g in pool.imap_unordered(process_one, chunk, chunksize=8):
            if fp is not None:
                feat2d_dict[idx] = fp
            if g is not None:
                graph_dict[idx] = g
            processed += 1
    with open(CACHE_PATH, "wb") as f:
        pickle.dump((feat2d_dict, graph_dict), f)
    elapsed = time.time() - t0
    rate = processed / elapsed if elapsed > 0 else 0
    remaining = len(all_args) - processed
    eta_min = (remaining / rate / 60) if rate > 0 else float("nan")
    log(f"청크 완료: {processed}/{len(all_args)} 신규 처리 | 2D누적 {len(feat2d_dict)} 3D누적 {len(graph_dict)} | {rate:.2f}mol/s | ETA {eta_min:.1f}min")

valid_idx = sorted(set(feat2d_dict.keys()) & set(graph_dict.keys()))
log(f"2D 성공: {len(feat2d_dict)} | 3D 성공: {len(graph_dict)} | 둘 다 성공(학습 사용): {len(valid_idx)}")
save_metrics({"n_2d_success": len(feat2d_dict), "n_3d_success": len(graph_dict), "n_valid_both": len(valid_idx)})

df_v = df.iloc[valid_idx].reset_index(drop=True)
labels_v = labels[valid_idx]
feat2d_v = np.stack([feat2d_dict[i] for i in valid_idx])
graphs_v = [graph_dict[i] for i in valid_idx]

with open(os.path.join(OUTDIR, "prepped_data.pkl"), "wb") as f:
    pickle.dump({"df_v": df_v, "labels_v": labels_v, "feat2d_v": feat2d_v, "graphs_v": graphs_v,
                 "CLASS_NAMES": CLASS_NAMES}, f)
log("피처 엔지니어링 완료, prepped_data.pkl 저장. STAGE1_DONE")
