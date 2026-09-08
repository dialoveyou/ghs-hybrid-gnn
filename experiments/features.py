"""Stage 1 replica: 22-class labels + 916-d 2D features + 3D graphs, cached."""
import os, sys, time, pickle
import numpy as np, pandas as pd
from multiprocessing import Pool
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, MACCSkeys, Descriptors, Crippen, rdMolDescriptors
from scipy.spatial import cKDTree
RDLogger.DisableLog("rdApp.*")

SEED = 42
MORGAN_BITS, MORGAN_RADIUS, PROXIMITY_CUTOFF = 729, 2, 5.0
CSV = "/tmp/src_raw/_wip_modeling_subset_14148.csv"
OUT = "/tmp/exp/features.pkl"

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
CLASS_NAMES = list(GHS_22_CLASS_MAP.keys())
HCODE_TO_CLASS = {c: k for k, v in GHS_22_CLASS_MAP.items() for c in v}
HYB = {Chem.HybridizationType.SP:0, Chem.HybridizationType.SP2:1, Chem.HybridizationType.SP3:2,
       Chem.HybridizationType.SP3D:3, Chem.HybridizationType.SP3D2:4}
N_DESC = 20

def build_label_vector(s):
    vec = np.zeros(len(CLASS_NAMES), dtype=np.float32)
    if pd.isna(s): return vec
    for c in s.split(","):
        c = c.strip()
        if c in HCODE_TO_CLASS: vec[CLASS_NAMES.index(HCODE_TO_CLASS[c])] = 1.0
    return vec

def _desc(mol):
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

def fp2d(mol):
    morgan = AllChem.GetMorganFingerprintAsBitVect(mol, MORGAN_RADIUS, nBits=MORGAN_BITS)
    maccs = MACCSkeys.GenMACCSKeys(mol)
    a = np.zeros((MORGAN_BITS,), dtype=np.float32); Chem.DataStructs.ConvertToNumpyArray(morgan, a)
    b = np.zeros((167,), dtype=np.float32); Chem.DataStructs.ConvertToNumpyArray(maccs, b)
    return np.concatenate([a, b, _desc(mol)])

def atom_feats(atom):
    return [atom.GetAtomicNum(), atom.GetFormalCharge(), int(atom.GetIsAromatic()),
            HYB.get(atom.GetHybridization(), 5), atom.GetTotalNumHs(), int(atom.IsInRing()),
            atom.GetDegree(), atom.GetExplicitValence(), atom.GetMass()/100.0]

def graph3d(mol, seed=SEED, cutoff=PROXIMITY_CUTOFF):
    molH = Chem.AddHs(Chem.Mol(mol))
    p = AllChem.ETKDGv3(); p.randomSeed = seed; p.useRandomCoords = True
    if AllChem.EmbedMolecule(molH, p) < 0: return None
    try: AllChem.MMFFOptimizeMolecule(molH, maxIters=200)
    except Exception: pass
    m = Chem.RemoveHs(molH); n = m.GetNumAtoms()
    if n == 0: return None
    conf = m.GetConformer()
    pos = np.array([list(conf.GetAtomPosition(i)) for i in range(n)], dtype=np.float32)
    x = np.array([atom_feats(a) for a in m.GetAtoms()], dtype=np.float32)
    bonded, src, dst, w = set(), [], [], []
    for b in m.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        ww = 1.0/max(float(np.linalg.norm(pos[i]-pos[j])), 1e-3)
        src += [i, j]; dst += [j, i]; w += [ww, ww]; bonded.add((min(i,j), max(i,j)))
    if n > 1:
        for i, j in cKDTree(pos).query_pairs(r=cutoff):
            if (min(i,j), max(i,j)) in bonded: continue
            ww = 1.0/max(float(np.linalg.norm(pos[i]-pos[j])), 1e-3)
            src += [i, j]; dst += [j, i]; w += [ww, ww]
    if not src: src, dst, w = [0], [0], [1.0]
    return x, np.array(src), np.array(dst), np.array(w, dtype=np.float32)

def process_one(args):
    idx, smi = args
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return idx, None, None
    try: Chem.SanitizeMol(mol)
    except Exception: return idx, None, None
    return idx, fp2d(mol), graph3d(mol)

if __name__ == "__main__":
    df = pd.read_csv(CSV)
    labels = np.stack(df["Y_GHS_H_Codes"].apply(build_label_vector).values)
    keep = labels.sum(axis=1) > 0
    df = df[keep].reset_index(drop=True); labels = labels[keep]
    print(f"chemicals with >=1 GHS label: {len(df)}", flush=True)

    f2d, g3d = {}, {}
    t0 = time.time()
    with Pool(processes=2) as pool:
        for k, (idx, fp, g) in enumerate(pool.imap_unordered(process_one, list(enumerate(df["SMILES"].tolist())), chunksize=16)):
            if fp is not None: f2d[idx] = fp
            if g is not None: g3d[idx] = g
            if (k+1) % 2000 == 0:
                print(f"  {k+1}/{len(df)}  {time.time()-t0:.0f}s", flush=True)
    valid = sorted(set(f2d) & set(g3d))
    print(f"2D ok {len(f2d)} | 3D ok {len(g3d)} | both {len(valid)}", flush=True)

    with open(OUT, "wb") as fh:
        pickle.dump({"labels_v": labels[valid],
                     "feat2d_v": np.stack([f2d[i] for i in valid]),
                     "graphs_v": [g3d[i] for i in valid],
                     "class_names": CLASS_NAMES}, fh)
    print("saved", OUT, f"({time.time()-t0:.0f}s total)", flush=True)
