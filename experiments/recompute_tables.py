"""Recompute every per-class table for the new best model (tree + 2D-3D hybrid, calibrated)."""
import pickle, json, numpy as np
from sklearn.metrics import f1_score, roc_auc_score, matthews_corrcoef, balanced_accuracy_score

st = pickle.load(open("/tmp/exp/p0_store.pkl","rb"))
ab = pickle.load(open("/tmp/exp/abl_store.pkl","rb"))
yte = st["yte"]; probs = ab["probs"]; bt = ab["bt"]; base_thr = ab["base_thr"]
CLS = ["Explosive SelfReactive","Flammable Gas","Gas Under Pressure","Flammable Liquid",
       "Flammable Solid","Spontaneous Combustion","Oxidizing","Corrosive to Metals",
       "Acute Toxicity Oral","Acute Toxicity Dermal","Acute Toxicity Inhalation",
       "Aspiration Hazard","Skin Corrosion Irritation","Eye Damage Irritation","Sensitization",
       "Mutagenicity","Carcinogenicity","Reproductive Toxicity","STOT Single Exposure Merged",
       "STOT Repeated Exposure","Aquatic Hazard","Ozone Layer Hazard"]

MODELS = {
    "tree":        (bt, base_thr),
    "d2_alone":    (probs["d2"][1],   None),
    "d3_alone":    (probs["d3"][1],   None),
    "both_alone":  (probs["both"][1], None),
    "stack_d3":    (probs["d3"][2],   probs["d3"][3]),
    "stack_d2":    (probs["d2"][2],   probs["d2"][3]),
    "stack_both":  (probs["both"][2], probs["both"][3]),
}
# standalone neural models need their tuned thresholds recomputed on val
from ablation import thr_grid
yva = st["yva"]
for k, mode in (("d2_alone","d2"), ("d3_alone","d3"), ("both_alone","both")):
    MODELS[k] = (probs[mode][1], thr_grid(probs[mode][0], yva))

def summary(p, t):
    pred = (p > t).astype(int)
    mcc = np.array([matthews_corrcoef(yte[:,c], pred[:,c]) if len(set(yte[:,c]))>1 else 0.0 for c in range(22)])
    auc = np.array([roc_auc_score(yte[:,c], p[:,c]) for c in range(22) if len(set(yte[:,c]))>1])
    ba  = np.array([balanced_accuracy_score(yte[:,c], pred[:,c]) for c in range(22)])
    return dict(macro_f1=f1_score(yte, pred, average="macro", zero_division=0),
                mcc_mean=mcc.mean(), mcc_std=mcc.std(),
                auc_mean=auc.mean(), auc_std=auc.std(),
                ba_mean=ba.mean(), ba_std=ba.std())

out = {"table2": {}}
for k, (p, t) in MODELS.items():
    out["table2"][k] = summary(p, t)
    out["table2"][k]["macro_f1_at_0.5"] = f1_score(yte, (p > 0.5).astype(int), average="macro", zero_division=0)
    print(f"{k:12s} F1@0.5 {out['table2'][k]['macro_f1_at_0.5']:.4f} | F1 {out['table2'][k]['macro_f1']:.4f} "
          f"| MCC {out['table2'][k]['mcc_mean']:.3f}±{out['table2'][k]['mcc_std']:.3f} "
          f"| AUC {out['table2'][k]['auc_mean']:.3f} | BA {out['table2'][k]['ba_mean']:.3f}", flush=True)

# ---- per-class table for the new best model ----
P, T = MODELS["stack_both"]
pred = (P > T).astype(int)
rows = []
for c in range(22):
    y, q = yte[:, c], pred[:, c]
    tp = int(((y==1)&(q==1)).sum()); fp = int(((y==0)&(q==1)).sum())
    fn = int(((y==1)&(q==0)).sum()); tn = int(((y==0)&(q==0)).sum())
    prec = tp/(tp+fp) if tp+fp else 0.0
    rec  = tp/(tp+fn) if tp+fn else 0.0
    spec = tn/(tn+fp) if tn+fp else 0.0
    npv  = tn/(tn+fn) if tn+fn else 0.0
    f1   = 2*prec*rec/(prec+rec) if prec+rec else 0.0
    rows.append(dict(cls=CLS[c], precision=prec, recall=rec, specificity=spec, npv=npv,
                     f1=f1, support=int(y.sum())))
out["table_perclass"] = rows
print("\n=== per-class, new best model (tree + 2D-3D hybrid, calibrated) ===")
for r in rows:
    print(f"  {r['cls']:28s} P {r['precision']:.2f} R {r['recall']:.2f} Sp {r['specificity']:.3f} "
          f"NPV {r['npv']:.3f} F1 {r['f1']:.2f}  n={r['support']}")

# ---- bootstrap CI for low-support classes ----
rng = np.random.default_rng(42)
cis = {}
for r in rows:
    if r["support"] >= 50: continue
    c = CLS.index(r["cls"]); y, q = yte[:, c], pred[:, c]
    n = len(y); vals = np.empty(5000)
    for i in range(5000):
        k = rng.integers(0, n, n)
        vals[i] = f1_score(y[k], q[k], zero_division=0)
    cis[r["cls"]] = [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
out["lowsupport_ci"] = cis
print("\n=== bootstrap 95% CI, support < 50 ===")
for k, v in cis.items(): print(f"  {k:28s} [{v[0]:.2f}, {v[1]:.2f}]")

# ---- Fuadah-overlapping endpoints ----
IDX = {"09 Acute Toxicity Oral":8, "10 Acute Toxicity Dermal":9, "11 Acute Toxicity Inhalation":10,
       "13 Skin Corrosion/Irritation":12, "14 Eye Damage/Irritation":13, "15 Sensitization":14}
h2h = {}
for name, c in IDX.items():
    h2h[name] = dict(auc=roc_auc_score(yte[:,c], P[:,c]),
                     f1=f1_score(yte[:,c], pred[:,c], zero_division=0),
                     balacc=balanced_accuracy_score(yte[:,c], pred[:,c]))
h2h["Mean"] = {k: float(np.mean([v[k] for n,v in h2h.items() if n!="Mean"])) for k in ("auc","f1","balacc")}
out["fuadah_h2h"] = h2h
print("\n=== Fuadah-overlapping endpoints, new model ===")
for k, v in h2h.items():
    print(f"  {k:30s} AUC {v['auc']:.3f}  F1 {v['f1']:.3f}  BalAcc {v['balacc']:.3f}")

json.dump(out, open("/tmp/exp/new_tables.json","w"), indent=2, default=float)
print("\nsaved new_tables.json")
