import pickle, json, numpy as np
from sklearn.metrics import f1_score
st = pickle.load(open("p0_store.pkl","rb")); ab = pickle.load(open("abl_store.pkl","rb"))
yte = st["yte"]; probs = ab["probs"]; bt = ab["bt"]; base_thr = ab["base_thr"]
half = np.full(22, 0.5)

def boot(y, pa, ta, pb, tb, n=5000, seed=42):
    rng = np.random.default_rng(seed); N=y.shape[0]; d=np.empty(n)
    for i in range(n):
        k = rng.integers(0,N,N)
        d[i] = (f1_score(y[k],(pb[k]>tb).astype(int),average="macro",zero_division=0)
               -f1_score(y[k],(pa[k]>ta).astype(int),average="macro",zero_division=0))
    lo,hi = np.percentile(d,[2.5,97.5]); p = 2*min((d<=0).mean(),(d>=0).mean())
    return d.mean(),lo,hi,p

both = (probs["both"][2], probs["both"][3])
d2   = (probs["d2"][2],   probs["d2"][3])
d3   = (probs["d3"][2],   probs["d3"][3])
tests = [
 ("tree(tuned) -> tree+2D+3D",      (bt,base_thr), both),
 ("tree(tuned) -> tree+2D-only",    (bt,base_thr), d2),
 ("tree+2D-only -> tree+2D+3D  [the 3D increment]", d2, both),
 ("tree(tuned) -> tree+3D-only",    (bt,base_thr), d3),
]
out={}
for name,(pa,ta),(pb,tb) in tests:
    m,lo,hi,p = boot(yte,pa,ta,pb,tb)
    sig = "significant" if (lo>0 or hi<0) else "not significant"
    out[name]=dict(delta=m,ci=[lo,hi],p=p,sig=sig)
    print(f"{name:48s} Δ={m:+.4f}  CI [{lo:+.4f}, {hi:+.4f}]  p={p:.4f}  {sig}", flush=True)
json.dump(out, open("abl_bootstrap.json","w"), indent=2, default=float)
