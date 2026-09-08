"""3D-isolation ablation.

Three architectures share the differentiable classifier chain and an identical training
protocol, split and sample set; the only difference is which representation branch feeds
the chain. No conformer augmentation, so the 2D-only model is not penalised by duplicated
rows. Each model is then stacked with the tree baseline under a matched threshold protocol.
"""
import pickle, json, time, copy, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score

SEED = 42; N_CLASSES = 22; N_FP = 896
EPOCHS, PATIENCE, MIN_DELTA, BATCH = 40, 6, 0.003, 64
torch.manual_seed(SEED); np.random.seed(SEED)
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

class GNNBranch3D(nn.Module):
    def __init__(self, in_dim=9, hidden=64, out_dim=128, dropout=0.2):
        super().__init__()
        self.conv1, self.conv2, self.conv3 = GCNConv(in_dim,hidden), GCNConv(hidden,hidden), GCNConv(hidden,out_dim)
        self.bn1, self.bn2 = nn.BatchNorm1d(hidden), nn.BatchNorm1d(hidden)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, ei, ew, b):
        h = self.dropout(F.relu(self.bn1(self.conv1(x, ei, ew))))
        h = self.dropout(F.relu(self.bn2(self.conv2(h, ei, ew))))
        return global_add_pool(self.conv3(h, ei, ew), b)

class MLPBranch2D(nn.Module):
    def __init__(self, in_dim, hidden=256, out_dim=128, dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim,hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(hidden,out_dim),
                                 nn.BatchNorm1d(out_dim), nn.ReLU())
    def forward(self, x): return self.net(x)

class ECC(nn.Module):
    def __init__(self, fusion_dim, n_classes=N_CLASSES, head_hidden=64):
        super().__init__()
        self.heads = nn.ModuleList([nn.Sequential(nn.Linear(fusion_dim+i, head_hidden), nn.ReLU(),
                                                  nn.Linear(head_hidden,1)) for i in range(n_classes)])
    def forward(self, f):
        outs, ci = [], f
        for h in self.heads:
            o = h(ci); outs.append(o); ci = torch.cat([ci, o], dim=1)
        return torch.cat(outs, dim=1)

class Net(nn.Module):
    """mode: 'both' | 'd2' | 'd3'"""
    def __init__(self, mode, feat2d_dim, atom_dim=9):
        super().__init__()
        self.mode = mode
        self.branch2d = MLPBranch2D(feat2d_dim) if mode in ("both","d2") else None
        self.branch3d = GNNBranch3D(atom_dim)   if mode in ("both","d3") else None
        self.chain = ECC(fusion_dim=256 if mode=="both" else 128)
    def forward(self, f2, gx, ei, ew, b):
        if self.mode == "both":
            z = torch.cat([self.branch2d(f2), self.branch3d(gx, ei, ew, b)], dim=1)
        elif self.mode == "d2":
            z = self.branch2d(f2)
        else:
            z = self.branch3d(gx, ei, ew, b)
        return self.chain(z)

class Focal(nn.Module):
    def __init__(self, pos_weight, gamma=2.0, alpha=0.25):
        super().__init__(); self.pw, self.g, self.a = pos_weight, gamma, alpha
    def forward(self, logits, t):
        bce = F.binary_cross_entropy_with_logits(logits, t, pos_weight=self.pw, reduction="none")
        p = torch.sigmoid(logits); pt = p*t + (1-p)*(1-t)
        return (( self.a*t + (1-self.a)*(1-t) ) * (1-pt).clamp(min=1e-6)**self.g * bce).mean()

def macro_f1(y, p, t): return f1_score(y, (p > t).astype(int), average="macro", zero_division=0)

def thr_grid(proba, y, n=181):
    grid = np.linspace(0.05, 0.95, n); out = np.full(proba.shape[1], 0.5)
    for c in range(proba.shape[1]):
        pred = proba[:, c][None, :] > grid[:, None]
        tp = (pred & (y[:, c][None, :] > 0.5)).sum(1); fp = (pred & (y[:, c][None, :] < 0.5)).sum(1)
        fn = ((~pred) & (y[:, c][None, :] > 0.5)).sum(1)
        f1 = np.divide(2*tp, 2*tp+fp+fn, out=np.zeros(len(grid)), where=(2*tp+fp+fn) > 0)
        out[c] = grid[int(f1.argmax())]
    return out

def blend_grid(g_val, b_val, y_val, n_a=21, n_t=91):
    """per-class alpha/threshold on validation, matching the paper's stacking objective"""
    A = np.linspace(0, 1, n_a); T = np.linspace(0.05, 0.95, n_t)
    al = np.zeros(y_val.shape[1]); th = np.zeros(y_val.shape[1])
    for c in range(y_val.shape[1]):
        blend = A[:, None]*g_val[:, c][None, :] + (1-A)[:, None]*b_val[:, c][None, :]   # (A, n)
        pred = blend[:, None, :] > T[None, :, None]                                     # (A, T, n)
        pos = y_val[:, c][None, None, :] > 0.5
        tp = (pred & pos).sum(-1); fp = (pred & ~pos).sum(-1); fn = ((~pred) & pos).sum(-1)
        f1 = np.divide(2*tp, 2*tp+fp+fn, out=np.zeros_like(tp, dtype=float), where=(2*tp+fp+fn) > 0)
        i, j = np.unravel_index(int(f1.argmax()), f1.shape)
        al[c], th[c] = A[i], T[j]
    return al, th

def make_ds(idxs, feat2d, graphs, labels):
    ds = []
    for i in idxs:
        x, s, d_, w = graphs[i]
        ds.append(Data(x=torch.tensor(x),
                       edge_index=torch.tensor(np.vstack([s, d_]), dtype=torch.long),
                       edge_weight=torch.tensor(w),
                       feat2d=torch.tensor(feat2d[i]).unsqueeze(0),
                       y=torch.tensor(labels[i]).unsqueeze(0)))
    return ds

def run(loader, model, crit, opt=None):
    model.train() if opt else model.eval()
    tot, L, Y = 0.0, [], []
    with torch.set_grad_enabled(opt is not None):
        for b in loader:
            lo = model(b.feat2d, b.x, b.edge_index, b.edge_weight, b.batch)
            loss = crit(lo, b.y)
            if opt: opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()*b.num_graphs; L.append(lo.detach()); Y.append(b.y)
    L, Y = torch.cat(L), torch.cat(Y)
    return tot/len(loader.dataset), f1_score(Y.numpy(), (torch.sigmoid(L)>0.5).int().numpy(),
                                             average="macro", zero_division=0), torch.sigmoid(L).numpy(), Y.numpy()

if __name__ == "__main__":
    feats = pickle.load(open("/tmp/exp/features.pkl","rb"))
    st = pickle.load(open("/tmp/exp/p0_store.pkl","rb"))
    feat2d, graphs, labels = feats["feat2d_v"].copy(), feats["graphs_v"], feats["labels_v"]
    tr, va, te = st["train_idx"], st["val_idx"], st["test_idx"]
    mu = feat2d[tr][:, N_FP:].mean(0); sd = feat2d[tr][:, N_FP:].std(0) + 1e-6
    feat2d[:, N_FP:] = (feat2d[:, N_FP:] - mu) / sd
    yva, yte = st["yva"], st["yte"]

    tr_ds, va_ds, te_ds = make_ds(tr, feat2d, graphs, labels), make_ds(va, feat2d, graphs, labels), make_ds(te, feat2d, graphs, labels)
    ytr = labels[tr]
    pos = ytr.sum(0); pw = torch.tensor((len(tr)-pos)/np.clip(pos,1,None), dtype=torch.float32)
    inv = 1.0/(np.clip(pos,1,None)/len(tr))
    sw = np.array([inv[r.astype(bool)].max() if r.sum()>0 else 1.0 for r in ytr])
    va_loader = DataLoader(va_ds, batch_size=128, shuffle=False)
    te_loader = DataLoader(te_ds, batch_size=128, shuffle=False)

    results, probs = {}, {}
    for mode, name in (("both","2D+3D GNN-ECC"), ("d2","2D-only MLP-ECC"), ("d3","3D-only GNN-ECC")):
        torch.manual_seed(SEED); np.random.seed(SEED)
        g = torch.Generator(); g.manual_seed(SEED)
        sampler = torch.utils.data.WeightedRandomSampler(torch.tensor(sw, dtype=torch.double),
                                                         num_samples=len(sw), replacement=True, generator=g)
        tr_loader = DataLoader(tr_ds, batch_size=BATCH, sampler=sampler)
        model = Net(mode, feat2d.shape[1]); crit = Focal(pw)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=3e-4)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", patience=3, factor=0.5)
        best, best_state, bad = -1, None, 0
        log(f"--- training {name} ---")
        for ep in range(1, EPOCHS+1):
            t0 = time.time()
            _, trf, _, _ = run(tr_loader, model, crit, opt)
            _, vaf, _, _ = run(va_loader, model, crit)
            sch.step(vaf)
            log(f"  ep{ep:02d} train f1 {trf:.4f} | val f1 {vaf:.4f} | {time.time()-t0:.0f}s")
            if vaf > best + MIN_DELTA:
                best, best_state, bad = vaf, copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
                if bad >= PATIENCE: log("  early stop"); break
        model.load_state_dict(best_state)
        _, _, pv, _ = run(va_loader, model, crit)
        _, _, pt, _ = run(te_loader, model, crit)
        probs[mode] = (pv, pt)
        t = thr_grid(pv, yva)
        results[name] = {"alone_thr0.5": macro_f1(yte, pt, 0.5), "alone_thrTuned": macro_f1(yte, pt, t)}
        log(f"  {name}: alone@0.5 {results[name]['alone_thr0.5']:.4f} | alone@tuned {results[name]['alone_thrTuned']:.4f}")

    # ---- stack each neural model with the tree baseline, matched protocol, no leakage ----
    bv, bt_ = st["baseline_trainonly_val"], st["baseline_trainonly_test"]
    base_thr = thr_grid(bv, yva)
    results["2D tree ensemble (baseline)"] = {"alone_thr0.5": macro_f1(yte, bt_, 0.5),
                                              "alone_thrTuned": macro_f1(yte, bt_, base_thr)}
    for mode, name in (("both","2D+3D GNN-ECC"), ("d2","2D-only MLP-ECC"), ("d3","3D-only GNN-ECC")):
        pv, pt = probs[mode]
        a, t = blend_grid(pv, bv, yva)
        sp = a*pt + (1-a)*bt_
        results[name]["stacked_with_tree"] = macro_f1(yte, sp, t)
        probs[mode] = (pv, pt, sp, t)
        log(f"  stacked tree + {name}: {results[name]['stacked_with_tree']:.4f}")

    pickle.dump({"probs":probs, "base_thr":base_thr, "bv":bv, "bt":bt_}, open("/tmp/exp/abl_store.pkl","wb"))
    json.dump(results, open("/tmp/exp/ablation_results.json","w"), indent=2, default=float)
    log("RESULTS " + json.dumps(results, default=float))
