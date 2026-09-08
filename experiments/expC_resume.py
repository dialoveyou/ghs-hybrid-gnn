"""C, resumed: the three random chain orderings (original/reversed already measured)."""
import json, time, numpy as np, sys
sys.path.insert(0, "/tmp/exp")
import expC_chainorder as C

KNOWN = {"original": {"both": 0.6369, "d2": 0.6338, "both_alone": 0.5777, "d2_alone": 0.5860},
         "reversed": {"both": 0.6368, "d2": 0.6335, "both_alone": 0.5936, "d2_alone": 0.6005}}
for k in KNOWN: KNOWN[k]["increment_3d"] = KNOWN[k]["both"] - KNOWN[k]["d2"]

rng = np.random.default_rng(7)
perms = [rng.permutation(22) for _ in range(3)]
res = dict(KNOWN); incs = [KNOWN["original"]["increment_3d"], KNOWN["reversed"]["increment_3d"]]
for k, perm in enumerate(perms, 1):
    name = f"random{k}"; row = {}
    for mode in ("both", "d2"):
        t0 = time.time()
        pv, pt = C.train_eval(mode, perm)
        al, tt = C.blend_grid(pv, C.bv, C.yva)
        row[mode] = C.macro_f1(C.yte, al * pt + (1 - al) * C.bt, tt)
        row[mode + "_alone"] = C.macro_f1(C.yte, pt, C.thr_grid(pv, C.yva))
        C.log(f"  {name} {mode:5s} alone {row[mode+'_alone']:.4f} | stacked {row[mode]:.4f} ({time.time()-t0:.0f}s)")
    row["increment_3d"] = row["both"] - row["d2"]; incs.append(row["increment_3d"])
    res[name] = row
    C.log(f"  {name} --> 3D increment {row['increment_3d']:+.4f}")
res["increment_mean"] = float(np.mean(incs)); res["increment_sd"] = float(np.std(incs, ddof=1))
res["increment_range"] = [float(min(incs)), float(max(incs))]
json.dump(res, open("/tmp/exp/expC_results.json", "w"), indent=2, default=float)
C.log(f"3D increment across 5 chain orderings: {np.mean(incs):+.4f} ± {np.std(incs, ddof=1):.4f} "
      f"(range {min(incs):+.4f} to {max(incs):+.4f})")
