"""Do KOs that share one control share a KO-vs-control signature? (Grosswendt: 10 KOs vs one Cdkn2a control;
the T3 analogue: Gata4 and beta-catenin vs the same WT E8.75 reference.)

1. Correlation of within-state KO shifts between every pair of examples (genes measured in both).
2. "Copy the other KOs of the same control": predict each Grosswendt KO as the mean per-state delta of the
   other 9, and score it against the floor. This is the Grosswendt analogue of predicting beta-catenin
   from Gata4.

  python analysis/t3_shared_control.py   -> runs/t3_shared_control.json  (remote box; data-derived)
"""
import json
from pathlib import Path

import numpy as np

from vec import t3

ex = {}
S = {k: t3.load(k) for k in ["WT_E8.75", "WT_E9.5", "Gata4_1", "Gata4_2", "Mab21l2"]}
t3.transfer_states(S["WT_E8.75"], [S["Gata4_1"], S["Gata4_2"]])
t3.transfer_states(S["WT_E9.5"], [S["Mab21l2"]])
for k, (tgt, w) in t3.KO.items():
    ex[k] = {"wt": S[w], "ko": S[k], "src": "challenge"}
wt_cache = {}
for name, m in t3.external_index().items():
    if m["wt"] not in wt_cache:
        wt_cache[m["wt"]] = t3.load(m["wt"])
    ex[name] = {"wt": wt_cache[m["wt"]], "ko": t3.load(m["ko"]), "src": m["source"]}
for e in ex.values():
    e["d"] = t3.decompose(e["ko"], e["wt"], min_cells=20)
    e["mask"] = np.ones(500, bool) if e["wt"].present is None else e["wt"].present

names = list(ex)
R = np.full((len(names), len(names)), np.nan)
for i, a in enumerate(names):
    for j, b in enumerate(names):
        m = ex[a]["mask"] & ex[b]["mask"]
        R[i, j] = np.corrcoef(ex[a]["d"]["within"][m], ex[b]["d"]["within"][m])[0, 1]
gw = [i for i, n in enumerate(names) if n.startswith("grosswendt")]
other = [i for i in range(len(names)) if i not in gw]
off = lambda idx: [R[i, j] for i in idx for j in idx if i < j]
cross = [R[i, j] for i in gw for j in other]
out = {"names": names, "corr_within": np.round(R, 3).tolist(),
       "mean_r_within_grosswendt": float(np.mean(off(gw))),
       "mean_r_grosswendt_vs_others": float(np.mean(cross)),
       "mean_r_among_non_grosswendt": float(np.nanmean([R[i, j] for i in other for j in other
                                                        if i < j and not (names[i].startswith('Gata4') and names[j].startswith('Gata4'))
                                                        and not (names[i].startswith('hand2') and names[j].startswith('hand2'))]))}
print({k: round(v, 3) for k, v in out.items() if k.startswith("mean")})

# copy-the-other-KOs baseline inside Grosswendt (all share the Cdkn2a control and its states)
states = ex[names[gw[0]]]["d"]["states"]
res = {}
for i in gw:
    n = names[i]
    others = [names[j] for j in gw if j != i]
    pred = np.mean([ex[o]["d"]["delta"] for o in others], 0)
    wt = ex[n]["wt"]
    sc = t3.Scorer(ex[n]["ko"].X, wt.X)
    r = {"floor": sc(t3.emit(wt, None, states)[0])["approx_total"]}
    for s in (0.5, 1.0):
        r[f"copy_others x{s}"] = sc(t3.emit(wt, s * pred, states)[0])["approx_total"]
    res[n] = r
    print(n, r, flush=True)
out["copy_others_grosswendt"] = res
gain = [v["copy_others x1.0"] - v["floor"] for v in res.values()]
out["copy_others_mean_gain"] = float(np.mean(gain))
print("copy-others mean gain", round(float(np.mean(gain)), 1), "beats floor", sum(g > 0 for g in gain), "/", len(gain))
Path("runs").mkdir(exist_ok=True)
Path("runs/t3_shared_control.json").write_text(json.dumps(out, indent=1))
