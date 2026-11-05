"""T3 local benchmark on the Gata4 replicates: decomposition, emission checks, simple baselines.

  python analysis/t3_baselines.py            -> private/t3_baselines.json (data-derived; git-ignored)

Every prediction is emitted from WT E8.75 cells (7,449 = board max) with their 3D coordinates and scored with
veckit's metrics against each Gata4 replicate (full KO file as truth, full WT E8.75 as reference).
"""
import json
import time
from pathlib import Path

import numpy as np

from vec import t3

t0 = time.time()
S = {k: t3.load(k) for k in ["WT_E8.5", "WT_E8.75", "WT_E9.5", "Gata4_1", "Gata4_2", "Mab21l2"]}
wt, wt95 = S["WT_E8.75"], S["WT_E9.5"]
t3.transfer_states(wt, [S["Gata4_1"], S["Gata4_2"], S["WT_E8.5"]])
t3.transfer_states(wt95, [S["Mab21l2"]])
print(f"loaded + mapped in {time.time()-t0:.0f}s")

D = {k: t3.decompose(S[k], wt) for k in ["Gata4_1", "Gata4_2"]}
D["Mab21l2"] = t3.decompose(S["Mab21l2"], wt95)
states = D["Gata4_1"]["states"]
out = {"decomposition": {}, "scores": {}}
cor = lambda a, b: float(np.corrcoef(a, b)[0, 1])
for k, d in D.items():
    out["decomposition"][k] = {
        "L2_total": float(np.linalg.norm(d["total"])), "L2_within": float(np.linalg.norm(d["within"])),
        "L2_comp": float(np.linalg.norm(d["comp"])),
        "r_total_within": cor(d["total"], d["within"]), "r_total_comp": cor(d["total"], d["comp"]),
        "n_DE(|total|>=.25)": int((np.abs(d["total"]) >= .25).sum()),
        "n_DE(|within|>=.25)": int((np.abs(d["within"]) >= .25).sum()),
        "top_comp_up": [(d["states"][i], round(float(d["comp_logratio"][i]), 2)) for i in np.argsort(-d["comp_logratio"])[:5]],
        "top_comp_dn": [(d["states"][i], round(float(d["comp_logratio"][i]), 2)) for i in np.argsort(d["comp_logratio"])[:5]],
    }
g1, g2 = D["Gata4_1"], D["Gata4_2"]
out["replicate_agreement"] = {
    "r_within": cor(g1["within"], g2["within"]), "r_comp": cor(g1["comp"], g2["comp"]),
    "r_total": cor(g1["total"], g2["total"]),
    "r_comp_logratio": cor(g1["comp_logratio"], g2["comp_logratio"]),
}

# stage axis at E8.75 (same vocabulary at E8.5): per-state E8.5 - E8.75 ("delay" direction)
stage = t3.decompose(S["WT_E8.5"], wt)

# Mab21l2 per-state deltas live in the E9.5 vocabulary; transfer to E8.75 states by nearest WT centroid
st95 = D["Mab21l2"]["states"]
mu875 = t3.state_stats(wt, states)["mu"]; mu95 = t3.state_stats(wt95, st95)["mu"]
z = lambda M: (M - M.mean(1, keepdims=True)) / (M.std(1, keepdims=True) + 1e-9)
match = np.argmax(z(np.nan_to_num(mu875)) @ z(np.nan_to_num(mu95)).T, 1)
mab_delta = D["Mab21l2"]["delta"][match]
mab_global = np.tile(D["Mab21l2"]["within"], (len(states), 1))

cands = {
    "wt_identity": (None, None),
    "oracle_self_full": "self_full", "oracle_self_within": "self_within", "oracle_self_comp": "self_comp",
    "cross_rep_full": "cross_full", "cross_rep_within": "cross_within",
    "delay_0.5 (E8.5 dir)": (0.5 * stage["delta"], None),
    "delay_1.0 (E8.5 dir)": (stage["delta"], None),
    "mab21l2_state_transfer": (mab_delta, None),
    "mab21l2_global": (mab_global, None),
}
for tgt, other in [("Gata4_1", "Gata4_2"), ("Gata4_2", "Gata4_1")]:
    sc = t3.Scorer(S[tgt].X, wt.X)
    me, ot = D[tgt], D[other]
    res = {}
    for name, c in cands.items():
        if c == "self_full": c = (me["delta"], me["comp_logratio"])
        elif c == "self_within": c = (me["delta"], None)
        elif c == "self_comp": c = (None, me["comp_logratio"])
        elif c == "cross_full": c = (ot["delta"], ot["comp_logratio"])
        elif c == "cross_within": c = (ot["delta"], None)
        X, C, L = t3.emit(wt, c[0], states, c[1], seed=0)
        res[name] = sc(X)
        print(tgt, name, res[name], flush=True)
    out["scores"][tgt] = res
Path("private").mkdir(exist_ok=True)
Path("private/t3_baselines.json").write_text(json.dumps(out, indent=1, default=str))
print(json.dumps({k: out[k] for k in ["decomposition", "replicate_agreement"]}, indent=1, default=str))
print(f"done in {time.time()-t0:.0f}s")
