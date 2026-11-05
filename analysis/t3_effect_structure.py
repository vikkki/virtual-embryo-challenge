"""Is the within-state KO shift state-specific (biology) or a gene-wise offset shared by all states (batch)?

  python analysis/t3_effect_structure.py   -> private/t3_effect_structure.json
"""
import json
from pathlib import Path

import numpy as np

from vec import t3

S = {k: t3.load(k) for k in ["WT_E8.75", "WT_E9.5", "Gata4_1", "Gata4_2", "Mab21l2"]}
wt, wt95 = S["WT_E8.75"], S["WT_E9.5"]
t3.transfer_states(wt, [S["Gata4_1"], S["Gata4_2"]])
t3.transfer_states(wt95, [S["Mab21l2"]])
genes = list(wt.genes)
D = {k: t3.decompose(S[k], wt) for k in ["Gata4_1", "Gata4_2"]}
D["Mab21l2"] = t3.decompose(S["Mab21l2"], wt95, )
cor = lambda a, b: float(np.corrcoef(a, b)[0, 1])
out = {}
offsets = {}
for k, d in D.items():
    ok = d["ok"] & (d["n_wt"] >= 100) & (d["n_ko"] >= 100)
    dl = d["delta"][ok]
    w = d["f_wt"][ok] / d["f_wt"][ok].sum()
    off = np.median(dl, 0)                       # gene-wise offset shared across states
    offsets[k] = off
    resid = dl - off
    var_tot = float((w[:, None] * dl ** 2).sum()); var_res = float((w[:, None] * resid ** 2).sum())
    tgt = d_tgt = {"Gata4_1": "Gata4", "Gata4_2": "Gata4", "Mab21l2": "Mab21l2"}[k]
    ref = wt if k.startswith("Gata4") else wt95
    st = [s for s, o in zip(d["states"], ok) if o]
    texpr = np.array([ref.X[ref.state == s, genes.index(tgt)].mean() for s in st])
    norms = np.linalg.norm(resid, axis=1)
    per_state_r = [cor(dl[i], off) for i in range(len(st))]
    out[k] = {
        "n_states_used": int(ok.sum()),
        "frac_energy_in_shared_offset": 1 - var_res / var_tot,
        "median_r(state delta, shared offset)": float(np.median(per_state_r)),
        "r(target expr in state, |state-specific residual|)": cor(texpr, norms),
        "target expr by state (top5)": sorted(zip(st, texpr.round(2).tolist()), key=lambda x: -x[1])[:5],
        "largest state-specific residuals": sorted(zip(st, norms.round(2).tolist()), key=lambda x: -x[1])[:6],
        "offset top up": [(genes[i], round(float(off[i]), 2)) for i in np.argsort(-off)[:10]],
        "offset top dn": [(genes[i], round(float(off[i]), 2)) for i in np.argsort(off)[:10]],
        "n genes |offset|>=0.25": int((np.abs(off) >= .25).sum()),
    }
out["r(offset Gata4_1, Gata4_2)"] = cor(offsets["Gata4_1"], offsets["Gata4_2"])
out["r(offset Gata4 mean, Mab21l2)"] = cor((offsets["Gata4_1"] + offsets["Gata4_2"]) / 2, offsets["Mab21l2"])
out["r(within Gata4 mean, within Mab21l2)"] = cor((D["Gata4_1"]["within"] + D["Gata4_2"]["within"]) / 2, D["Mab21l2"]["within"])
Path("private/t3_effect_structure.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
