"""Does a prior-only prediction (regulatory network j -> g, no KO data) beat wt_identity on the KOs we have?

This is the cellworld "no sources" setting, i.e. exactly what beta-catenin forces. The network is CollecTRI
(mouse, via OmniPath). The target's activity per state is either its own transcript level (only possible when
it is on the panel) or the mean z-score of its regulon in that WT state (the only option for Ctnnb1).

  python analysis/t3_prior_check.py   -> private/t3_prior_check.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from vec import t3

net = pd.read_csv(t3.ROOT / "data/priors/collectri_mouse.csv.gz")
S = {k: t3.load(k) for k in ["WT_E8.75", "Gata4_1", "Gata4_2"]}
wt = S["WT_E8.75"]
t3.transfer_states(wt, [S["Gata4_1"], S["Gata4_2"]])
genes = list(wt.genes)
states = sorted(set(wt.state))
st = t3.state_stats(wt, states)
mu = np.nan_to_num(st["mu"])


def regulon(tf):
    sub = net[(net.source_genesymbol == tf) & net.target_genesymbol.isin(genes) & (net.target_genesymbol != tf)]
    w = np.zeros(len(genes))
    for _, r in sub.iterrows():
        w[genes.index(r.target_genesymbol)] = 1.0 if r.is_stimulation else (-1.0 if r.is_inhibition else 0.5)
    return w


def activity_from_regulon(w):
    z = (mu - mu.mean(0)) / (mu.std(0) + 1e-6)
    a = (z * w).sum(1) / max(np.abs(w).sum(), 1)
    return np.clip((a - a.min()) / (a.max() - a.min() + 1e-9), 0, 1)


def predict(w, a, mode):
    """KO of an activator j: its stimulated targets lose expression where j is active."""
    if mode == "proportional":   # lose a fraction of current expression
        return -(a[:, None] * w[None, :]) * mu
    if mode == "global":         # same everywhere, ignoring state activity
        return -np.tile(w, (len(states), 1)) * mu
    raise ValueError(mode)


out = {}
w = regulon("Gata4")
a_tx = mu[:, genes.index("Gata4")] / mu[:, genes.index("Gata4")].max()
a_reg = activity_from_regulon(w)
out["r(activity from transcript, from regulon)"] = float(np.corrcoef(a_tx, a_reg)[0, 1])
out["n_regulon_on_panel"] = int((w != 0).sum())
for rep in ["Gata4_1", "Gata4_2"]:
    sc = t3.Scorer(S[rep].X, wt.X)
    res = {"wt_identity": sc(t3.emit(wt, None, states)[0])}
    for act_name, a in [("transcript", a_tx), ("regulon", a_reg)]:
        for mode in ["proportional", "global"]:
            for lam in (0.1, 0.3, 0.6):
                d = lam * predict(w, a, mode)
                res[f"{mode}/{act_name} x{lam}"] = sc(t3.emit(wt, d, states)[0])
    # random regulon of the same size (null): is it the network or just "some genes go down"?
    rng = np.random.default_rng(0)
    nulls = []
    for k in range(20):
        wr = np.zeros_like(w); wr[rng.choice(len(w), int((w != 0).sum()), replace=False)] = 1
        nulls.append(sc(t3.emit(wt, 0.3 * predict(wr, a_tx, "proportional"), states)[0])["approx_total"])
    res["random_regulon_null (proportional/transcript x0.3)"] = {"mean": float(np.mean(nulls)), "sd": float(np.std(nulls)),
                                                                 "max": float(np.max(nulls))}
    out[rep] = res
    print(rep)
    for k, v in res.items():
        if "null" in k:
            print(f"  {k:50s} {v}"); continue
        print(f"  {k:50s} total {v['approx_total']:5.1f}  de {v['de_score']:+.3f}  dir {v['de_direction']:+.3f}"
              f"  slope {v['severity_slope']:+.2f}  mmd {v['mmd_u']:.4f}", flush=True)
print({k: out[k] for k in list(out)[:2]})
Path("private/t3_prior_check.json").write_text(json.dumps(out, indent=1))
