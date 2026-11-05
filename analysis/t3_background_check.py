"""How do the T3 metrics treat a generic 'highly expressed genes decrease' background, alone and under a
regulon signal? Checked on every KO we have (Gata4 x2 at E8.75, Mab21l2 at E9.5).

  python analysis/t3_background_check.py   -> private/t3_background_check.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from vec import t3

net = pd.read_csv(t3.ROOT / "data/priors/collectri_mouse.csv.gz")
S = {k: t3.load(k) for k in ["WT_E8.75", "WT_E9.5", "Gata4_1", "Gata4_2", "Mab21l2"]}
t3.transfer_states(S["WT_E8.75"], [S["Gata4_1"], S["Gata4_2"]])
t3.transfer_states(S["WT_E9.5"], [S["Mab21l2"]])
genes = list(S["WT_E8.75"].genes)


def regulon(tf):
    sub = net[(net.source_genesymbol == tf) & net.target_genesymbol.isin(genes) & (net.target_genesymbol != tf)]
    w = np.zeros(len(genes))
    for _, r in sub.iterrows():
        w[genes.index(r.target_genesymbol)] = 1.0 if r.is_stimulation else (-1.0 if r.is_inhibition else 0.5)
    return w


out = {}
for ko, (target, wtn) in t3.KO.items():
    wt = S[wtn]
    states = sorted(set(wt.state))
    mu = np.nan_to_num(t3.state_stats(wt, states)["mu"])
    w = regulon(target)
    j = genes.index(target)
    a = mu[:, j] / (mu[:, j].max() + 1e-9)
    sig = -(a[:, None] * w[None, :]) * mu          # regulon targets lose expression where target is active
    sc = t3.Scorer(S[ko].X, wt.X)
    import common.core_metrics as cm
    up, dn, _ = cm.de_genes(S[ko].X, wt.X)
    res = {"truth_n_up": len(up), "truth_n_dn": len(dn), "regulon_size": int((w != 0).sum()),
           "wt_identity": sc(t3.emit(wt, None, states)[0])}
    for eps in (0.02, 0.05, 0.1):
        res[f"shrink {eps}"] = sc(t3.emit(wt, -eps * mu, states)[0])
        if (w != 0).any():
            for lam in (0.2, 0.5):
                res[f"shrink {eps} + regulon x{lam}"] = sc(t3.emit(wt, -eps * mu + lam * sig, states)[0])
    out[ko] = res
    print(ko, {k: res[k] for k in ["truth_n_up", "truth_n_dn", "regulon_size"]})
    for k, v in res.items():
        if isinstance(v, dict):
            print(f"  {k:32s} total {v['approx_total']:5.1f}  de {v['de_score']:+.3f}  dir {v['de_direction']:+.3f}"
                  f"  slope {v['severity_slope']:+.2f} (r2 {v['slope_r2']})  mmd {v['mmd_u']:.4f}", flush=True)
Path("private/t3_background_check.json").write_text(json.dumps(out, indent=1))
