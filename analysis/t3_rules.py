"""Target-agnostic rules for a KO's within-state shift, scored zero-shot on every KO sample we have.

A rule sees only the stage-matched WT and the name of the knocked-out gene, never the KO cells. A rule that
beats wt_identity on BOTH Gata4 and Mab21l2 is a candidate for the beta-catenin prior.

  python analysis/t3_rules.py   -> private/t3_rules.json
"""
import json
from pathlib import Path

import numpy as np

from vec import t3

S = {k: t3.load(k) for k in ["WT_E8.75", "WT_E9.5", "Gata4_1", "Gata4_2", "Mab21l2"]}
t3.transfer_states(S["WT_E8.75"], [S["Gata4_1"], S["Gata4_2"]])
t3.transfer_states(S["WT_E9.5"], [S["Mab21l2"]])
genes = list(S["WT_E8.75"].genes)


def rules(wt, target):
    states = sorted(set(wt.state))
    st = t3.state_stats(wt, states)
    mu, f = np.nan_to_num(st["mu"]), st["f"]
    j = genes.index(target)
    t = mu[:, j] / (mu[:, j].max() + 1e-9)                      # target level per state, 0..1
    gbar = (f[:, None] * mu).sum(0)                              # WT pseudobulk
    ident = mu - gbar                                            # state identity (marker) profile
    Xc = wt.X - wt.X.mean(0)
    c = (Xc * Xc[:, [j]]).mean(0) / (Xc.std(0) * Xc[:, j].std() + 1e-9)   # co-expression with target
    c[j] = 0
    lose_target = np.zeros_like(mu); lose_target[:, j] = -mu[:, j]       # target transcript lost
    R = {
        "lose_identity (t_s * (gbar - mu_s))": -t[:, None] * ident,
        "coexpr_down (t_s * -corr)": -t[:, None] * c[None, :],
        "coexpr_down_global (-corr)": -np.tile(c, (len(states), 1)),
        "lose_target_only": lose_target,
        "lose_identity + lose_target": -t[:, None] * ident + lose_target,
    }
    return states, R


out = {}
for tgt_sample, (target, wt_name) in t3.KO.items():
    wt = S[wt_name]
    sc = t3.Scorer(S[tgt_sample].X, wt.X)
    states, R = rules(wt, target)
    res = {"wt_identity": sc(t3.emit(wt, None, states)[0])}
    for name, d in R.items():
        for lam in (0.25, 0.5, 1.0):
            res[f"{name} x{lam}"] = sc(t3.emit(wt, lam * d, states)[0])
    out[tgt_sample] = res
    print(tgt_sample)
    for k, v in res.items():
        print(f"  {k:45s} total {v['approx_total']:5.1f}  de {v['de_score']:+.3f}  dir {v['de_direction']:+.3f}"
              f"  slope {v['severity_slope']:+.2f}  mmd {v['mmd_u']:.4f}", flush=True)
Path("private/t3_rules.json").write_text(json.dumps(out, indent=1))
