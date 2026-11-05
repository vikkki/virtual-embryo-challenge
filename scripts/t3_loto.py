#!/usr/bin/env python
"""Leave-one-target-out check of the prior-only spike-and-slab model (vec.t3model) against wt_identity.

  python scripts/t3_loto.py [--epochs 300] [--no-network]     -> runs/t3_loto.json

Folds: train on Mab21l2 -> predict Gata4 (scored vs each replicate); train on Gata4 (both reps) -> predict
Mab21l2. A sanity fold trains on Gata4 rep1 and predicts rep2 (same target, so it only shows whether the
features can express a Gata4-like response at all; it is NOT a generalisation test).
"""
import argparse
import json
import time
from pathlib import Path

from vec import t3, t3model

p = argparse.ArgumentParser()
p.add_argument("--epochs", type=int, default=300)
p.add_argument("--no-network", action="store_true", help="drop the CollecTRI pair prior")
a = p.parse_args()

T = time.time()
S = {k: t3.load(k) for k in ["WT_E8.75", "WT_E9.5", "Gata4_1", "Gata4_2", "Mab21l2"]}
t3.transfer_states(S["WT_E8.75"], [S["Gata4_1"], S["Gata4_2"]])
t3.transfer_states(S["WT_E9.5"], [S["Mab21l2"]])
net = None if a.no_network else t3model.load_network()
ctx = {"Gata4": t3model.build_context(S["WT_E8.75"], "Gata4", net),
       "Mab21l2": t3model.build_context(S["WT_E9.5"], "Mab21l2", net)}
lab = {k: t3.decompose(S[k], S[w]) for k, (_, w) in t3.KO.items()}
row = lambda k: (ctx[t3.KO[k][0]], lab[k]["delta"], lab[k]["ok"])

folds = {
    "train Mab21l2 -> Gata4": ([row("Mab21l2")], "Gata4", ["Gata4_1", "Gata4_2"]),
    "train Gata4 -> Mab21l2": ([row("Gata4_1"), row("Gata4_2")], "Mab21l2", ["Mab21l2"]),
    "sanity: Gata4 rep1 -> rep2": ([row("Gata4_1")], "Gata4", ["Gata4_2"]),
}
out = {}
for name, (train, target, tests) in folds.items():
    pred = t3model.fit(train, epochs=a.epochs)(ctx[target])
    wt = S[t3.KO[tests[0]][1]]
    out[name] = {"mean_pi": float(pred["pi"].mean())}
    for k in tests:
        sc = t3.Scorer(S[k].X, wt.X)
        res = {"wt_identity": sc(t3.emit(wt, None, ctx[target].states)[0])}
        for s in (0.5, 1.0, 2.0):
            res[f"model x{s}"] = sc(t3.emit(wt, s * pred["yhat"], ctx[target].states)[0])
        out[name][k] = res
        for m, v in res.items():
            print(f"{name:28s} {k:8s} {m:12s} total {v['approx_total']:5.1f}  de {v['de_score']:+.3f}  "
                  f"dir {v['de_direction']:+.3f}  slope {v['severity_slope']:+.2f}  mmd {v['mmd_u']:.4f}", flush=True)
Path("runs").mkdir(exist_ok=True)
Path("runs/t3_loto.json").write_text(json.dumps(out, indent=1))
print(f"done in {time.time() - T:.0f}s")
