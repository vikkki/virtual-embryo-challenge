#!/usr/bin/env python
"""Leave-one-GENE-out check of the prior-only model (vec.t3model) against wt_identity, over every KO we have:
the challenge KOs (Gata4 x2 at E8.75, Mab21l2 at E9.5) plus every external pair in data/processed/index.json.

For each KO gene: train on all OTHER genes' examples, predict this gene in each of its own contexts (stages /
replicates), emit cells from that context's WT and score with veckit's T3 metrics against the real KO.

  python scripts/t3_loto.py [--epochs 300] [--no-network] [--skip grosswendt]    -> runs/t3_loto.json
Training needs CUDA (run on the remote box).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from vec import t3, t3model

p = argparse.ArgumentParser()
p.add_argument("--epochs", type=int, default=300)
p.add_argument("--no-network", action="store_true", help="drop the CollecTRI pair prior")
p.add_argument("--skip", nargs="*", default=[], help="drop external pairs whose name contains any of these")
a = p.parse_args()

T = time.time()
net = None if a.no_network else t3model.load_network()
ex = []          # one entry per KO example: name, target, wt, ko, ctx, label
S = {k: t3.load(k) for k in ["WT_E8.75", "WT_E9.5", "Gata4_1", "Gata4_2", "Mab21l2"]}
t3.transfer_states(S["WT_E8.75"], [S["Gata4_1"], S["Gata4_2"]])
t3.transfer_states(S["WT_E9.5"], [S["Mab21l2"]])
for k, (tgt, w) in t3.KO.items():
    ex.append({"name": k, "target": tgt, "wt": S[w], "ko": S[k], "source": "challenge"})
for name, meta in t3.external_index().items():
    if any(s in name for s in a.skip):
        continue
    ex.append({"name": name, "target": meta["target"], "wt": t3.load(meta["wt"]), "ko": t3.load(meta["ko"]),
               "source": meta["source"]})
ctx_cache = {}
for e in ex:
    key = (id(e["wt"]), e["target"])
    if key not in ctx_cache:
        ctx_cache[key] = t3model.build_context(e["wt"], e["target"], net)
    e["ctx"] = ctx_cache[key]
    d = t3.decompose(e["ko"], e["wt"], min_cells=20)
    e["label"] = (d["delta"], d["ok"])
print(f"{len(ex)} KO examples, {len({e['target'] for e in ex})} genes; loaded in {time.time() - T:.0f}s", flush=True)

out = {}
for tgt in sorted({e["target"] for e in ex}):
    train = [(e["ctx"], *e["label"]) for e in ex if e["target"] != tgt]
    predict = t3model.fit(train, epochs=a.epochs)
    for e in [e for e in ex if e["target"] == tgt]:
        wt, ctx = e["wt"], e["ctx"]
        n = min(7449, wt.X.shape[0])
        sc = t3.Scorer(e["ko"].X, wt.X)
        yhat = predict(ctx)["yhat"]
        res = {"wt_identity": sc(t3.emit(wt, None, ctx.states, n_cells=n)[0])}
        for s in (0.5, 1.0):
            res[f"model x{s}"] = sc(t3.emit(wt, s * yhat, ctx.states, n_cells=n)[0])
        out[e["name"]] = {"target": tgt, "source": e["source"], **res}
        for m, v in res.items():
            print(f"{e['name']:28s} {m:12s} total {v['approx_total']:5.1f}  de {v['de_score']:+.3f}  "
                  f"dir {v['de_direction']:+.3f}  slope {v['severity_slope']:+.2f}  mmd {v['mmd_u']:.4f}", flush=True)

gain = {k: v["model x1.0"]["approx_total"] - v["wt_identity"]["approx_total"] for k, v in out.items()}
print("\nmodel x1 minus floor, per example:", {k: round(g, 1) for k, g in gain.items()})
print(f"mean {np.mean(list(gain.values())):+.1f}; beats floor on {sum(g > 0 for g in gain.values())}/{len(gain)}")
Path("runs").mkdir(exist_ok=True)
Path("runs/t3_loto.json").write_text(json.dumps(out, indent=1))
print(f"done in {time.time() - T:.0f}s")
