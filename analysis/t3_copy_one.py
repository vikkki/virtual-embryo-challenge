"""Grosswendt analogue of "predict beta-catenin from Gata4": predict KO i from ONE other KO j of the same
control, for every ordered pair, as (a) j's full per-state delta and (b) j's gene-wise offset only (median over
states, applied to every state). Reports the distribution of gains over the floor.

  python analysis/t3_copy_one.py   -> runs/t3_copy_one.json  (remote box; data-derived)
"""
import json
from pathlib import Path

import numpy as np

from vec import t3

idx = {k: v for k, v in t3.external_index().items() if k.startswith("grosswendt")}
wt = t3.load(next(iter(idx.values()))["wt"])
ko = {k: t3.load(v["ko"]) for k, v in idx.items()}
D = {k: t3.decompose(x, wt, min_cells=20) for k, x in ko.items()}
states = D[next(iter(D))]["states"]
floor, res = {}, {}
for i in ko:
    sc = t3.Scorer(ko[i].X, wt.X)
    floor[i] = sc(t3.emit(wt, None, states)[0])["approx_total"]
    for j in ko:
        if i == j:
            continue
        dj = D[j]["delta"]
        off = np.median(dj[D[j]["ok"]], 0)
        full = sc(t3.emit(wt, dj, states)[0])["approx_total"] - floor[i]
        offset = sc(t3.emit(wt, np.tile(off, (len(states), 1)), states)[0])["approx_total"] - floor[i]
        res[f"{i} <- {j}"] = {"full": full, "offset": offset}
    print(i, "floor", floor[i], "median gain full/offset:",
          np.median([v["full"] for k, v in res.items() if k.startswith(i + " ")]).round(1),
          np.median([v["offset"] for k, v in res.items() if k.startswith(i + " ")]).round(1), flush=True)
for kind in ("full", "offset"):
    g = np.array([v[kind] for v in res.values()])
    print(f"{kind:6s}: mean {g.mean():+.1f}  median {np.median(g):+.1f}  p10 {np.percentile(g, 10):+.1f}  "
          f"worst {g.min():+.1f}  beats floor {np.mean(g > 0):.0%} of {len(g)} pairs")
Path("runs/t3_copy_one.json").write_text(json.dumps({"floor": floor, "pairs": res}, indent=1))
