#!/usr/bin/env python
"""Held-out check of the prior-only model (vec.t3model) against wt_identity, over every KO we have: the
challenge KOs (Gata4 x2 at E8.75, Mab21l2 at E9.5) plus every external pair in data/processed/index.json.

  --split gene   for each KO gene: train on all OTHER genes' examples, predict this gene in each of its contexts
  --split study  for each study: train on the other studies, minus any example of a gene the held-out study
                 knocks out; predict the whole study (the honest test for "a new lab, a new control")
  --balance study  every study gets the same total training weight
  --nuisance       KOs that share one control (and cover >=2 genes) share a learned offset while training,
                   which is dropped at prediction time
  --blind test     describe each held-out target the way we can describe Ctnnb1 (off the panel: no expression
                   pattern, no co-expression, no regulon); --blind all does the same for the training targets
  --cre            conditional challenge KOs with a known Cre (Gata4: Mesp1-Cre) act only in that lineage:
                   activity = lineage weight x target expression; under --blind, the lineage weight alone
                   (what we know for Ctnnb1)
  --gate           with --cre, zero the predicted shift outside the lineage (cell-autonomous effects only)
  --keep-regulon   under --blind, keep the target's CollecTRI regulon (only expression info is hidden)
  --shuffle-regulon SEED  null control: move the held-out target's regulon onto random panel genes (same size
                   and signs) in the test context
  --seed-sets K    refit each fold K times with different seeds, to see how much of a gain is noise
  --folds A B      only run these folds (gene names or study ids)

Each prediction is emitted from that context's WT cells and scored with veckit's T3 metrics against the real KO.

  python scripts/t3_loto.py [--split gene] [--balance study] [--nuisance] [--blind test] [--seed-sets 3]
                            [--epochs 300] [--no-network] [--skip grosswendt] [--out runs/t3_loto.json]
Training needs CUDA (run on the remote box).
"""
import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np

from vec import lineage, t3, t3model

p = argparse.ArgumentParser()
p.add_argument("--epochs", type=int, default=300)
p.add_argument("--no-network", action="store_true", help="drop the CollecTRI pair prior")
p.add_argument("--skip", nargs="*", default=[], help="drop external pairs whose name contains any of these")
p.add_argument("--split", choices=["gene", "study"], default="gene")
p.add_argument("--balance", choices=["none", "study"], default="none")
p.add_argument("--nuisance", action="store_true", help="shared learned offset per multi-gene control group")
p.add_argument("--blind", choices=["none", "test", "all"], default="none")
p.add_argument("--cre", action="store_true")
p.add_argument("--gate", action="store_true")
p.add_argument("--keep-regulon", action="store_true")
p.add_argument("--shuffle-regulon", type=int, default=None)
p.add_argument("--seed-sets", type=int, default=1)
p.add_argument("--folds", nargs="*", default=None)
p.add_argument("--out", default="runs/t3_loto.json")
a = p.parse_args()

T = time.time()
net = None if a.no_network else t3model.load_network()
ex = t3model.examples(net, skip=a.skip, cre=a.cre)
if a.blind != "none":
    cache = {}
    for e in ex:
        key = (id(e["wt"]), e["target"], e["cre"])
        if key not in cache:
            act = t3model.cre_activity(e["wt"], None, e["cre"]) if e["cre"] else None
            cache[key] = t3model.build_context(e["wt"], e["target"], net, blind=True, activity=act,
                                               keep_regulon=a.keep_regulon)
        e["ctx_blind"] = cache[key]
print(f"{len(ex)} KO examples, {len({e['target'] for e in ex})} genes; loaded in {time.time() - T:.0f}s", flush=True)

if a.split == "gene":
    folds = [(t, [e for e in ex if e["target"] == t]) for t in sorted({e["target"] for e in ex})]
else:
    folds = [(s, [e for e in ex if e["source"] == s]) for s in sorted({e["source"] for e in ex})]
if a.folds:
    folds = [f for f in folds if f[0] in a.folds]
SCALES = (0.5, 1.0)


def shuffled_regulon(ctx, seed):
    """Same context with the regulon features (reg_w, activity x reg_w) moved to random measured genes."""
    idx = [t3model.FEATURES.index(f) for f in ("reg_w", "t_x_reg")]
    genes = np.flatnonzero(ctx.gmask > 0)
    perm = np.arange(ctx.X.shape[1])
    perm[genes] = np.random.default_rng(seed).permutation(genes)
    X = ctx.X.copy()
    X[:, :, idx] = ctx.X[:, perm][:, :, idx]
    return dataclasses.replace(ctx, X=X)


out = {}
for fold, test in folds:
    held = {e["target"] for e in test}
    tr = [e for e in ex if e["target"] not in held and (a.split == "gene" or e["source"] != fold)]
    if a.blind == "all":
        tr = [{**e, "ctx": e["ctx_blind"]} for e in tr]
    print(f"fold {fold}: train {len(tr)} examples / {len({e['target'] for e in tr})} genes", flush=True)
    for e in test:
        out[e["name"]] = {"target": e["target"], "source": e["source"], "runs": []}
    for k in range(a.seed_sets):
        predict = t3model.fit_examples(tr, balance=a.balance, nuisance=a.nuisance, epochs=a.epochs,
                                       seeds=tuple(range(3 * k, 3 * k + 3)))
        for e in test:
            wt = e["wt"]
            ctx = e["ctx_blind"] if a.blind != "none" else e["ctx"]
            if a.shuffle_regulon is not None:
                ctx = shuffled_regulon(ctx, a.shuffle_regulon)
            n = min(7449, wt.X.shape[0])
            sc = t3.Scorer(e["ko"].X, wt.X)
            if k == 0:
                out[e["name"]]["wt_identity"] = sc(t3.emit(wt, None, ctx.states, n_cells=n)[0])
            yhat = predict(ctx)["yhat"]
            if a.gate and e["cre"]:
                yhat = yhat * lineage.weights(ctx.states, e["cre"])[:, None]
            res = {f"model x{s}": sc(t3.emit(wt, s * yhat, ctx.states, n_cells=n)[0]) for s in SCALES}
            out[e["name"]]["runs"].append(res)
            w = out[e["name"]]["wt_identity"]
            for m, v in res.items():
                print(f"{e['name']:28s} seeds{k} {m:10s} total {w['approx_total']:5.1f} -> {v['approx_total']:5.1f}  "
                      f"de {v['de_score']:+.3f}  dir {v['de_direction']:+.3f}  slope {v['severity_slope']:+.2f}  "
                      f"mmd {v['mmd_u']:.4f}", flush=True)

print(f"\nconfig: {vars(a)}")
for s in SCALES:
    gains = {k: np.array([r[f"model x{s}"]["approx_total"] for r in v["runs"]]) - v["wt_identity"]["approx_total"]
             for k, v in out.items()}
    print(f"model x{s} minus floor, per example (mean over seed sets, [min, max]):")
    for k, g in gains.items():
        print(f"  {k:28s} {g.mean():+5.1f}  [{g.min():+5.1f}, {g.max():+5.1f}]")
    groups = {"all": list(gains), "non-Grosswendt": [k for k in gains if out[k]["source"] != "GSE137337"]}
    groups |= {f"source {x}": [k for k in gains if out[k]["source"] == x] for x in sorted({v["source"] for v in out.values()})}
    for name, ks in groups.items():
        if ks:
            per_set = np.array([gains[k] for k in ks]).mean(0)          # group mean, one value per seed set
            v = [gains[k].mean() for k in ks]
            print(f"  {name:28s} mean {np.mean(v):+5.1f} (seed sets {', '.join(f'{x:+.1f}' for x in per_set)})  "
                  f"above floor {sum(x > 0 for x in v)}/{len(v)}")
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text(json.dumps({"config": vars(a), "results": out}, indent=1))
print(f"done in {time.time() - T:.0f}s")
