#!/usr/bin/env python
"""Build a T3 prediction file from WT E8.75 cells (keeps their 3D coordinates).

  python scripts/make_t3_pred.py --model wt_identity --out submissions/T3_bcatenin/wt_identity.h5ad
  python scripts/make_t3_pred.py --model prototype   --out submissions/T3_bcatenin/prototype_v0.h5ad

wt_identity : stratified WT E8.75 cells with the subsample's sampling noise removed (the board floor).
prototype   : vec.t3model trained on every challenge KO (Gata4 x2, Mab21l2), predicting Ctnnb1 prior-only.
              --ctnnb1-regulon adds CollecTRI's Ctnnb1 targets as the pair prior. That is effectively a
              curated Wnt target list: an "ask the organisers first" item in docs/external_data.md.
The file is validated against the board spec; nothing is uploaded.
"""
import argparse

from vec import t3, t3model
from vec.panels import board, genes
from vec.validate import validate

p = argparse.ArgumentParser()
p.add_argument("--model", choices=["wt_identity", "prototype"], required=True)
p.add_argument("--board", default="T3:bcatenin")
p.add_argument("--target", default="Ctnnb1")
p.add_argument("--ctnnb1-regulon", action="store_true")
p.add_argument("--scale", type=float, default=1.0)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", required=True)
a = p.parse_args()

spec = board(a.board)
panel = list(genes(a.board))
S = {k: t3.load(k) for k in ["WT_E8.75", "WT_E9.5", "Gata4_1", "Gata4_2", "Mab21l2"]}
wt = S["WT_E8.75"]
assert list(wt.genes) == panel, "WT gene order differs from the board panel"
t3.transfer_states(wt, [S["Gata4_1"], S["Gata4_2"]])
t3.transfer_states(S["WT_E9.5"], [S["Mab21l2"]])
states = sorted(set(wt.state))

delta = None
if a.model == "prototype":
    net = t3model.load_network()
    ctx = {"Gata4": t3model.build_context(wt, "Gata4", net), "Mab21l2": t3model.build_context(S["WT_E9.5"], "Mab21l2", net)}
    train = []
    for k, (tgt, w) in t3.KO.items():
        d = t3.decompose(S[k], S[w])
        train.append((ctx[tgt], d["delta"], d["ok"]))
    target_ctx = t3model.build_context(wt, a.target, net if a.ctnnb1_regulon else None)
    pred = t3model.fit(train)(target_ctx)
    delta = a.scale * pred["yhat"]
    print(f"predicted mean pi {pred['pi'].mean():.3f}; |delta| max {abs(delta).max():.3f}")

X, C, L = t3.emit(wt, delta, states, n_cells=spec["max_cells"], seed=a.seed)
path = t3.write_pred(a.out, X, C, panel, L)
problems = validate(path, a.board)
print(f"wrote {path} {X.shape}; validation: {'OK' if not problems else problems}")
