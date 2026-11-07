#!/usr/bin/env python
"""Build a T3 prediction file from WT E8.75 cells (keeps their 3D coordinates).

  python scripts/make_t3_pred.py --model wt_identity --out submissions/T3_bcatenin/wt_identity.h5ad
  python scripts/make_t3_pred.py --model prototype   --out submissions/T3_bcatenin/prototype_v0.h5ad
  python scripts/make_t3_pred.py --model trained --balance study --nuisance --out submissions/T3_bcatenin/<name>.h5ad

wt_identity : stratified WT E8.75 cells with the subsample's sampling noise removed (the board floor).
prototype   : vec.t3model trained on every challenge KO (Gata4 x2, Mab21l2), predicting Ctnnb1 prior-only.
              --ctnnb1-regulon adds CollecTRI's Ctnnb1 targets as the pair prior. That is effectively a
              curated Wnt target list: an "ask the organisers first" item in docs/external_data.md.
gata4_offset: the shared "Mesp1-Cre KO vs this WT" signature. Gata4 and beta-catenin are both Mesp1-Cre KOs
              scored against the same WT E8.75. For each Gata4 replicate, take the median within-state shift
              per gene over the states OUTSIDE the Mesp1 lineage (--offset-states nonlineage; 'all' = every
              state, as in analysis/t3_copy_one.py), average the replicates, and add it to every state.
              Uses the official Gata4 validation release (allowed for P3 preparation).
knowledge   : gata4_offset (--base) plus a knowledge layer: CollecTRI's Ctnnb1 regulon (a curated Wnt target list:
              ask the organisers before uploading) acting only in the Mesp1 lineage. In state s, activated
              targets drop and repressed targets rise by kappa x lineage weight x the target's own WT mean:
              delta[s,g] = -kappa * sign_g * L_s * mu[s,g]. Most canonical Wnt readouts (Axin2, Sp5, Lef1,
              Nkd1, Notum) are not on the panel, hence the curated regulon. --targets curated uses
              docs/wnt_targets_curated.csv instead: the Nusse-lab Wnt target list (wnt.stanford.edu) merged with
              CollecTRI, weight 1 where both agree, 0.5 for one source, conflicts and held-out-derived entries dropped.
trained     : same model trained on every KO example (challenge + data/processed external pairs), with the
              --balance / --nuisance options of scripts/t3_loto.py. Needs CUDA: run it on the remote box.
The file is validated against the board spec; nothing is uploaded.
"""
import argparse

from vec import t3, t3model
from vec.panels import board, genes
from vec.validate import validate

p = argparse.ArgumentParser()
p.add_argument("--model", choices=["wt_identity", "prototype", "trained", "gata4_offset", "knowledge"], required=True)
p.add_argument("--base", choices=["gata4_offset", "none"], default="gata4_offset")
p.add_argument("--kappa", type=float, default=0.5)
p.add_argument("--targets", choices=["collectri", "curated"], default="collectri",
               help="curated = docs/wnt_targets_curated.csv (Nusse-lab Wnt target list + CollecTRI, sourced)")
p.add_argument("--offset-states", choices=["nonlineage", "all"], default="nonlineage")
p.add_argument("--board", default="T3:bcatenin")
p.add_argument("--target", default="Ctnnb1")
p.add_argument("--ctnnb1-regulon", action="store_true")
p.add_argument("--balance", choices=["none", "study"], default="study")
p.add_argument("--nuisance", action="store_true")
p.add_argument("--epochs", type=int, default=300)
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
elif a.model in ("gata4_offset", "knowledge"):
    import numpy as np
    from vec import lineage
    lin = lineage.weights(states, "Mesp1")
    offs = []
    for rep in ("Gata4_1", "Gata4_2"):
        d = t3.decompose(S[rep], wt)
        use = d["ok"] & ((lin == 0) if a.offset_states == "nonlineage" else True)
        offs.append(np.median(d["delta"][use], 0))
        print(f"{rep}: offset from {use.sum()} states: {[s for s, u in zip(states, use) if u]}")
    r = np.corrcoef(offs[0], offs[1])[0, 1]
    off = a.scale * np.mean(offs, 0)
    print(f"replicate offsets r = {r:.2f}; genes with |offset| > 0.05: {(abs(off) > 0.05).sum()}; "
          f"max |offset| {abs(off).max():.3f}")
    delta = np.tile(off, (len(states), 1))
    if a.model == "knowledge":
        if a.base == "none":
            delta[:] = 0.0
        if a.targets == "curated":
            import pandas as pd
            cur = pd.read_csv(t3.ROOT / "docs/wnt_targets_curated.csv").set_index("gene")["weight"]
            w = np.array([float(cur.get(g, 0.0)) for g in panel])
        else:
            w = t3model.regulon(t3model.load_network(), a.target, panel)
        mu = t3.state_stats(wt, states)["mu"]
        layer = -a.kappa * lin[:, None] * w[None, :] * np.nan_to_num(mu)
        delta = delta + layer
        hit = np.flatnonzero(w)
        print(f"knowledge layer: {len(hit)} {a.target} targets ({(w > 0).sum()} activated -> down, "
              f"{(w < 0).sum()} repressed -> up) in {int((lin > 0).sum())} lineage states; "
              f"pseudobulk change of the layer: {np.round(np.sort((wt_f := t3.state_stats(wt, states)['f']) @ layer)[:5], 3)} ...")
elif a.model == "trained":
    net = t3model.load_network()
    ex = t3model.examples(net)
    wt = next(e["wt"] for e in ex if e["name"] == "Gata4_1")       # WT E8.75 with its states
    assert list(wt.genes) == panel and sorted(set(wt.state)) == states
    print(f"training on {len(ex)} examples / {len({e['target'] for e in ex})} genes "
          f"(balance={a.balance}, nuisance={a.nuisance})")
    predict = t3model.fit_examples(ex, balance=a.balance, nuisance=a.nuisance, epochs=a.epochs)
    pred = predict(t3model.build_context(wt, a.target, net if a.ctnnb1_regulon else None))
    delta = a.scale * pred["yhat"]
    print(f"predicted mean pi {pred['pi'].mean():.3f}; |delta| max {abs(delta).max():.3f}")

X, C, L = t3.emit(wt, delta, states, n_cells=spec["max_cells"], seed=a.seed)
path = t3.write_pred(a.out, X, C, panel, L)
problems = validate(path, a.board)
print(f"wrote {path} {X.shape}; validation: {'OK' if not problems else problems}")
