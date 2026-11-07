"""Prior-only spike-and-slab model for T3 (cellworld-v3-inspired, no "sources" branch).

Unit: (KO target j, WT context c, state s, gene g). The model never sees KO cells of the target it predicts;
for beta-catenin the rules forbid them, so the target is described only by prior features.

  target encoder   z_j = MLP(target summary)                        (prior-only: no sources, no attention)
  gene decoder     h = Linear(x_{s,g}) -> 2 FiLM residual blocks conditioned on z_j -> (logit pi, m, log tau)
  output           y_hat = (1 - pi) * m          spike at 0 with prob pi, slab N(m, tau^2) otherwise
  loss             spike-and-slab NLL + MSE(y_hat), weighted by WT state fraction (pseudobulk weight)

Labels are within-state deltas mu_s^KO - mu_s^WT against the stage-matched WT (vec.t3.decompose), so the
developmental stage shift never enters the labels. Composition is left at WT (local data shows the KO
composition shifts are largely dissection-specific and do not replicate across targets).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from . import lineage, t3

FEATURES = ["t_s", "log_f_s", "mu", "nz", "ident", "t_x_ident", "coexpr", "is_target", "reg_w", "t_x_reg",
            "gene_sd"]
TARGET_FEATURES = ["breadth", "max_level", "on_panel", "regulon_size", "is_tf"]


@dataclass
class Context:
    """Features for one (target, WT context): X (S, G, F), target summary (T,), states, weights (S,)."""
    target: str
    states: list[str]
    X: np.ndarray
    tsum: np.ndarray
    f: np.ndarray
    gmask: np.ndarray        # (G,) 1 = gene measured in this context (external scRNA may miss some)


def load_network(path=None) -> pd.DataFrame:
    return pd.read_csv(path or t3.ROOT / "data/priors/collectri_mouse.csv.gz")


def regulon(net: pd.DataFrame, tf: str, genes: list[str]) -> np.ndarray:
    sub = net[(net.source_genesymbol == tf) & net.target_genesymbol.isin(genes) & (net.target_genesymbol != tf)]
    w = np.zeros(len(genes))
    for tg, stim, inh in zip(sub.target_genesymbol, sub.is_stimulation, sub.is_inhibition):
        w[genes.index(tg)] = 1.0 if stim else (-1.0 if inh else 0.5)
    return w


def build_context(wt: t3.Sample, target: str, net: pd.DataFrame | None = None,
                  activity: np.ndarray | None = None, blind: bool = False, keep_regulon: bool = False) -> Context:
    """Target activity per state, in order of preference: `activity` (S,) given explicitly; the target's
    measured all-gene expression in the WT cells (`wt.target_expr`, external data); its panel transcript;
    otherwise uniform (unknown).

    blind=True describes the target the way we can describe Ctnnb1 for T3:bcatenin: off the panel, no
    expression pattern, no co-expression and no regulon (unless `activity` is given). keep_regulon=True keeps
    the regulon (what we would have for Ctnnb1 if curated Wnt target lists are allowed)."""
    genes = list(wt.genes)
    states = sorted(set(wt.state))
    st = t3.state_stats(wt, states)
    mu, nz, f = np.nan_to_num(st["mu"]), np.nan_to_num(st["nz"]), st["f"]
    S, G = mu.shape
    on_panel = not blind and target in genes and (wt.present is None or wt.present[genes.index(target)])
    texpr = None if blind else wt.target_expr
    if blind and not keep_regulon:
        net = None
    if activity is not None:
        a = np.asarray(activity, float)
    elif texpr is not None:
        a = np.array([texpr[wt.state == s].mean() for s in states])
    elif on_panel:
        a = mu[:, genes.index(target)]
    else:
        a = np.ones(S)                       # unknown: assume uniform
    a = a / (a.max() + 1e-9)
    gbar = (f[:, None] * mu).sum(0)
    ident = mu - gbar
    if on_panel or texpr is not None:
        t = texpr if texpr is not None else wt.X[:, genes.index(target)]
        Xc, tc = wt.X - wt.X.mean(0), t - t.mean()
        co = (Xc * tc[:, None]).mean(0) / (Xc.std(0) * tc.std() + 1e-9)
        if on_panel:
            co[genes.index(target)] = 0
    else:
        co = np.zeros(G)
    w = regulon(net, target, genes) if net is not None else np.zeros(G)
    is_t = np.zeros(G)
    if on_panel:
        is_t[genes.index(target)] = 1.0
    gsd = mu.std(0)
    rep = lambda v: np.tile(v, (S, 1))
    col = lambda v: np.tile(v[:, None], (1, G))
    X = np.stack([col(a), col(np.log(f + 1e-4)), mu, nz, ident, col(a) * ident, rep(co), rep(is_t), rep(w),
                  col(a) * rep(w), rep(gsd)], -1).astype(np.float32)
    tsum = np.array([(a > 0.2).mean(), a.max(), float(on_panel), float((w != 0).sum()) / 50,
                     float(net is not None and (net.source_genesymbol == target).any())], np.float32)
    gmask = np.ones(G, np.float32) if wt.present is None else wt.present.astype(np.float32)
    return Context(target, states, X, tsum, f, gmask)


def cre_activity(wt: t3.Sample, target: str | None, cre: str) -> np.ndarray:
    """Activity of a conditional KO per WT state: Cre-lineage weight x the target's own expression in that state
    (lineage weight alone when the target is not on the panel, as for Ctnnb1 or a blind test)."""
    states = sorted(set(wt.state))
    w = lineage.weights(states, cre)
    genes = list(wt.genes)
    if target is not None and target in genes:
        x = wt.X[:, genes.index(target)]
        w = w * np.array([x[wt.state == s].mean() for s in states])
    return w


def examples(net: pd.DataFrame | None, skip=(), min_cells: int = 20, cre: bool = False) -> list[dict]:
    """Every KO example we can train on: the challenge KOs (Gata4 x2 vs WT E8.75, Mab21l2 vs WT E9.5) plus every
    external pair in data/processed/index.json. Each dict has name, target, source (study), control (shared
    control group), cre (driver or None), wt, ko, ctx (Context) and label (within-state delta, ok mask).

    cre=True: conditional challenge KOs with a known driver (vec.lineage.CRE) get activity = lineage x expression."""
    ex = []
    S = {k: t3.load(k) for k in ["WT_E8.75", "WT_E9.5", "Gata4_1", "Gata4_2", "Mab21l2"]}
    t3.transfer_states(S["WT_E8.75"], [S["Gata4_1"], S["Gata4_2"]])
    t3.transfer_states(S["WT_E9.5"], [S["Mab21l2"]])
    for k, (tgt, w) in t3.KO.items():
        ex.append({"name": k, "target": tgt, "wt": S[w], "ko": S[k], "source": "challenge",
                   "control": f"challenge|{w}"})
    for name, meta in t3.external_index().items():
        if any(s in name for s in skip):
            continue
        ex.append({"name": name, "target": meta["target"], "wt": t3.load(meta["wt"]), "ko": t3.load(meta["ko"]),
                   "source": meta["source"], "control": f"{meta['source']}|{meta['control']}|{meta['stage']}"})
    cache = {}
    for e in ex:
        e["cre"] = lineage.CRE.get(e["name"]) if cre else None
        key = (id(e["wt"]), e["target"], e["cre"])
        if key not in cache:
            act = cre_activity(e["wt"], e["target"], e["cre"]) if e["cre"] else None
            cache[key] = build_context(e["wt"], e["target"], net, activity=act)
        e["ctx"] = cache[key]
        d = t3.decompose(e["ko"], e["wt"], min_cells=min_cells)
        e["label"] = (d["delta"], d["ok"])
    return ex


def fit_examples(ex: list[dict], balance: str = "none", nuisance: bool = False, **kw):
    """`fit` on example dicts from `examples`, with optional study balancing and shared-control offsets."""
    w = study_weights([e["source"] for e in ex]) if balance == "study" else None
    g = nuisance_groups([e["control"] for e in ex], [e["target"] for e in ex]) if nuisance else None
    return fit([(e["ctx"], *e["label"]) for e in ex], weights=w, groups=g, **kw)


class FiLMBlock(nn.Module):
    def __init__(self, d: int, dz: int):
        super().__init__()
        self.ln, self.mlp, self.film = nn.LayerNorm(d), nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d)), nn.Linear(dz, 2 * d)

    def forward(self, x, z):
        g, b = self.film(z).chunk(2, -1)
        return x + self.mlp(self.ln(x) * (1 + g) + b)


class SpikeSlab(nn.Module):
    def __init__(self, nf: int = len(FEATURES), nt: int = len(TARGET_FEATURES), d: int = 64, dz: int = 16):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(nt, dz), nn.GELU(), nn.Linear(dz, dz))
        self.inp = nn.Linear(nf, d)
        self.blocks = nn.ModuleList([FiLMBlock(d, dz), FiLMBlock(d, dz)])
        self.head = nn.Linear(d, 3)
        nn.init.constant_(self.head.bias, 0.0)
        self.head.bias.data[0] = 2.0         # start near "no effect" (pi ~ 0.88)

    def forward(self, x, tsum):
        z = self.enc(tsum)[..., None, None, :].expand(*x.shape[:-1], -1)
        h = self.inp(x)
        for b in self.blocks:
            h = b(h, z)
        lp, m, lt = self.head(h).unbind(-1)
        return torch.sigmoid(lp), m, nn.functional.softplus(lt) + 0.02


def _loss(model, x, ts, y, wgt, sigma0=0.05):
    pi, m, tau = model(x, ts)
    spike = torch.distributions.Normal(0.0, sigma0).log_prob(y)
    slab = torch.distributions.Normal(m, torch.sqrt(tau ** 2 + sigma0 ** 2)).log_prob(y)
    nll = -torch.logsumexp(torch.stack([torch.log(pi + 1e-6) + spike, torch.log(1 - pi + 1e-6) + slab]), 0)
    mse = ((1 - pi) * m - y) ** 2
    return ((nll + 10 * mse) * wgt).sum() / wgt.sum()


def _norm_stats(ctxs):
    X = np.concatenate([c.X.reshape(-1, c.X.shape[-1]) for c in ctxs])
    return X.mean(0), X.std(0) + 1e-6


def study_weights(studies: list[str]) -> np.ndarray:
    """Per-example weights giving every study the same total weight (1 / number of its examples)."""
    n = pd.Series(studies).value_counts()
    return np.array([1.0 / n[s] for s in studies])


def nuisance_groups(controls: list[str], targets: list[str], min_targets: int = 2) -> list[str | None]:
    """Group id per example for the shared-control offset: examples that share a control AND cover at least
    `min_targets` different KO genes. Replicates of one gene against one control get no group, otherwise the
    offset would absorb that gene's own effect."""
    df = pd.DataFrame({"c": controls, "t": targets})
    many = df.groupby("c").t.nunique()
    return [c if many[c] >= min_targets else None for c in controls]


def fit(train: list[tuple[Context, np.ndarray, np.ndarray]], seeds=(0, 1, 2), epochs: int = 300, lr: float = 3e-3,
        weights=None, groups=None):
    """train: list of (context, label delta (S,G), ok mask (S,)). Returns an ensemble predictor.

    weights: per-example loss weights (default: equal), e.g. `study_weights` so a large series of KOs against
    one control doesn't dominate. groups: per-example group id or None. Examples sharing a group share a learned
    (state, gene) offset that is subtracted from their labels while training and never used for prediction. It
    absorbs the "these KOs vs that control" signature, so the network learns the gene-specific part.

    Trains on CUDA. Training is meant to run on the remote GPU box (scripts/remote.sh); on a machine
    without CUDA it refuses unless VEC_ALLOW_CPU_TRAIN=1 is set.
    """
    if torch.cuda.is_available():
        dev = torch.device("cuda")
    elif os.environ.get("VEC_ALLOW_CPU_TRAIN") == "1":
        dev = torch.device("cpu")
    else:
        raise RuntimeError("no CUDA here: run training on the remote box (scripts/remote.sh run ...), "
                           "or set VEC_ALLOW_CPU_TRAIN=1 to train on CPU deliberately")
    xm, xs = _norm_stats([c for c, _, _ in train])
    T = lambda a, dt=torch.float32: torch.as_tensor(np.asarray(a), dtype=dt, device=dev)
    ew = np.ones(len(train)) if weights is None else np.asarray(weights, float)
    ew = ew / ew.sum()
    groups = [None] * len(train) if groups is None else list(groups)
    gstates = {}                                     # group -> union of its examples' states (row order)
    for (c, _, _), g in zip(train, groups):
        if g is not None:
            gstates.setdefault(g, [])
            gstates[g] += [s for s in c.states if s not in gstates[g]]
    rows = [None if g is None else torch.as_tensor([gstates[g].index(s) for s in c.states], device=dev)
            for (c, _, _), g in zip(train, groups)]
    data = [(T((c.X - xm) / xs), T(c.tsum), T(np.nan_to_num(y)),
             T((c.f * ok)[:, None] * c.gmask[None, :])) for c, y, ok in train]
    G = train[0][0].X.shape[1]
    models = []
    for seed in seeds:
        torch.manual_seed(seed)
        model = SpikeSlab().to(dev)
        off = {g: torch.zeros(len(st), G, device=dev, requires_grad=True) for g, st in gstates.items()}
        opt = torch.optim.AdamW([{"params": model.parameters(), "weight_decay": 1e-3},
                                 {"params": list(off.values()), "weight_decay": 0.0}], lr=lr)
        for ep in range(epochs):
            opt.zero_grad()
            loss = 0.0
            for (x, ts, y, w), e, g, r in zip(data, ew, groups, rows):
                loss = loss + e * _loss(model, x, ts, y if g is None else y - off[g][r], w)
            loss.backward()
            opt.step()
        models.append(model.eval())

    def predict(c: Context) -> dict[str, np.ndarray]:
        x = T((c.X - xm) / xs)
        with torch.no_grad():
            outs = [m(x, T(c.tsum)) for m in models]
        pi = torch.stack([o[0] for o in outs]).mean(0).cpu().numpy()
        yhat = torch.stack([(1 - o[0]) * o[1] for o in outs]).mean(0).cpu().numpy()
        return {"pi": pi, "yhat": yhat * c.gmask[None, :]}
    return predict
