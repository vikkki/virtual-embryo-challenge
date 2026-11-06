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

from . import t3

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


def load_network(path=None) -> pd.DataFrame:
    return pd.read_csv(path or t3.ROOT / "data/priors/collectri_mouse.csv.gz")


def regulon(net: pd.DataFrame, tf: str, genes: list[str]) -> np.ndarray:
    sub = net[(net.source_genesymbol == tf) & net.target_genesymbol.isin(genes) & (net.target_genesymbol != tf)]
    w = np.zeros(len(genes))
    for tg, stim, inh in zip(sub.target_genesymbol, sub.is_stimulation, sub.is_inhibition):
        w[genes.index(tg)] = 1.0 if stim else (-1.0 if inh else 0.5)
    return w


def build_context(wt: t3.Sample, target: str, net: pd.DataFrame | None = None,
                  activity: np.ndarray | None = None) -> Context:
    """`activity` (S,) overrides the target's per-state level; needed when the target is off the panel."""
    genes = list(wt.genes)
    states = sorted(set(wt.state))
    st = t3.state_stats(wt, states)
    mu, nz, f = np.nan_to_num(st["mu"]), np.nan_to_num(st["nz"]), st["f"]
    S, G = mu.shape
    on_panel = target in genes
    if activity is not None:
        a = np.asarray(activity, float)
    elif on_panel:
        a = mu[:, genes.index(target)]
    else:
        a = np.ones(S)                       # unknown: assume uniform
    a = a / (a.max() + 1e-9)
    gbar = (f[:, None] * mu).sum(0)
    ident = mu - gbar
    if on_panel:
        j = genes.index(target)
        Xc = wt.X - wt.X.mean(0)
        co = (Xc * Xc[:, [j]]).mean(0) / (Xc.std(0) * Xc[:, j].std() + 1e-9)
        co[j] = 0
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
    return Context(target, states, X, tsum, f)


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


def fit(train: list[tuple[Context, np.ndarray, np.ndarray]], seeds=(0, 1, 2), epochs: int = 300, lr: float = 3e-3):
    """train: list of (context, label delta (S,G), ok mask (S,)). Returns an ensemble predictor.

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
    models = []
    for seed in seeds:
        torch.manual_seed(seed)
        model = SpikeSlab().to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
        data = [(T((c.X - xm) / xs), T(c.tsum), T(np.nan_to_num(y)),
                 T((c.f * ok)[:, None].repeat(y.shape[1], 1))) for c, y, ok in train]
        for ep in range(epochs):
            opt.zero_grad()
            loss = sum(_loss(model, x, ts, y, w) for x, ts, y, w in data) / len(data)
            loss.backward()
            opt.step()
        models.append(model.eval())

    def predict(c: Context) -> dict[str, np.ndarray]:
        x = T((c.X - xm) / xs)
        with torch.no_grad():
            outs = [m(x, T(c.tsum)) for m in models]
        pi = torch.stack([o[0] for o in outs]).mean(0).cpu().numpy()
        yhat = torch.stack([(1 - o[0]) * o[1] for o in outs]).mean(0).cpu().numpy()
        return {"pi": pi, "yhat": yhat}
    return predict
