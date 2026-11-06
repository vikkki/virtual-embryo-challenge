"""T3 building blocks: load samples, map cells onto WT states, per-state statistics, emission, local scoring.

Conventions (match veckit): expression is .X = log1p(normalize_total 1e4); pseudobulk = mean of .X over cells.
A "state" is a WT cell type at the stage being predicted (e.g. WT E8.75 `celltype`). KO cells are mapped onto
those states by kNN label transfer, so every KO effect is expressed relative to its stage-matched WT:

    pb(KO) - pb(WT) = sum_s f_s^WT (mu_s^KO - mu_s^WT)        within-state ("expression") part
                    + sum_s (f_s^KO - f_s^WT) mu_s^KO          composition part
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"

SAMPLES = {
    "WT_E8.5": "T2/E8.5.h5ad",
    "WT_E8.75": "T2/E8.75.h5ad",
    "WT_E9.5": "T2/E9.5.h5ad",
    "Gata4_1": "T3/E8.75_gata4_ko_1.h5ad",
    "Gata4_2": "T3/E8.75_gata4_ko_2.h5ad",
    "Mab21l2": "T3/E9.5_mab21l2_ko.h5ad",
}
# KO sample -> (target gene, stage-matched WT sample)
KO = {
    "Gata4_1": ("Gata4", "WT_E8.75"),
    "Gata4_2": ("Gata4", "WT_E8.75"),
    "Mab21l2": ("Mab21l2", "WT_E9.5"),
}


@dataclass
class Sample:
    name: str
    X: np.ndarray            # cells x genes, float32, log1p-normalised
    genes: np.ndarray
    coords: np.ndarray       # cells x 3
    celltype: np.ndarray     # file's own labels (vocabularies differ between files)
    state: np.ndarray | None = None   # labels in the stage-matched WT vocabulary
    present: np.ndarray | None = None  # genes the dataset measured (external scRNA); None = all
    target_expr: np.ndarray | None = None  # per-cell log-expression of the KO target over all genes (WT only)


def external_index() -> dict:
    """data/processed/index.json written by vec.external (empty if nothing has been ingested)."""
    import json
    p = DATA / "processed/index.json"
    return json.loads(p.read_text()) if p.exists() else {}


def load(name: str, n: int | None = None, seed: int = 0) -> Sample:
    """Challenge samples by key (SAMPLES), or processed external files by name (e.g. "hand2_null_E8.25_WT"),
    which come with their states already assigned."""
    path = DATA / SAMPLES[name] if name in SAMPLES else DATA / "processed" / f"{name}.h5ad"
    a = ad.read_h5ad(path)
    if n and a.n_obs > n:
        a = a[np.sort(np.random.default_rng(seed).choice(a.n_obs, n, replace=False))]
    X = a.X.toarray() if sp.issparse(a.X) else np.asarray(a.X)
    ct = np.asarray(a.obs["celltype"]).astype(str) if "celltype" in a.obs else np.full(a.n_obs, "NA")
    s = Sample(name, X.astype(np.float32), np.asarray(a.var_names),
               np.asarray(a.obsm["spatial_3D"], dtype=np.float32), ct)
    if "state" in a.obs:
        s.state = np.asarray(a.obs["state"]).astype(str)
    if "present" in a.var:
        s.present = np.asarray(a.var["present"]).astype(bool)
    if "target_expr" in a.obs:
        s.target_expr = np.asarray(a.obs["target_expr"], dtype=np.float32)
    return s


def transfer_states(ref: Sample, queries: list[Sample], k: int = 15, n_pc: int = 50) -> None:
    """kNN label transfer of `ref.celltype` onto each query (in a PCA space fitted on the reference)."""
    ref.state = ref.celltype.copy()
    pca = PCA(n_pc, random_state=0).fit(ref.X)
    clf = KNeighborsClassifier(n_neighbors=k, n_jobs=-1).fit(pca.transform(ref.X), ref.celltype)
    for q in queries:
        q.state = clf.predict(pca.transform(q.X))


def state_stats(s: Sample, states: list[str]) -> dict[str, np.ndarray]:
    """Per-state fraction f (S), mean mu (S x G), nonzero fraction nz (S x G), and counts n (S)."""
    G = s.X.shape[1]
    f, n = np.zeros(len(states)), np.zeros(len(states), int)
    mu, nz = np.full((len(states), G), np.nan), np.full((len(states), G), np.nan)
    for i, st in enumerate(states):
        m = s.state == st
        n[i] = m.sum()
        if n[i]:
            mu[i], nz[i] = s.X[m].mean(0), (s.X[m] > 0).mean(0)
    f = n / n.sum()
    return {"f": f, "mu": mu, "nz": nz, "n": n}


def decompose(ko: Sample, wt: Sample, min_cells: int = 30) -> dict:
    """Split pb(KO) - pb(WT) into within-state and composition parts (see module docstring).

    Within-state deltas for states with < min_cells KO cells are set to 0 (too noisy).
    """
    states = sorted(set(wt.state))
    K, W = state_stats(ko, states), state_stats(wt, states)
    ok = (K["n"] >= min_cells) & (W["n"] >= min_cells)
    delta = np.where(ok[:, None], K["mu"] - W["mu"], 0.0)
    within = (W["f"][:, None] * delta).sum(0)
    mu_ko = np.where(ok[:, None], K["mu"], W["mu"])
    comp = ((K["f"] - W["f"])[:, None] * np.nan_to_num(mu_ko)).sum(0)
    total = ko.X.mean(0) - wt.X.mean(0)
    with np.errstate(divide="ignore"):
        logratio = np.log((K["f"] + 1e-4) / (W["f"] + 1e-4))
    return {"states": states, "delta": delta, "ok": ok, "within": within, "comp": comp, "total": total,
            "f_ko": K["f"], "f_wt": W["f"], "comp_logratio": logratio, "n_ko": K["n"], "n_wt": W["n"]}


# ----------------------------------------------------------------------------------------------- emission
def _shift_down(X: np.ndarray, d: np.ndarray, iters: int = 40) -> np.ndarray:
    """Per gene, find c<=0 with mean(max(X+c,0)) = mean(X)+d (d<=0); X zeros stay zero."""
    target = np.maximum(X.mean(0) + d, 0.0)
    lo, hi = -(X.max(0) + 1e-6), np.zeros(X.shape[1])
    for _ in range(iters):
        c = (lo + hi) / 2
        too_high = np.maximum(X + c, 0).mean(0) > target
        hi, lo = np.where(too_high, c, hi), np.where(too_high, lo, c)
    return np.maximum(X + (lo + hi) / 2, 0)


def _shift_up(X: np.ndarray, d: np.ndarray, rng, max_c: float = 1.5) -> np.ndarray:
    """Per gene, raise mean by d>=0: shift expressing cells by up to `max_c`, then recruit non-expressing
    cells (given values drawn from the expressing cells, or 1.0 if none express) for the remainder."""
    X = X.copy()
    n = X.shape[0]
    for g in np.where(d > 1e-6)[0]:
        x = X[:, g]
        on = x > 0
        p = on.mean()
        add = d[g] * n                       # total log-units to add
        if p > 0:
            c = min(add / on.sum(), max_c)
            x[on] += c
            add -= c * on.sum()
        if add > 1e-6:
            off = np.where(~on)[0]
            vals = x[x > 0] if (x > 0).any() else np.array([1.0])
            k = min(len(off), int(np.ceil(add / max(vals.mean(), 1e-3))))
            if k:
                pick = rng.choice(off, k, replace=False)
                x[pick] = rng.choice(vals, k)
            add -= x[pick].sum() if k else 0
            if add > 1e-6:                   # still short (all cells used): uniform shift
                x += add / n
        X[:, g] = x
    return X


def emit(wt: Sample, delta: np.ndarray | None, states: list[str], comp_logratio: np.ndarray | None = None,
         n_cells: int = 7449, seed: int = 0, exact_wt: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Turn per-state deltas (S x G, on mean-log scale) and composition log-ratios (S) into cells.

    Cells are drawn from the WT by state (stratified, so WT composition is reproduced exactly when
    comp_logratio is None), their 3D coordinates are kept, and each state's genes are shifted so that the
    state's mean moves by delta[s]. With exact_wt, the shift is measured from the FULL WT state mean, so the
    subsample's own sampling noise is removed (a zero delta then reproduces the WT pseudobulk exactly).
    """
    rng = np.random.default_rng(seed)
    f = np.array([(wt.state == s).mean() for s in states])
    if comp_logratio is not None:
        f = f * np.exp(comp_logratio)
    f = f / f.sum()
    n_s = np.floor(f * n_cells).astype(int)
    rem = n_cells - n_s.sum()
    n_s[np.argsort(-(f * n_cells - n_s))[:rem]] += 1
    Xs, Cs, Ls = [], [], []
    for i, s in enumerate(states):
        if n_s[i] == 0:
            continue
        pool = np.where(wt.state == s)[0]
        idx = rng.choice(pool, n_s[i], replace=n_s[i] > len(pool))
        X = wt.X[idx].copy()
        d = np.zeros(X.shape[1]) if delta is None else np.nan_to_num(delta[i]).astype(float)
        if exact_wt:
            d = d + (wt.X[pool].mean(0) - X.mean(0))
        dn, up = np.minimum(d, 0), np.maximum(d, 0)
        if (dn < 0).any():
            X = _shift_down(X, dn)
        if (up > 0).any():
            X = _shift_up(X, up, rng)
        Xs.append(X); Cs.append(wt.coords[idx]); Ls.append(np.full(len(idx), s))
    return np.vstack(Xs).astype(np.float32), np.vstack(Cs), np.concatenate(Ls)


def write_pred(path, X: np.ndarray, coords: np.ndarray, genes, state: np.ndarray | None = None) -> Path:
    path = Path(path)
    obs = pd.DataFrame(index=[f"cell{i}" for i in range(X.shape[0])])
    if state is not None:
        obs["celltype"] = pd.Categorical(state)
    a = ad.AnnData(sp.csr_matrix(X, dtype=np.float32), obs=obs, var=pd.DataFrame(index=list(genes)))
    a.obsm["spatial_3D"] = np.asarray(coords, dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    a.write_h5ad(path, compression="gzip")
    return path


# ------------------------------------------------------------------------------------ local scoring (fast)
class Scorer:
    """veckit's T3 primary metrics against a fixed (truth, WT) pair, with the truth's DE genes cached.

    Uses veckit's own functions; only `de_genes` is memoised (it is the slow Wilcoxon step and depends on
    truth and WT only). mmd_u is veckit's unbiased multi-kernel MMD in a PCA fitted on the truth.
    """

    def __init__(self, truth: np.ndarray, wt: np.ndarray, board_key: str = "T3:gata4"):
        import common.core_metrics as cm
        if not hasattr(cm, "_orig_de_genes"):
            cm._orig_de_genes, cm._de_cache = cm.de_genes, {}

            def cached(X_cond, X_ref, *a, **k):
                key = (id(X_cond), id(X_ref), a, tuple(sorted(k.items())))
                if key not in cm._de_cache:
                    cm._de_cache[key] = cm._orig_de_genes(X_cond, X_ref, *a, **k)
                return cm._de_cache[key]
            cm.de_genes = cached
        self.cm, self.truth, self.wt, self.key = cm, truth, wt, board_key

    def __call__(self, pred: np.ndarray, mmd: bool = True) -> dict:
        from .score import approx_score
        cm = self.cm
        de = cm.de_score(pred, self.truth, self.wt)
        slope, r2 = cm.severity_slope(pred, self.truth, self.wt)
        ratio, _ = cm.severity_ratio(pred, self.truth, self.wt)
        m = {"de_score": de["score"], "de_direction": cm.de_direction(pred, self.truth, self.wt),
             "severity_slope": slope, "severity_ratio": ratio, "slope_r2": r2}
        if mmd:
            m["mmd_u"] = cm.mmd_unbiased(pred, self.truth)
            m.update({f"approx_{k}": v for k, v in approx_score(m, self.key).items()})
        return {k: (round(float(v), 4) if v is not None else None) for k, v in m.items()}
