"""T3 exploration: stage shift vs KO effect, and sparsity of KO responses (genes and cells).

Writes numbers to private/ (data-derived; keep out of the public repo).
"""
import json
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

rng = np.random.default_rng(0)
D = Path("data")
N = 15000  # cells per sample for the kNN analysis


def load(f, n=None):
    a = ad.read_h5ad(D / f)
    if n and a.n_obs > n:
        a = a[np.sort(rng.choice(a.n_obs, n, replace=False))].copy()
    X = a.X.toarray() if sp.issparse(a.X) else np.asarray(a.X)
    return a, X.astype(np.float32)


S = {
    "WT_E8.5": "T2/E8.5.h5ad",
    "WT_E8.75": "T2/E8.75.h5ad",
    "WT_E9.5": "T2/E9.5.h5ad",
    "Gata4_1": "T3/E8.75_gata4_ko_1.h5ad",
    "Gata4_2": "T3/E8.75_gata4_ko_2.h5ad",
    "Mab21l2": "T3/E9.5_mab21l2_ko.h5ad",
}
A, X = {}, {}
for k, f in S.items():
    A[k], X[k] = load(f, N)
genes = np.array(A["WT_E8.75"].var_names)
pb = {k: np.log1p(np.expm1(x).mean(0)) for k, x in X.items()}  # pseudobulk on log scale

def d(a, b):
    return pb[a] - pb[b]

pairs = {
    "stage E9.5-E8.75 (WT)": d("WT_E9.5", "WT_E8.75"),
    "stage E8.75-E8.5 (WT)": d("WT_E8.75", "WT_E8.5"),
    "Gata4_1 - WT E8.75": d("Gata4_1", "WT_E8.75"),
    "Gata4_2 - WT E8.75": d("Gata4_2", "WT_E8.75"),
    "Gata4 rep1 - rep2 (noise)": d("Gata4_1", "Gata4_2"),
    "Mab21l2 - WT E9.5": d("Mab21l2", "WT_E9.5"),
}
out = {"pseudobulk": {}}
for name, v in pairs.items():
    out["pseudobulk"][name] = {
        "L2": float(np.linalg.norm(v)),
        "frac_|lfc|>0.5": float((np.abs(v) > 0.5).mean()),
        "frac_|lfc|>1": float((np.abs(v) > 1).mean()),
        "top_up": genes[np.argsort(-v)[:8]].tolist(),
        "top_dn": genes[np.argsort(v)[:8]].tolist(),
    }
cor = lambda a, b: float(np.corrcoef(pairs[a], pairs[b])[0, 1])
out["correlations"] = {
    "Gata4_1 vs Gata4_2 effect": cor("Gata4_1 - WT E8.75", "Gata4_2 - WT E8.75"),
    "Gata4 effect vs stage E9.5-E8.75": cor("Gata4_1 - WT E8.75", "stage E9.5-E8.75 (WT)"),
    "Mab21l2 effect vs stage E9.5-E8.75": cor("Mab21l2 - WT E9.5", "stage E9.5-E8.75 (WT)"),
    "Mab21l2 effect vs Gata4 effect": cor("Mab21l2 - WT E9.5", "Gata4_1 - WT E8.75"),
}

# --- cell-level: how many KO cells sit in KO-only neighbourhoods? ---
def knn_mix(ko, wt, k=30):
    Z = np.vstack([X[ko], X[wt]])
    lab = np.r_[np.ones(len(X[ko])), np.zeros(len(X[wt]))]
    P = PCA(30, random_state=0).fit_transform(Z - Z.mean(0))
    idx = NearestNeighbors(n_neighbors=k + 1).fit(P).kneighbors(P, return_distance=False)[:, 1:]
    frac_ko = lab[idx].mean(1)  # expected ~ |KO|/(|KO|+|WT|) if well-mixed
    base = lab.mean()
    s = frac_ko[lab == 1]
    return {"expected_frac_if_mixed": float(base),
            "KO_cells_mean_KOfrac": float(s.mean()),
            "KO_cells_with_KOfrac>0.9": float((s > 0.9).mean()),
            "KO_cells_with_KOfrac<base+0.1": float((s < base + 0.1).mean())}

out["knn_mixing"] = {
    "Gata4_1 vs WT_E8.75": knn_mix("Gata4_1", "WT_E8.75"),
    "Gata4_2 vs WT_E8.75": knn_mix("Gata4_2", "WT_E8.75"),
    "Gata4_1 vs Gata4_2 (noise ref)": knn_mix("Gata4_1", "Gata4_2"),
    "Mab21l2 vs WT_E9.5": knn_mix("Mab21l2", "WT_E9.5"),
    "WT_E9.5 vs WT_E8.75 (stage ref)": knn_mix("WT_E9.5", "WT_E8.75"),
}
Path("private").mkdir(exist_ok=True)
Path("private/t3_explore.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
