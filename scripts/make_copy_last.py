#!/usr/bin/env python
"""Copy-last baseline: take the last observed stage as the prediction (official 50-point floor).

  python scripts/make_copy_last.py --board T1:test --source data/T1/E9.5_RNA.h5ad \
      --out submissions/T1_test/copy_last.h5ad

Subsamples to the board's max_cells, reorders genes to the board panel (missing genes -> 0),
log-normalises if .X looks like raw counts, and carries spatial coords into obsm["spatial_3D"].
"""
import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from vec.panels import board, genes


def looks_raw(X) -> bool:
    v = X.data[:100000] if sp.issparse(X) else np.asarray(X).ravel()[:100000]
    return v.size > 0 and np.allclose(v, np.round(v)) and v.max() > 20


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--board", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--coords-key", default=None, help="obsm key holding 3D coords in the source")
    p.add_argument("--layer", default=None, help="use this layer instead of .X")
    p.add_argument("--n-cells", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    spec = board(a.board)
    panel = list(genes(a.board))
    src = ad.read_h5ad(a.source)
    print(f"source {src.shape}; layers={list(src.layers)} obsm={list(src.obsm)}")

    n = min(a.n_cells or spec["max_cells"], spec["max_cells"], src.n_obs)
    if n < spec["min_cells"]:
        raise SystemExit(f"only {n} cells available, board needs >= {spec['min_cells']}")
    idx = np.sort(np.random.default_rng(a.seed).choice(src.n_obs, n, replace=False))
    src = src[idx].copy()

    X = src.layers[a.layer] if a.layer else src.X
    X = sp.csr_matrix(X, dtype=np.float32)
    if looks_raw(X):
        print("X looks like raw counts -> normalize_total(1e4) + log1p")
        lib = np.asarray(X.sum(1)).ravel()
        lib[lib == 0] = 1
        X = sp.csr_matrix(sp.diags(1e4 / lib) @ X)
        X.data = np.log1p(X.data)

    pos = pd.Index(src.var_names).get_indexer(panel)
    have = pos >= 0
    print(f"panel {len(panel)} genes; {have.sum()} in source, {(~have).sum()} filled with 0")
    out = sp.lil_matrix((n, len(panel)), dtype=np.float32)
    Xc = X.tocsc()
    out[:, np.where(have)[0]] = Xc[:, pos[have]]
    out = out.tocsr()

    pred = ad.AnnData(out, obs=pd.DataFrame(index=[f"cell{i}" for i in range(n)]),
                      var=pd.DataFrame(index=panel))
    if spec.get("obsm_required"):
        key = a.coords_key or next((k for k in ("spatial_3D", "spatial", "X_spatial") if k in src.obsm), None)
        if key is None:
            raise SystemExit(f"no coords found; obsm has {list(src.obsm)} — pass --coords-key")
        c = np.asarray(src.obsm[key], dtype=np.float32)
        if c.shape[1] != 3:
            raise SystemExit(f"obsm[{key}] has shape {c.shape}, need 3 columns")
        pred.obsm["spatial_3D"] = c

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pred.write_h5ad(a.out, compression="gzip")
    print(f"wrote {a.out} {pred.shape} ({Path(a.out).stat().st_size / 2**20:.1f} MB)")


if __name__ == "__main__":
    main()
