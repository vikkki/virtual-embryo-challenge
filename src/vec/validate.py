"""Offline format check for a prediction .h5ad against an official board spec.

Mirrors the published submission rules: log-normalised, finite, non-negative .X;
var_names exactly equal (and in order) to the board's gene list; obsm["spatial_3D"]
(n_cells x 3) where required; cell count within limits; file <= 1200 MB.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.sparse as sp

from .panels import board, genes

MAX_BYTES = 1200 * 1024 * 1024


def validate(path: str | Path, key: str) -> list[str]:
    import anndata as ad

    path = Path(path)
    spec = board(key)
    errors: list[str] = []

    if path.stat().st_size > MAX_BYTES:
        errors.append(f"file is {path.stat().st_size / 2**20:.0f} MB > 1200 MB")

    a = ad.read_h5ad(path)
    n = a.n_obs
    if not spec["min_cells"] <= n <= spec["max_cells"]:
        errors.append(f"n_cells={n} outside [{spec['min_cells']}, {spec['max_cells']}]")

    want = genes(key)
    got = tuple(a.var_names)
    if got != want:
        missing = len(set(want) - set(got))
        extra = len(set(got) - set(want))
        errors.append(
            f"var_names mismatch: have {len(got)}, need {len(want)} "
            f"(missing {missing}, extra {extra}, order differs={missing == extra == 0})"
        )

    X = a.X.data if sp.issparse(a.X) else np.asarray(a.X)
    if X.size:
        if not np.isfinite(X).all():
            errors.append(".X has non-finite values")
        if (X < 0).any():
            errors.append(".X has negative values (centred/scaled? submit log-normalised)")
        if np.allclose(X, np.round(X)) and X.max() > 20:
            errors.append(".X looks like raw counts (integers, max > 20); apply log normalisation")

    for k in spec.get("obsm_required", []):
        if k not in a.obsm:
            errors.append(f'missing obsm["{k}"]')
        else:
            c = np.asarray(a.obsm[k])
            if c.shape != (n, 3):
                errors.append(f'obsm["{k}"] shape {c.shape}, expected ({n}, 3)')
            elif not np.isfinite(c).all():
                errors.append(f'obsm["{k}"] has non-finite values')
    return errors
