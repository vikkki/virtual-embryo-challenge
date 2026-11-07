"""Turn external KO datasets into (WT, KO) pairs on the T3 500-gene panel.

Each pair is written to data/processed/<name>_WT.h5ad and <name>_KO.h5ad and registered in
data/processed/index.json. Both files hold:

  .X            log1p(normalize_total 1e4) computed over the PANEL genes only (mimics the MERFISH files)
  .var.present  whether the dataset measured that panel gene (absent genes are all zero and must be masked)
  .obs.state    cell state: the dataset's own annotation where it has one, else KMeans clusters of the WT
                with KO cells mapped onto them by kNN
  .obs.target_expr  the KO target's own log-expression over ALL genes (WT file only), for target activity

Heavy: run on the remote box (scripts/remote.sh), not on the laptop. Archives are untrusted input: they are
unpacked into data/external/_extract/<dataset>/ with tarfile's "data" filter.
"""
from __future__ import annotations

import csv
import gzip
import json
import tarfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

from .panels import genes as panel_genes
from .t3 import ROOT

EXT = ROOT / "data/external"
WORK = EXT / "_extract"
OUT = ROOT / "data/processed"
PANEL = list(panel_genes("T3:bcatenin"))
PIDX = {g: i for i, g in enumerate(PANEL)}
PIDX_CI = {g.lower(): i for i, g in enumerate(PANEL)}


# ------------------------------------------------------------------------------------------------ helpers
def extract(archive: Path, dest: Path) -> Path:
    if not (dest / ".done").exists():
        dest.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive) as t:
            t.extractall(dest, filter="data")
        (dest / ".done").touch()
    return dest


OLD_NAMES = {"Wisp2": "Ccn5", "Tmem2": "Cemip2"}    # older annotations (e.g. Ensembl 92 / CellRanger mm10)


def panel_col(symbol: str) -> int:
    symbol = OLD_NAMES.get(symbol, symbol)
    return PIDX.get(symbol, PIDX_CI.get(str(symbol).lower(), -1))


def ensembl_to_symbol() -> dict[str, str]:
    """Ensembl -> symbol from a 10x features file (Grosswendt, plain .tsv.gz) or else the Pijuan-Sala genes.tsv."""
    f = EXT / "ko_grosswendt2020/GSE137337_DNMT1_E8.5_1ab.features.tsv.gz"
    if f.exists():
        g = pd.read_csv(f, sep="\t", header=None)
    else:
        d = extract(EXT / "tal1_chimera_pijuansala2019/chimera_tal1_data.tar.gz", WORK / "tal1")
        g = pd.read_csv(d / "chimera-tal1/genes.tsv", sep="\t", header=None)
    return dict(zip(g[0], g[1]))


def read_mtx_panel(path: Path, row_to_col: np.ndarray, chunksize: int = 20_000_000):
    """Stream a MatrixMarket genes x cells file. Returns (cells x 500 CSR of the panel rows, per-cell totals
    over ALL genes). row_to_col[i] = panel column of gene row i (1-based), or -1."""
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as fh:
        line = fh.readline()
        while line.startswith("%"):
            line = fh.readline()
        nrow, ncol, _ = map(int, line.split())
        tot = np.zeros(ncol + 1)
        R, C, V = [], [], []
        for ch in pd.read_csv(fh, sep=" ", header=None, names=["i", "j", "v"], chunksize=chunksize,
                              dtype={"i": np.int32, "j": np.int32, "v": np.float32}):
            i, j, v = ch.i.to_numpy(), ch.j.to_numpy(), ch.v.to_numpy()
            tot += np.bincount(j, weights=v, minlength=ncol + 1)
            col = row_to_col[i]
            m = col >= 0
            R.append(j[m] - 1); C.append(col[m]); V.append(v[m])
    M = sp.csr_matrix((np.concatenate(V), (np.concatenate(R), np.concatenate(C))), shape=(ncol, len(PANEL)))
    return M, tot[1:]


def lognorm(M) -> np.ndarray:
    M = sp.csr_matrix(M, dtype=np.float32)
    s = np.asarray(M.sum(1)).ravel()
    s[s == 0] = 1
    X = sp.diags(1e4 / s) @ M
    return np.log1p(X.toarray()).astype(np.float32)


def kmeans_states(wt_X: np.ndarray, ko_X: np.ndarray, k: int, seed: int = 0):
    pca = PCA(min(30, wt_X.shape[1] - 1), random_state=seed).fit(wt_X)
    Zw, Zk = pca.transform(wt_X), pca.transform(ko_X)
    lab = KMeans(k, n_init=10, random_state=seed).fit_predict(Zw)
    wt_s = np.array([f"c{c:02d}" for c in lab])
    ko_s = KNeighborsClassifier(15, n_jobs=-1).fit(Zw, wt_s).predict(Zk)
    return wt_s, ko_s


def save_pair(name: str, *, target: str, stage: str, platform: str, source: str, control: str,
              wt_X, wt_state, ko_X, ko_state, present, wt_target_expr=None, min_cells: int = 30) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    var = pd.DataFrame({"present": present}, index=PANEL)
    for suffix, X, st in [("WT", wt_X, wt_state), ("KO", ko_X, ko_state)]:
        obs = pd.DataFrame({"state": pd.Categorical(st), "celltype": pd.Categorical(st)},
                           index=[f"{suffix}{i}" for i in range(X.shape[0])])
        if suffix == "WT" and wt_target_expr is not None:
            obs["target_expr"] = np.asarray(wt_target_expr, np.float32)
        a = ad.AnnData(sp.csr_matrix(X), obs=obs, var=var)
        a.obsm["spatial_3D"] = np.zeros((X.shape[0], 3), np.float32)     # no coordinates in scRNA
        a.write_h5ad(OUT / f"{name}_{suffix}.h5ad", compression="gzip")
    idx_path = OUT / "index.json"
    idx = json.loads(idx_path.read_text()) if idx_path.exists() else {}
    vc_w, vc_k = pd.Series(wt_state).value_counts(), pd.Series(ko_state).value_counts()
    idx[name] = {"target": target, "stage": stage, "platform": platform, "source": source, "control": control,
                 "wt": f"{name}_WT", "ko": f"{name}_KO", "n_wt": int(wt_X.shape[0]), "n_ko": int(ko_X.shape[0]),
                 "genes_present": int(np.sum(present)),
                 "states_with_min_cells": int(((vc_w >= min_cells) & (vc_k.reindex(vc_w.index).fillna(0) >= min_cells)).sum())}
    idx_path.write_text(json.dumps(idx, indent=1, sort_keys=True))
    print(f"[{name}] WT {wt_X.shape[0]} / KO {ko_X.shape[0]} cells, {int(np.sum(present))}/500 genes, "
          f"{idx[name]['states_with_min_cells']} usable states")
    return idx[name]


# ----------------------------------------------------------------------------------------------- datasets
def tal1_chimera(stage: str = "E8.5"):
    """Pijuan-Sala 2019 Tal1-/- chimeras: tdTomato+ = injected Tal1-/- cells, tdTomato- = WT host cells of the
    same embryos. States = the authors' atlas mapping (celltype.mapped) for both."""
    d = extract(EXT / "tal1_chimera_pijuansala2019/chimera_tal1_data.tar.gz", WORK / "tal1") / "chimera-tal1"
    g = pd.read_csv(d / "genes.tsv", sep="\t", header=None)
    meta = pd.read_csv(d / "meta.tab", sep="\t")
    row_to_col = np.full(len(g) + 1, -1)
    for i, s in enumerate(g[1], start=1):
        row_to_col[i] = panel_col(s)
    M, tot = read_mtx_panel(d / "raw_counts.mtx", row_to_col)
    present = np.zeros(len(PANEL), bool); present[row_to_col[row_to_col >= 0]] = True
    keep = (meta.stage.to_numpy() == stage) & meta["celltype.mapped"].notna().to_numpy() & (tot > 0)
    tom = meta.tomato.astype(str).str.upper().eq("TRUE").to_numpy()
    X = lognorm(M)
    texpr = None
    if "Tal1" in PIDX:          # Tal1 is on the panel: its all-gene log-expression from panel counts / total
        texpr = np.log1p(M[:, PIDX["Tal1"]].toarray().ravel() / np.maximum(tot, 1) * 1e4)
    st = meta["celltype.mapped"].astype(str).to_numpy()
    w, k = keep & ~tom, keep & tom
    return save_pair(f"tal1_chimera_{stage}", target="Tal1", stage=stage, platform="10x",
                     source="E-MTAB-7325", control="WT host cells in the same chimeric embryos",
                     wt_X=X[w], wt_state=st[w], ko_X=X[k], ko_state=st[k], present=present,
                     wt_target_expr=None if texpr is None else texpr[w])


HAND2_GROUPS = {   # file -> stage -> (WT gem groups, KO gem groups), from the GEO series matrices
    "GSE126128_E775E825_WTKO_10X.csv.gz": {"E7.75": ([1, 2, 3, 4, 5], [6, 7, 8]), "E8.25": ([9, 10], [11, 12])},
    "GSE126128_E925_WTKO_10X.csv.gz": {"E9.25": ([1, 2], [3, 4])},
}


def read_dense_csv(path: Path, extra: tuple[str, ...] = ()):
    """genes x cells CSV of log-normalised values -> (barcodes, cells x 500 pseudo-counts, present, extras)."""
    with gzip.open(path, "rt") as fh:
        bc = next(csv.reader([fh.readline()]))[1:]
        M = np.zeros((len(bc), len(PANEL)), np.float32)
        present = np.zeros(len(PANEL), bool)
        ex = {}
        for line in fh:
            k = line.find(",")
            gname = line[:k].strip('"')
            c = panel_col(gname)
            if c < 0 and gname not in extra:
                continue
            v = np.expm1(np.array(line[k + 1:].split(","), dtype=np.float32))   # back to normalised counts
            if c >= 0:
                M[:, c] = v; present[c] = True
            if gname in extra:
                ex[gname] = v
    return np.array(bc), M, present, ex


def hand2_null():
    out = []
    for fname, stages in HAND2_GROUPS.items():
        bc, M, present, ex = read_dense_csv(EXT / "hand2_desoysa2019" / fname, extra=("Hand2",))
        grp = np.array([int(b.rsplit("-", 1)[1]) for b in bc])
        keep = (M > 0).sum(1) >= 20
        X = lognorm(M)
        for stage, (wg, kg) in stages.items():
            w, k = keep & np.isin(grp, wg), keep & np.isin(grp, kg)
            ws, ks = kmeans_states(X[w], X[k], k=12)
            out.append(save_pair(f"hand2_null_{stage}", target="Hand2", stage=stage, platform="10x",
                                 source="GSE126128", control="littermate WT, same study",
                                 wt_X=X[w], wt_state=ws, ko_X=X[k], ko_state=ks, present=present,
                                 wt_target_expr=X[w][:, PIDX["Hand2"]] if "Hand2" in PIDX else None))
    return out


def nkx25_null():
    """Li 2016 ATLAS-seq (Fluidigm C1): E9.5 Nkx2-5 WT vs mutant cardiomyocytes. One state ("CM")."""
    sm = EXT / "nkx25_li2016/GSE76118_series_matrix.txt.gz"
    titles, gsm = [], []
    with gzip.open(sm, "rt") as fh:
        for line in fh:
            if line.startswith("!Sample_title"):
                titles = [t.strip('"') for t in line.rstrip("\n").split("\t")[1:]]
            elif line.startswith("!Sample_geo_accession"):
                gsm = [t.strip('"') for t in line.rstrip("\n").split("\t")[1:]]
    geno = {a: ("WT" if "Nkx2.5wt" in t else "KO") for a, t in zip(gsm, titles) if "Nkx2.5wt" in t or "Nkx2.5mut" in t}
    e2s = ensembl_to_symbol()
    rows, labels, tot, nkx = [], [], [], []
    nkx_id = next((e for e, s in e2s.items() if s == "Nkx2-5"), None)
    with tarfile.open(EXT / "nkx25_li2016/GSE76118_RAW.tar") as t:
        for m in t.getmembers():
            a = m.name.split("_")[0]
            if a not in geno:
                continue
            v = np.zeros(len(PANEL), np.float32)
            total, nk = 0.0, 0.0
            with gzip.open(t.extractfile(m), "rt") as fh:
                for line in fh:
                    e, c = line.rstrip("\n").split("\t")
                    if e.startswith("__"):
                        continue
                    c = float(c); total += c
                    if e == nkx_id:
                        nk = c
                    col = panel_col(e2s.get(e, ""))
                    if col >= 0:
                        v[col] += c
            rows.append(v); labels.append(geno[a]); tot.append(total); nkx.append(nk)
    M = np.vstack(rows); labels = np.array(labels); tot = np.array(tot)
    present = np.zeros(len(PANEL), bool)
    for e, s in e2s.items():
        if panel_col(s) >= 0:
            present[panel_col(s)] = True
    keep = (M > 0).sum(1) >= 20
    X = lognorm(M)
    texpr = np.log1p(np.array(nkx) / np.maximum(tot, 1) * 1e4)
    w, k = keep & (labels == "WT"), keep & (labels == "KO")
    return save_pair("nkx25_null_E9.5", target="Nkx2-5", stage="E9.5", platform="fluidigm-c1",
                     source="GSE76118", control="WT cardiomyocytes, same study",
                     wt_X=X[w], wt_state=np.full(w.sum(), "CM"), ko_X=X[k], ko_state=np.full(k.sum(), "CM"),
                     present=present, wt_target_expr=texpr[w], min_cells=20)


GROSSWENDT = {  # sample -> (target gene symbol, how stored)
    "CDKN2A": ("Cdkn2a", "tar"), "DNMT3B": ("Dnmt3b", "tar"), "EED": ("Eed", "tar"), "G9A": ("Ehmt2", "tar"),
    "KDM2B": ("Kdm2b", "tar"), "KMT2A": ("Kmt2a", "tar"), "KMT2B": ("Kmt2b", "tar"), "RNF2": ("Rnf2", "tar"),
    "L3MBTL2": ("L3mbtl2", "tar1"), "DNMT1": ("Dnmt1", "mtx"), "DNMT3A": ("Dnmt3a", "mtx"),
    # EED.CDKN2A is a double KO; skipped (its control would be the Cdkn2a sample itself)
}


def _grosswendt_matrix(sample: str, min_umi: int):
    kind = GROSSWENDT[sample][1]
    base = EXT / "ko_grosswendt2020"
    if kind == "mtx":
        files = {p: base / f"GSE137337_{sample}_E8.5_1ab.{p}" for p in ("barcodes.tsv.gz", "features.tsv.gz", "matrix.mtx.gz")}
    else:
        tgz = base / (f"GSE137337_{sample}_E8.5_1.tar.gz" if kind == "tar1" else f"GSE137337_{sample}_E8.5_1ab.tar.gz")
        d = extract(tgz, WORK / f"grosswendt_{sample}")
        files = {p: next(d.rglob(p)) for p in ("barcodes.tsv.gz", "features.tsv.gz", "matrix.mtx.gz")}
    feat = pd.read_csv(files["features.tsv.gz"], sep="\t", header=None)
    row_to_col = np.full(len(feat) + 1, -1)
    for i, s in enumerate(feat[1], start=1):
        row_to_col[i] = panel_col(s)
    M, tot = read_mtx_panel(files["matrix.mtx.gz"], row_to_col)
    present = np.zeros(len(PANEL), bool); present[row_to_col[row_to_col >= 0]] = True
    keep = tot >= min_umi
    return M[keep], tot[keep], present, feat, files, keep


def _gene_vector(files, feat, keep, symbol: str, tot) -> np.ndarray | None:
    """Log-expression of one (off-panel) gene per kept cell, from a second pass over the matrix."""
    hit = np.where(feat[1].to_numpy() == symbol)[0]
    if not len(hit):
        return None
    r2c = np.full(len(feat) + 1, -1); r2c[hit[0] + 1] = 0
    M, _ = read_mtx_panel(files["matrix.mtx.gz"], r2c)
    return np.log1p(M[:, 0].toarray().ravel()[keep] / np.maximum(tot, 1) * 1e4)


def grosswendt(samples=None, min_umi: int = 2000, k: int = 20):
    """Grosswendt 2020 zygotic KOs at E8.5. The series has no WT scRNA, so the CDKN2A KO (same lab, protocol and
    strain; Cdkn2a is barely expressed at E8.5) is the control for every other KO."""
    Mc, totc, pres_c, featc, filesc, keepc = _grosswendt_matrix("CDKN2A", min_umi)
    Xc = lognorm(Mc)
    out = []
    for s in samples or [x for x in GROSSWENDT if x != "CDKN2A"]:
        tgt = GROSSWENDT[s][0]
        M, tot, pres, feat, files, keep = _grosswendt_matrix(s, min_umi)
        X = lognorm(M)
        ws, ks = kmeans_states(Xc, X, k=k)
        texpr = _gene_vector(filesc, featc, keepc, tgt, totc)
        out.append(save_pair(f"grosswendt_{s.lower()}_E8.5", target=tgt, stage="E8.5", platform="10x",
                             source="GSE137337", control="Cdkn2a KO embryos (same lab/protocol), used as WT proxy",
                             wt_X=Xc, wt_state=ws, ko_X=X, ko_state=ks, present=pres_c & pres,
                             wt_target_expr=texpr))
    return out


DATASETS = {"tal1": tal1_chimera, "hand2": hand2_null, "nkx25": nkx25_null, "grosswendt": grosswendt}
