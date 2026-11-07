# Formal submission log

Each test board allows **2** formal submissions in total. Log every one here *before* uploading.

| Date | Board | File / commit | Method | Local checks | Official score | Notes |
|---|---|---|---|---|---|---|

## Candidates (built and format-checked, not uploaded)

| Date | Board | File | Method | Notes |
|---|---|---|---|---|
| 2026-11-07 | T3:bcatenin | `submissions/T3_bcatenin/wt_identity.h5ad` | WT E8.75 cells, sampling noise removed | the floor; fallback |
| 2026-11-07 | T3:bcatenin | `submissions/T3_bcatenin/gata4_offset_v1.h5ad` | shared Mesp1-Cre KO signature: median Gata4 within-state shift over states outside the Mesp1 lineage (official Gata4 validation release), added to every WT E8.75 state | `scripts/make_t3_pred.py --model gata4_offset`; no external data; untestable locally for β-catenin |
