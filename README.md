# Virtual Embryo Challenge (NeurIPS 2026)

Workspace for the [Virtual Embryo Challenge](https://virtualembryo.ai/challenge): predict mouse embryo
cell-state distributions, 3D tissue structure, and knockout effects.

**Phase:** P3 (final test). **Final deadline: 2026-12-02** (exact time/timezone not stated — confirm with organisers).

## Boards (P3 test)

| Board key | Task | Genes | Cells | Coords |
|---|---|---:|---|---|
| `T1:test` | E8.5, E9.5 scRNA → E12.5 | 32,285 | 638–5,118 | – |
| `T2:heart:test_extrap` | E8.25/E8.75/E9.5 spatial → E12.5 | 500 | 1,000–34,641 | `spatial_3D` |
| `T2:embryo:test_interp` | E6.75/E7.25/E8.0 → E7.75 | **497** | 1,000–5,000 | `spatial_3D` |
| `T3:bcatenin` | Mab21l2 KO + WT → β-catenin KO (E8.75) | 500 | 1,000–7,449 | `spatial_3D` |

Specs and gene lists live in [`panels/`](panels/) (from `virtualembryo.ai/challenge/panels/`);
refresh with `scripts/update_panels.sh`.

## Setup

```bash
conda create -n vec python=3.11 -y && conda activate vec
pip install -e ".[dev]"
cp .env.example .env   # add VEC_API_KEY
```

Download training data from the account page into `data/` (git-ignored), keeping the official layout
(`data/T1/E8.5_RNA.h5ad`, `data/T2/...`, `data/T3/...`) and verify with `shasum -a 256`.

## Workflow

1. Build a prediction → `submissions/<board>/<name>.h5ad` (git-ignored).
2. Format check (free, unlimited): `python scripts/validate_submission.py pred.h5ad T1:test`
3. Local metric sanity check with [`veckit`](https://github.com/aristoteleo/veckit) (pinned 0.1.2) —
   not a preview of the real score.
4. Record every formal submission in [`docs/submission_log.md`](docs/submission_log.md).

## Rules that bite — read before submitting

- **P2 → P3 is one-way, team-wide**: fetching validation answers *or* the first accepted P3 submission
  permanently ends P2 (live validation) for all tasks.
- **2 formal submissions per test board, for the whole phase.** Format checks don't count.
- `.X` must be log-normalised, finite, non-negative. No raw counts, no centred matrices.
- `var_names` must match the board's list exactly, in order (T2 embryo test = 497, not val's 498).
- File ≤ 1200 MB. Cell-type labels are ignored.
- External data/pretrained models allowed only if disclosed and they don't touch protected stages
  (e.g. T1/T2 extrapolation: nothing from after E9.5 through E13.5 inclusive). Check foundation-model
  training data.
- Publishing results using challenge data needs prior approval from the data team.
- Agent track: once a run is locked, no human steering; keep prompts + trajectory + harness.

## Layout

```
src/vec/      package: panels.py (board specs), validate.py (format check)
scripts/      CLI entry points
panels/       official board specs + gene lists (committed)
configs/      experiment configs
notebooks/    exploration
docs/         notes, submission log
data/         challenge data (ignored)
submissions/  prediction files (ignored)
```
