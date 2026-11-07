# T3 approach (β-catenin KO at E8.75)

Method notes only. Data-derived numbers stay in the git-ignored `private/` folder.

## Pipeline

1. **States.** WT E8.75 `celltype` labels are the shared vocabulary. KO cells are mapped onto them by kNN
   label transfer in a PCA space fitted on the WT. Each KO is mapped to its own stage-matched WT
   (`vec.t3.transfer_states`).
2. **Labels.** For each training KO, `vec.t3.decompose` splits pb(KO) − pb(WT) into a within-state shift
   (μ_s^KO − μ_s^WT, weighted by WT fractions) and a composition shift. Each label is measured against the
   stage-matched WT, so the developmental stage shift never enters it.
3. **Model.** `vec.t3model` is a prior-only spike-and-slab model inspired by cellworld v3, without its
   "sources" branch. The rules forbid same-gene evidence for β-catenin.
   - The target encoder sees only prior features of the KO target.
   - The gene decoder sees per-(state, gene) WT features and pair priors (CollecTRI).
   - Its FiLM blocks are conditioned on the target latent.
   - Each (state, gene) gets an output (π, m, τ), and the prediction is ŷ = (1 − π)·m.
   - Training options for mixed external data (`vec.t3model.fit`):
     - `study_weights`: every study gets the same total weight, so a large series of KOs doesn't dominate.
     - `nuisance_groups`: KOs of different genes measured against one shared control get a learned
       (state, gene) offset that is subtracted from their labels while training and dropped at prediction
       time. KOs against a shared control move together regardless of the gene, and this term soaks that up
       so the network learns the gene-specific part. Replicates of one gene get no offset.
4. **Emission.** `vec.t3.emit` draws WT E8.75 cells by state and keeps their 3D coordinates. It shifts each
   state's genes so the state mean moves by the predicted delta (zeros stay zero for decreases, and
   non-expressing cells are recruited for large increases). With `exact_wt`, it also removes the WT
   subsample's own sampling noise, so a zero delta reproduces the WT pseudobulk exactly.
5. **Local scoring.** `vec.t3.Scorer` wraps veckit's T3 metrics (`de_score`, `de_direction`,
   `severity_slope`, `mmd_u`) and caches the truth's DE genes. `vec.score.approx_score` maps the metrics
   onto the board anchors.

## Scripts

| Script | Purpose |
|---|---|
| `analysis/t3_baselines.py` | decomposition, emission oracles, cross-replicate ceiling, simple transfers |
| `analysis/t3_effect_structure.py` | state-specific vs gene-wise shared parts of a KO shift |
| `analysis/t3_rules.py` | target-agnostic zero-shot rules |
| `analysis/t3_prior_check.py` | regulon-only (CollecTRI) predictions vs a random-regulon null |
| `analysis/t3_background_check.py` | how the metrics treat a generic "highly expressed genes decrease" background |
| `scripts/t3_loto.py` | held-out check of the model: `--split gene` (leave one KO gene out) or `--split study` (leave one study out, plus its genes); `--balance study`, `--nuisance` |
| `analysis/t3_shared_control.py`, `analysis/t3_copy_one.py` | how much KOs that share one control move together |
| `analysis/t3_reference_bump.py` | whether a KO's sample-wide shift is an artefact of its WT reference |
| `scripts/make_t3_pred.py` | build and validate a T3 prediction file (never uploads) |

## Metric notes (from veckit 0.1.2 source)

- `de_score` ranks **all** genes by the predicted pseudobulk change and compares the result against the
  better of two biology-free rankings: ±reference expression. A prediction that moves only a few genes and
  ties the rest at zero can therefore score below the do-nothing floor.
- `de_direction` is a rank-partial correlation with reference expression partialled out.
- `severity_slope` is log of the OLS slope over the truth's DE genes. A prediction that doesn't track the
  truth (r² ≤ 0.01 or slope ≤ 0) gets the floor, log(1e-3).
- `mmd_u` is an unbiased multi-kernel MMD in a PCA fitted on the truth.

## External inputs

- CollecTRI (mouse, via OmniPath) → `data/priors/collectri_mouse.csv.gz` (git-ignored).
- The Ctnnb1 regulon is used only with `--ctnnb1-regulon`. It is effectively a curated Wnt target list, an
  "ask the organisers first" item in [external_data.md](external_data.md).

## Where things run

Training runs on a remote GPU box, not on the laptop. `vec.t3model.fit` refuses to train without CUDA
unless `VEC_ALLOW_CPU_TRAIN=1`. `scripts/remote.sh` reads the host from the git-ignored `.env`:

```bash
scripts/remote.sh sync                                   # push code
scripts/remote.sh data                                   # push the T3 data subset (once)
scripts/remote.sh bg loto python scripts/t3_loto.py      # train + evaluate, detached
scripts/remote.sh run tail -n 30 logs/loto.log           # follow the log
scripts/remote.sh fetch runs/t3_loto.json                # pull results back
```

Remote env: a virtualenv that reuses the container's CUDA PyTorch (`python -m venv --system-site-packages`)
plus anndata, scipy, pandas, scikit-learn, h5py and veckit 0.1.2. The local laptop env is used only for
light inspection and scoring.
