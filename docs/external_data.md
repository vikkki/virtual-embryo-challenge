# External data catalog (T3 focus)

Every source used must be **disclosed with the submission**. Rules (FAQ/Rules, read 2026-11-04):
no measured data from a held-out stage or genotype "by any route" — directly, via a pretrained model,
or inside a public dataset. For T3 the held-out conditions are **Gata4 KO and β-catenin KO at E8.75**
(Gata4 is now usable via the official P3 validation release). Also treated as held-out:
"another allele of the same gene at a comparable stage" and "a perturbation that phenocopies them".

Status legend: ✅ allowed · ⚠️ ask organisers first · ❌ excluded

## A. Wild-type background (stage shift E8.5 → E9.5)

Purpose: model how the WT background moves with stage, so a KO effect learned at one stage
(Mab21l2 @ E9.5) can be transported to another (E8.75).

| Source | What | Stages | Access | Status |
|---|---|---|---|---|
| Imaz-Rosshandler et al. 2024, *Development* 151 (doi 10.1242/dev.201867) | scRNA atlas, ~300k cells, 6 h sampling | E8.5–E9.5 (+ E6.5–E8.5 → 430k cells) | raw only: E-MTAB-11763 (needs alignment); browser: marionilab.github.io/ExtendedMouseAtlas | ✅ (deferred: no processed matrix) for T3. If a model is shared with T2 heart, drop cells strictly between E8.25–E8.75 |
| Pijuan-Sala et al. 2019, *Nature* 566 | scRNA gastrulation atlas, 116k cells | E6.5–E8.5 | MouseGastrulationData (E-MTAB-6967) | ✅ (T2 embryo: drop strictly-between E7.25–E8.0) |
| Qiu et al. 2024, *Nature* (JAX time-lapse) | sci-RNA-seq3, 12.4M nuclei | E8 → P0 | GEO (see paper) | ✅ for T3 at ≤E9.5. ❌ E9.5–E13.5 for any model shared with T1 |
| Lohoff et al. 2022, *Nat Biotech* | seqFISH, 351 genes, 3 embryos | E8.5 | Marioni lab portal | ✅ spatial reference |
| de Soysa et al. 2019, *Nature* 572 | cardiogenic-region scRNA, WT | E7.75, E8.25, E9.25 | GEO GSE126128 | ✅ (heart-focused, matches T3 tissue) |

## B. Perturbation examples near stage (learning "what a KO looks like")

Purpose: more than one training perturbation, for priors on sparsity (which cells and genes respond,
and how strongly). The baselines page warns that copying a training KO's delta gets the direction wrong.
So these teach the *shape* of a response, not its content.

| Source | Perturbation | Stage | Access | Status |
|---|---|---|---|---|
| Pijuan-Sala et al. 2019 | *Tal1*⁻/⁻ chimeras (KO tdTom+ vs WT host) | E8.5 | E-MTAB-7325 / `Tal1ChimeraData` | ✅ in-embryo WT control, which is useful for separating background from effect |
| de Soysa et al. 2019 | *Hand2*-null cardiac progenitors | E7.75–E9.25 | GSE126128 | ✅ cardiac TF KO, same tissue and stage |
| Li et al. 2016, *Dev Cell* (ATLAS-seq) | *Nkx2-5*⁻/⁻ cardiomyocytes | E9.5 | GSE76118 (Fluidigm, small) | ✅ |
| Grosswendt et al. 2020, *Nature* | 10 epigenetic regulator zygotic KOs (PRC1/2, Kdm2b, L3mbtl2, Kmt2a, …) | E8.5 (WT E6.5–E8.5) | GSE137337 | ✅ includes delay phenotypes (e.g. L3mbtl2, PRC2) |
| Tet-TKO chimeras (PMC9432429; Argelaguet repo `10x_gastrulation_TetChimera`) | *Tet1/2/3* TKO, KO tdTom+ vs WT host | E7.5–E8.5 | GEO (see repo) | ⚠️ papers report marked Wnt-signalling perturbation, so it could be ruled a phenocopy |
| Guibentif et al. 2021, *Dev Cell* | *T* (Brachyury)⁻/⁻ chimeras | E8.5 | MouseGastrulationData | ⚠️ T is a direct Wnt/β-catenin target, and the Wnt3a/T posterior-truncation phenotypes overlap, so this could be ruled a phenocopy |
| Huang et al. 2023, *Nature* 623 | 22 mutants, whole embryo | E13.5 | see paper | ⚠️ stage is far from E8.75; ❌ in any model shared with T1 (E13.5 inclusive is excluded) |

## X. Excluded: β-catenin held-out or likely phenocopies

- Any *Ctnnb1* allele (null, conditional, floxed exon 3 / gain of function) in embryos at a comparable stage.
- Wnt ligand / secretion / receptor loss: *Wls* (e.g. *Mesp1-Cre;Wls* cKO, GSE123765 / GSE123771),
  *Porcn*, *Wnt3*, *Wnt3a*, *Lrp5/6*, *Lef1/Tcf1* compound mutants; Wnt-inhibitor-treated embryos or
  gastruloids.
- Pretrained models whose training data include any of the above. Check before use. Navigo's preprint
  validates on six KO genotypes (not named in the abstract); confirm they are not in its training set.

## C. Prior knowledge (no KO measurements)

| Resource | Use | Status |
|---|---|---|
| GenePT / scGPT gene embeddings | gene-similarity features for the target gene and the readout genes | ✅ check training corpora |
| CollecTRI / DoRothEA / OmniPath / STRING | target → gene regulatory priors | ✅ |
| Literature Wnt/TCF target gene sets, TCF/LEF ChIP in WT tissue | expected response genes for β-catenin | ⚠️ not KO data, but closest to the line; ask organisers |

## Questions for the organisers (send before using ⚠️ items)

1. Is *T*/Brachyury-null data considered a phenocopy of the β-catenin KO? *Open: the site gives no list of
   phenocopies.*
2. Are curated Wnt target gene lists / TCF-LEF ChIP-seq (WT tissue, no KO) permitted for T3? *Open: the rules
   ban measured data from the held-out genotype, another allele "at a comparable stage", or a phenocopy, and
   say nothing about literature-curated target lists. The published `gene_ko` baseline only zeroes the KO
   gene, so it is no precedent.*
3. ~~Which β-catenin allele and Cre driver is the test KO?~~ **Answered on the data page
   (virtualembryo.ai/challenge/data, read 2026-11-07):** test = *Mesp1-Cre; β-catenin F/F* (two replicates);
   validation = *Mesp1-Cre; Gata4 F/F; Gata6 F/+*. Both deletions are restricted to the Mesp1 mesoderm
   lineage. Consequence: any published *Mesp1-Cre; Ctnnb1* (or other *Ctnnb1* allele) result at a comparable
   stage is the held-out genotype and must not be used, including results quoted in papers.

Ask in public on Slack (#virtual-embryo-challenge) or by email (virtual.embryo.moonshot@gmail.com). The rules
say an answer given in advance is not a finding of misconduct afterwards.

## Downloaded (scripts/download_external.sh → data/external/, git-ignored)

| Folder | Source | Files |
|---|---|---|
| `tal1_chimera_pijuansala2019/` | ArrayExpress E-MTAB-7325 | `chimera_tal1_data.tar.gz`, sdrf |
| `atlas_pijuansala2019/` | ArrayExpress E-MTAB-6967 | `atlas_data.tar.gz`, sdrf |
| `hand2_desoysa2019/` | GEO GSE126128 | WT all timepoints; WT+KO E7.75/E8.25; WT+KO E9.25 (dense CSV); series matrix |
| `nkx25_li2016/` | GEO GSE76118 | `RAW.tar` (HTSeq counts per cell); series matrix (genotype per cell) |
| `ko_grosswendt2020/` | GEO GSE137337 | E8.5 scRNA for 11 KOs (bigwig methylation RAW.tar skipped); series matrices |

Checksums: `data/external/SHA256SUMS` (made after download).
