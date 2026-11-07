#!/usr/bin/env bash
# Download allowed external datasets (see docs/external_data.md) into data/external/ (git-ignored).
#   bash scripts/download_external.sh                       # everything
#   bash scripts/download_external.sh hand2 nkx25 grosswendt # only these (tal1 atlas hand2 nkx25 grosswendt)
set -uo pipefail
WANT="${*:-tal1 atlas hand2 nkx25 grosswendt}"
want() { [[ " $WANT " == *" $1 "* ]]; }
cd "$(dirname "$0")/../data" && mkdir -p external && cd external
GEO=https://ftp.ncbi.nlm.nih.gov/geo/series
AE=https://ftp.ebi.ac.uk/biostudies/fire/E-MTAB-
get() { mkdir -p "$1"; (cd "$1" && curl -sSfL -C - --retry 5 -O "$2" && echo "ok  $1/$(basename "$2")" || echo "ERR $1/$(basename "$2")"); }

# Tal1-/- chimeras E8.5 (Pijuan-Sala 2019), E-MTAB-7325
want tal1 && for f in chimera_tal1_data.tar.gz E-MTAB-7325.sdrf.txt; do get tal1_chimera_pijuansala2019 "$AE/325/E-MTAB-7325/Files/$f"; done
# WT gastrulation atlas E6.5-E8.5 (Pijuan-Sala 2019), E-MTAB-6967
want atlas && for f in atlas_data.tar.gz E-MTAB-6967.sdrf.txt; do get atlas_pijuansala2019 "$AE/967/E-MTAB-6967/Files/$f"; done
# Hand2-null + WT cardiac progenitors (de Soysa 2019), GSE126128
want hand2 && for f in GSE126128_AllTimePoints_WT_10X.csv.gz GSE126128_E775E825_WTKO_10X.csv.gz GSE126128_E925_WTKO_10X.csv.gz; do get hand2_desoysa2019 "$GEO/GSE126nnn/GSE126128/suppl/$f"; done
want hand2 && for p in GPL19057 GPL21103; do get hand2_desoysa2019 "$GEO/GSE126nnn/GSE126128/matrix/GSE126128-${p}_series_matrix.txt.gz"; done
# Nkx2-5-/- E9.5 cardiomyocytes, ATLAS-seq (Li 2016), GSE76118
want nkx25 && get nkx25_li2016 "$GEO/GSE76nnn/GSE76118/suppl/GSE76118_RAW.tar"
want nkx25 && get nkx25_li2016 "$GEO/GSE76nnn/GSE76118/matrix/GSE76118_series_matrix.txt.gz"
# Epigenetic-regulator zygotic KOs at E8.5 (Grosswendt 2020), GSE137337 scRNA only (skip bigwig RAW.tar)
want grosswendt && for k in CDKN2A DNMT3B EED.CDKN2A EED G9A KDM2B KMT2A KMT2B RNF2; do get ko_grosswendt2020 "$GEO/GSE137nnn/GSE137337/suppl/GSE137337_${k}_E8.5_1ab.tar.gz"; done
want grosswendt && get ko_grosswendt2020 "$GEO/GSE137nnn/GSE137337/suppl/GSE137337_L3MBTL2_E8.5_1.tar.gz"
want grosswendt && for k in DNMT1 DNMT3A; do for p in barcodes.tsv.gz features.tsv.gz matrix.mtx.gz; do get ko_grosswendt2020 "$GEO/GSE137nnn/GSE137337/suppl/GSE137337_${k}_E8.5_1ab.$p"; done; done
want grosswendt && for p in GPL21103 GPL24247; do get ko_grosswendt2020 "$GEO/GSE137nnn/GSE137337/matrix/GSE137337-${p}_series_matrix.txt.gz"; done
echo DONE
