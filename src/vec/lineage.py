"""Where a conditional knockout acts: Cre-lineage weights over the WT E8.75 cell states.

The challenge data page (virtualembryo.ai/challenge/data) states that both E8.75 knockouts are Mesp1-Cre driven:
Mesp1-Cre; Gata4 F/F; Gata6 F/+ (validation) and Mesp1-Cre; beta-catenin F/F (test). The deletion is restricted
to the Mesp1 lineage.

Weights are set a priori, before looking at any knockout data, from (a) the published Mesp1-Cre lineage, which
labels essentially all of the heart (myocardium, endocardium), second heart field / pharyngeal mesoderm, head
mesoderm, much of the anterior lateral plate, extraembryonic mesoderm and many endothelial cells, and only part
of the somites; and (b) each WT state's identity from marker genes on the panel (Myl7; Cdh5/Pecam1/Nfatc1;
Tbx1/Isl1/Fgf10; Meox1/Tcf15/Pax3; Hand1/Twist1/Foxf1; Sox2/Tfap2a/Wnt6; Sox10/Foxd3; Foxa2/Sox17).
1 = fully in the lineage, 0 = outside it, in between = partly labelled or a mixed state.
"""
from __future__ import annotations

import warnings

import numpy as np

MESP1_LINEAGE = {
    # heart and second heart field
    "V-CM": 1.0, "IFT-CM": 1.0,
    "Endo": 1.0,                 # endocardium (Cdh5, Pecam1, Nfatc1)
    "JCF": 1.0,                  # juxta-cardiac / proepicardial mesoderm (Hand1, Hand2, Twist1, Tbx18)
    "aPHM": 1.0, "pPHM": 1.0,    # pharyngeal mesoderm / second heart field (Tbx1, Fgf10, Tcf21)
    "Peri": 0.8,                 # Hand1+/Twist1+ mesoderm next to the heart
    # head and lateral mesoderm
    "PAM-1": 0.8, "PAM-2": 0.8,  # Tbx1+ head / pharyngeal arch mesoderm
    "PAM-4": 0.7,                # Twist1-high head mesenchyme
    "PAM-3": 0.4,                # Meox1/Tcf15/Pax3: paraxial (somitic) mesoderm, partly labelled
    "LPM": 0.8,                  # Foxf1+ lateral plate
    "SOM": 0.3,                  # somites: only partly Mesp1-derived
    # extraembryonic mesoderm and vessels
    "ExEM-1": 0.7, "ExEM-2": 0.6, "Allantois": 0.4,
    "Intra-Endoth-1": 0.7, "Intra-Endoth-2": 0.6, "HEM-Endoth": 0.5,
    # mixed (Myl7-high, unannotated)
    "Unknown": 0.3,
    # outside the lineage: ectoderm, neural crest, neural, epiblast, endoderm
    "d-CSE": 0.0, "V-CSE": 0.0, "pSE": 0.0, "NCC": 0.0, "Neural Tube": 0.0, "Forebrain": 0.0,
    "Caudal Epiblast": 0.0, "D-FG": 0.0, "a-FG": 0.0, "Lateral FG": 0.0, "Gut Endoderm": 0.0, "Hindgut": 0.0,
    "EXE-Endoderm": 0.0,
}

# challenge KO examples and the Cre that drives them (Mab21l2's driver is not stated, so it is left out)
CRE = {"Gata4_1": "Mesp1", "Gata4_2": "Mesp1", "Ctnnb1": "Mesp1"}
LINEAGES = {"Mesp1": MESP1_LINEAGE}


def weights(states: list[str], cre: str = "Mesp1") -> np.ndarray:
    """Lineage weight per state (S,), 0 for states the map does not know (with a warning)."""
    lin = LINEAGES[cre]
    missing = [s for s in states if s not in lin]
    if missing:
        warnings.warn(f"no {cre} lineage weight for {missing}; using 0")
    return np.array([lin.get(s, 0.0) for s in states], float)
