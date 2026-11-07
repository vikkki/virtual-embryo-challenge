"""Is Gata4's sample-wide offset an artefact of the WT E8.75 reference sample (so beta-catenin, scored against the
same reference, would share it) or Gata4's own biology?

If the reference carries its own artefact a_g, then KO - WT = -a_g + biology, and the WT E8.75 sample should stand
out from its neighbours WT E8.5 / WT E9.5 by +a_g ("bump" = E8.75 minus the stage interpolation, per state).
A strong NEGATIVE correlation between bump and Gata4's offset supports the artefact reading.

  python analysis/t3_reference_bump.py   -> runs/t3_reference_bump.json  (remote box; data-derived)
"""
import json
from pathlib import Path

import numpy as np

from vec import t3

S = {k: t3.load(k) for k in ["WT_E8.5", "WT_E8.75", "WT_E9.5", "Gata4_1", "Gata4_2", "Mab21l2"]}
wt = S["WT_E8.75"]
t3.transfer_states(wt, [S["WT_E8.5"], S["WT_E9.5"], S["Gata4_1"], S["Gata4_2"]])
genes = list(wt.genes)
states = sorted(set(wt.state))
st = {k: t3.state_stats(S[k], states) for k in ["WT_E8.5", "WT_E8.75", "WT_E9.5", "Gata4_1", "Gata4_2"]}
ok = np.all([st[k]["n"] >= 50 for k in st], 0)
mu = {k: st[k]["mu"][ok] for k in st}
bump = np.median(mu["WT_E8.75"] - (0.75 * mu["WT_E8.5"] + 0.25 * mu["WT_E9.5"]), 0)
stage = np.median(mu["WT_E8.75"] - mu["WT_E8.5"], 0)
g4 = mu["WT_E8.75"][:, genes.index("Gata4")]
non = g4 < 0.3                                   # states that do not express Gata4
off = np.median(((mu["Gata4_1"] + mu["Gata4_2"]) / 2 - mu["WT_E8.75"])[non], 0)
# Mab21l2 against ITS reference (E9.5): does the E9.5 sample carry a bump of its own that matches Mab21l2's offset?
t3.transfer_states(S["WT_E9.5"], [S["Mab21l2"]])
dm = t3.decompose(S["Mab21l2"], S["WT_E9.5"], min_cells=50)
off_m = np.median(dm["delta"][dm["ok"]], 0)
cor = lambda a, b: float(np.corrcoef(a, b)[0, 1])
out = {
    "n_states": int(ok.sum()), "n_non_gata4_states": int(non.sum()),
    "r(bump E8.75, Gata4 offset)": cor(bump, off),
    "r(stage E8.75-E8.5, Gata4 offset)": cor(stage, off),
    "r(bump E8.75, Mab21l2 offset) [different reference: control]": cor(bump, off_m),
    "slope of offset on bump": float(np.polyfit(bump, off, 1)[0]),
    "top offset genes with bump": [(genes[i], round(float(off[i]), 2), round(float(bump[i]), 2))
                                   for i in np.argsort(-np.abs(off))[:15]],
}
print(json.dumps(out, indent=1))
Path("runs/t3_reference_bump.json").write_text(json.dumps(out, indent=1))
