"""Approximate leaderboard-style score from veckit metrics using the board anchors in panels/index.json.

Each metric is mapped linearly so the board's floor model -> 50 and the split-half ceiling -> 100
(works for lower-is-better metrics too, since floor/ceiling carry the direction). The weights below are
the published primary-metric weights; this is an approximation, not the official formula.
"""
from __future__ import annotations

from .panels import board

WEIGHTS = {
    "T1": {"de_score": 25, "de_direction": 25, "mmd_u": 30, "variogram": 20},
    # T3: response genes 30, direction 25, magnitude (severity_slope) 25, cell-state distribution 20
    "T3": {"de_score": 30, "de_direction": 25, "severity_slope": 25, "mmd_u": 20},
}


def component(value: float, floor: float, ceiling: float) -> float:
    return 50 + 50 * (value - floor) / (ceiling - floor)


def approx_score(metrics: dict, key: str) -> dict:
    spec = board(key)
    w = WEIGHTS[spec["task"]]
    parts = {m: component(metrics[m], **spec["anchors"][m]) for m in w}
    total = sum(w[m] * parts[m] for m in w) / sum(w.values())
    return {"total": round(total, 1), **{m: round(v, 1) for m, v in parts.items()}}
