"""Official board specs and gene panels (downloaded into ../../panels)."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

PANELS_DIR = Path(__file__).resolve().parents[2] / "panels"


@lru_cache
def boards() -> dict:
    return json.loads((PANELS_DIR / "index.json").read_text())


def board(key: str) -> dict:
    b = boards()
    if key not in b:
        raise KeyError(f"unknown board {key!r}; known: {sorted(b)}")
    return b[key]


@lru_cache
def genes(key: str) -> tuple[str, ...]:
    path = PANELS_DIR / board(key)["genes_file"]
    return tuple(line.strip() for line in path.read_text().splitlines() if line.strip())
