#!/usr/bin/env python
"""Turn downloaded external KO datasets into (WT, KO) pairs on the T3 panel (vec.external).

  python scripts/ingest_external.py tal1 hand2 nkx25 grosswendt      -> data/processed/*.h5ad + index.json

Heavy (multi-GB matrices): run on the remote box, e.g. scripts/remote.sh bg ingest python scripts/ingest_external.py ...
"""
import sys
import time

from vec import external

names = sys.argv[1:] or list(external.DATASETS)
for n in names:
    t = time.time()
    print(f"== {n}", flush=True)
    external.DATASETS[n]()
    print(f"== {n} done in {time.time() - t:.0f}s", flush=True)
