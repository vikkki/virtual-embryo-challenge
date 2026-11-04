#!/usr/bin/env python
"""Usage: python scripts/validate_submission.py PRED.h5ad BOARD_KEY  (e.g. T1:test)"""
import sys

from vec.validate import validate

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    errs = validate(sys.argv[1], sys.argv[2])
    for e in errs:
        print("FAIL:", e)
    print("OK" if not errs else f"{len(errs)} problem(s)")
    sys.exit(1 if errs else 0)
