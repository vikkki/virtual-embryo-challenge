#!/usr/bin/env bash
# Re-download official board specs + gene lists (re-run if organisers update them).
set -euo pipefail
cd "$(dirname "$0")/../panels"
base=https://virtualembryo.ai/challenge/panels
curl -sfL "$base/index.json" -o index.json
python3 -c "import json;[print(v['genes_file']) for v in json.load(open('index.json')).values()]" |
  while read -r f; do curl -sfL "$base/$f" -o "$f"; done
git -C .. diff --stat -- panels || true
