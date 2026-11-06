#!/usr/bin/env bash
# Train and run heavy jobs on the remote GPU box, not on this machine.
# Host settings come from .env (git-ignored): VEC_REMOTE, VEC_REMOTE_PORT, VEC_REMOTE_KEY, VEC_REMOTE_DIR,
# VEC_REMOTE_VENV. See .env.example.
#
#   scripts/remote.sh sync              push git-tracked code (including uncommitted edits)
#   scripts/remote.sh data              push the data subset the T3 model needs
#   scripts/remote.sh run CMD...        run CMD in the remote repo (venv active, PYTHONPATH=src)
#   scripts/remote.sh bg NAME CMD...    same, detached with nohup; log in logs/NAME.log on the remote
#   scripts/remote.sh fetch PATH...     pull results back (e.g. runs/ submissions/T3_bcatenin/)
#   scripts/remote.sh ssh               interactive shell on the remote
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; [ -f .env ] && . ./.env; set +a
: "${VEC_REMOTE:?set VEC_REMOTE in .env}"
PORT="${VEC_REMOTE_PORT:-22}"; KEY="${VEC_REMOTE_KEY:-$HOME/.ssh/id_ed25519}"; KEY="${KEY/#\~/$HOME}"
DIR="${VEC_REMOTE_DIR:-/workspace/virtual-embryo-challenge}"; VENV="${VEC_REMOTE_VENV:-/workspace/venv}"
SSH=(ssh -o BatchMode=yes -p "$PORT" -i "$KEY")
RSH="ssh -o BatchMode=yes -p $PORT -i $KEY"
DATA=(data/T2/E8.5.h5ad data/T2/E8.75.h5ad data/T2/E9.5.h5ad data/T3/E8.75_gata4_ko_1.h5ad
      data/T3/E8.75_gata4_ko_2.h5ad data/T3/E9.5_mab21l2_ko.h5ad data/priors/collectri_mouse.csv.gz)
q() { printf '%q ' "$@"; }

cmd="${1:-}"; shift || true
case "$cmd" in
  sync)  git ls-files -z | rsync -rlpt --from0 --files-from=- -e "$RSH" ./ "$VEC_REMOTE:$DIR/" ;;
  data)  rsync -rlptR --partial -e "$RSH" "${DATA[@]}" "$VEC_REMOTE:$DIR/" ;;
  run)   "${SSH[@]}" "$VEC_REMOTE" "cd $DIR && . $VENV/bin/activate && PYTHONPATH=src $(q "$@")" ;;
  bg)    name="$1"; shift
         "${SSH[@]}" "$VEC_REMOTE" "cd $DIR && mkdir -p logs && . $VENV/bin/activate && \
           PYTHONPATH=src nohup $(q "$@") > logs/$name.log 2>&1 < /dev/null & echo started $name, log $DIR/logs/$name.log" ;;
  fetch) for p in "$@"; do rsync -rlptR -e "$RSH" "$VEC_REMOTE:$DIR/./$p" ./; done ;;
  ssh)   exec "${SSH[@]}" -t "$VEC_REMOTE" "cd $DIR && . $VENV/bin/activate && exec bash" ;;
  *)     sed -n '2,12p' "$0"; exit 1 ;;
esac
