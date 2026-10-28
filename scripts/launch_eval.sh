#!/usr/bin/env bash
# Run one evaluation configuration as two shards on the two T4 GPUs of a Kaggle session.
# Usage: bash scripts/launch_eval.sh <run_name> <max_hours> <run_eval args...>
#   e.g. bash scripts/launch_eval.sh spatial_baseline 10 --suite libero_spatial --method baseline
# Each shard appends one CSV row per finished episode and is resumable, so a run interrupted by a
# session limit (or continued on another Kaggle account) is restarted with the same command.
set -uo pipefail
NAME=$1; MAXH=$2; shift 2
OUT=${OUT_DIR:-/kaggle/working/results/raw}
mkdir -p "$OUT"
export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd)/src:${PYTHONPATH:-}"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
for i in 0 1; do
  nohup python -u -m evla.run_eval "$@" --shard "$i/2" --device "cuda:$i" \
    --max-hours "$MAXH" --out "$OUT/${NAME}_s$i.csv" > "$OUT/${NAME}_s$i.stdout" 2>&1 < /dev/null &
  echo "shard $i pid $!"
done
