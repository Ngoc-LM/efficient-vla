#!/usr/bin/env bash
# Run one evaluation configuration as SHARDS (default 2) processes on the two T4 GPUs of a Kaggle session.
# Usage: bash scripts/launch_eval.sh <run_name> <max_hours> <run_eval args...>
#   e.g. bash scripts/launch_eval.sh spatial_baseline 10 --suite libero_spatial --method baseline
# Processes are started with setsid so that they survive the end of the notebook cell.
# Each shard appends one CSV row per finished episode and is resumable, so a run interrupted by a
# session limit (or continued on another Kaggle account) is restarted with the same command.
set -uo pipefail
NAME=$1; MAXH=$2; shift 2
OUT=${OUT_DIR:-/kaggle/working/results/raw}
mkdir -p "$OUT"
export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd)/src:${PYTHONPATH:-}"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
# one CPU thread per process: several processes share 4 vCPUs, torch defaults to all of them
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
N=${SHARDS:-2}   # processes; shard i runs on GPU i%2 (2 per GPU is faster: batch-1 inference is launch-bound)
for ((i = 0; i < N; i++)); do
  setsid -f nohup python -u -m evla.run_eval "$@" --shard "$i/$N" --device "cuda:$((i % 2))" \
    --skip-from "$OUT/${NAME}_s*.csv" \
    --max-hours "$MAXH" --out "$OUT/${NAME}_s$i.csv" > "$OUT/${NAME}_s$i.stdout" 2>&1 < /dev/null
done
sleep 1
pgrep -af "evla.run_eval" | grep -- "$NAME" | head -4
