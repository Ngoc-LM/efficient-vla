#!/usr/bin/env bash
# Runs after queue v6 has finished: D2 probe first (success rate), then D3 latency/equivalence runs.
# Start detached:  bash scripts/bg.sh /kaggle/working/results/queue_v7.log bash scripts/followup_v7.sh
cd "$(dirname "$0")/.."
while pgrep -f "run_queu[e].sh scripts/queue_spatial_v6" > /dev/null; do sleep 60; done
bash scripts/run_queue.sh scripts/queue_spatial_v7.txt
OUT_DIR=/kaggle/working/results/latency SHARDS=2 \
  COMMON_ARGS="--suite libero_spatial --tasks 0-9 --inits 0-1 --save-actions" \
  bash scripts/run_queue.sh scripts/queue_latency_v7.txt
