#!/usr/bin/env bash
# Runs after follow-up v7 (D2 probe + latency) has finished: confirmation run for S=1.
# Start detached:  bash scripts/bg.sh /kaggle/working/results/queue_v8.log bash scripts/queues/followup_v8.sh
cd "$(dirname "$0")/../.."
while pgrep -f "followup_v[7].sh" > /dev/null; do sleep 60; done
bash scripts/run_queue.sh scripts/queues/queue_spatial_v8.txt
