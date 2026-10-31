# Run queues

Each line of a queue file is `<run name> <run_eval options>`; `scripts/run_queue.sh <queue>` runs the lines
one after another on a Kaggle session and pushes results to `results/t4/`. Versions record what was run, in order;
logs of each queue are in `results/t4/logs/`.

| Queue | Runs | Used in the report |
|---|---|---|
| `queue_spatial.txt` … `queue_spatial_v4.txt` | open loop (k = 5, 10), gripper events, stale image, bf16 S = 3, fp16 S = 3 on initial states 0–19, relaunched after session losses; v1–v2 also list variants dropped before running (warm start, others). The released policy on states 0–19 was run directly with `scripts/launch_eval.sh`. `results/t4/raw/*.csv` shows what actually ran. | Table 4 (H1, bf16/fp16 S = 3) |
| `queue_spatial_v5.txt`, `queue_spatial_v6.txt` | fp16 S = 3 and released policy on states 20–49 (added after the 200-pair interim look), fp16 S = 2 and S = 1 | §5 (500 pairs), Table 4 (S-curve) |
| `queue_spatial_v7.txt` + `followup_v7.sh` | fixed-noise probe; in-loop latency and graph check (`queue_latency_v7.txt`, one process per GPU) | Table 4 (probe), §5 (in-loop latency) |
| `queue_spatial_v8.txt` + `followup_v8.sh` | fp16 S = 1 on states 20–49 (confirmation fixed before the run) | §5 (S = 1 on fresh pairs) |
