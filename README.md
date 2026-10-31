# Efficient VLA: SmolVLA on LIBERO

Reproduction, profiling and an inference-time efficiency study of
[SmolVLA](https://arxiv.org/abs/2506.01844) (checkpoint `HuggingFaceVLA/smolvla_libero`,
LeRobot v0.6.1) on LIBERO-Spatial, run on Kaggle Tesla T4 GPUs.
The question: which ways of making a re-plan cheaper keep closed-loop success, tested on
identical episodes with paired statistics.

## Results

<!-- RESULTS: filled from results/t4/summary.json once the last runs finish -->

All numbers come from `results/t4/summary.json` (paired comparisons) and
`results/t4/profile/` (latency). The report (`paper/`) explains the protocol and the analysis.

## Layout

| Path | Content |
|---|---|
| `scripts/kaggle_setup.sh` | Pinned environment setup (LeRobot v0.6.1, MuJoCo < 3.4, NVIDIA EGL) |
| `scripts/run_queue.sh` | Runs a list of configurations one after another, pushes results every 20 min |
| `scripts/launch_eval.sh` | Runs one configuration as several shards on the two T4 GPUs |
| `scripts/queue_*.txt` | The configurations actually run, in order |
| `scripts/monitor.py` | Prints progress of running shards |
| `src/evla/loading.py` | Builds policy, processors and LIBERO envs as `lerobot-eval` does |
| `src/evla/run_eval.py` | Resumable per-episode evaluation (one CSV row per episode) |
| `src/evla/methods.py` | Inference variants, common random numbers, CUDA-graph replay, timing |
| `src/evla/profile.py` | Latency / memory / FLOPs breakdown of one re-plan (CUDA events) |
| `src/evla/bench_graph.py` | Re-plan latency over precision x Euler steps x eager/graph |
| `src/evla/calibrate_warm.py` | Offline calibration of warm-started flow (rejected variant) |
| `src/evla/stats.py` | Paired statistics (exact McNemar, Newcombe CI, task-cluster bootstrap, Holm) |
| `src/evla/analyze.py` | Builds `summary.json` and the comparison table from the episode CSVs |
| `results/t4/raw/` | Episode CSVs, run metadata, executed actions, a few videos |
| `paper/` | LaTeX report (`main.tex`) |
| `deliverables/` | Literature matrix (`literature_matrix.xlsx`) and its sources |
| `notebooks/kaggle_efficient_vla.ipynb` | Clean copy of the Kaggle notebook (setup, profiling, queued runs, analysis) |
| `tests/test_offline.py` | CPU tests (`python -m pytest -q tests`) |

## Environment

Kaggle notebook, accelerator "GPU T4 x2", Internet on.
`scripts/kaggle_setup.sh` installs, with versions pinned on purpose:

| Package | Version | Why pinned |
|---|---|---|
| lerobot (`[smolvla,libero]`) | 0.6.1 (git tag) | Checkpoint and processors match this release |
| mujoco | 3.3.7 (< 3.4) | From 3.4 the stored init states of task 5 settle differently (lerobot #4390) |
| torch | 2.11.0 + CUDA 12.8 | Kaggle image |
| transformers | 5.5.4 | Pulled by lerobot |
| hf-libero / robosuite | 0.1.4 / 1.4.0 | Pulled by lerobot |

The script also registers the NVIDIA EGL vendor library (otherwise MuJoCo renders in software)
and answers LIBERO's first-import prompt. Python 3.13.

## Reproducing

```bash
git clone https://github.com/Ngoc-LM/efficient-vla && cd efficient-vla
bash scripts/kaggle_setup.sh

# 1) component profile and the latency table (isolated calls, batch 1)
PYTHONPATH=src python -m evla.profile --out results/profile_t4.json
PYTHONPATH=src python -m evla.bench_graph --out results/graph_t4.json

# 2) closed-loop runs: each queue line is "<name> <run_eval options>"
bash scripts/bg.sh /kaggle/working/results/queue.log bash scripts/run_queue.sh scripts/queue_spatial_v6.txt
python scripts/monitor.py /kaggle/working/results/raw

# 3) paired analysis (CPU is enough)
PYTHONPATH=src python -m evla.analyze --root results/t4/raw
```

Episodes are keyed by `e = 1000*task + init`; the environment seed is `1000 + e` and the flow
noise of the re-plan at step `k` is drawn with seed `10000*e + k` for every method (common random
numbers), so two methods see the same initial state and the same noise. Interrupted runs resume:
finished episodes in the CSV (or in `--skip-from`) are skipped.

### Variants

| Variant | `run_eval` options |
|---|---|
| Released | `--method baseline` |
| Open loop | `--method open_loop --k 5` (or 10) |
| Gripper events | `... --event-trigger --hold 5` |
| Stale image | `--method stale --k 5 --flow-steps 3` |
| Fewer Euler steps | `--method flow --flow-steps 3` (2, 1) |
| Float16 | `... --dtype fp16` |
| CUDA graph | `... --graph` |
| Fixed noise | `... --fixed-noise` |

Warm-starting the flow from the previous chunk was calibrated offline
(`python -m evla.calibrate_warm --out results/calib.json`) and was no more accurate than a cold start with the same
number of steps, so it was not run in closed loop.

## Statistics

Success-rate differences are paired over identical episodes: exact McNemar test, Newcombe 95%
CI for the paired difference, task-cluster bootstrap CI, and Holm adjustment over the
comparisons with the released policy. A variant is non-inferior when the lower CI bound exceeds
-5 points (margin fixed before the runs).

## Attribution

SmolVLA, LeRobot and LIBERO belong to their authors and keep their own licenses; this
repository only wraps them for evaluation.
