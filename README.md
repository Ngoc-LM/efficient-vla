# Efficient VLA: SmolVLA on LIBERO

Reproduction, profiling and an inference-time efficiency study of
[SmolVLA](https://arxiv.org/abs/2506.01844) (checkpoint `HuggingFaceVLA/smolvla_libero`,
LeRobot v0.6.1) on the LIBERO benchmark, run on a single Kaggle Tesla T4.

> Work in progress. Results, setup notes and the report are added as the runs finish.

## Layout

| Path | Content |
|---|---|
| `scripts/kaggle_setup.sh` | Pinned environment setup (LeRobot v0.6.1, MuJoCo < 3.4, NVIDIA EGL) |
| `scripts/launch_eval.sh` | Runs one configuration as two shards on the two T4 GPUs |
| `scripts/monitor.py` | Prints progress of running shards |
| `src/evla/loading.py` | Builds policy, processors and LIBERO envs as `lerobot-eval` does |
| `src/evla/run_eval.py` | Resumable per-episode evaluation (one CSV row per episode) |
| `src/evla/methods.py` | Inference variants (open-loop chunking, fewer flow steps, stale-vision KV reuse), common random numbers, timing |
| `src/evla/profile.py` | Latency / memory / FLOPs breakdown with CUDA events |
| `src/evla/stats.py` | Paired statistics (McNemar, Newcombe CI, task-level bootstrap, non-inferiority) |
| `tests/test_offline.py` | CPU tests (`python -m pytest -q tests`) |

## Quick start (Kaggle, GPU T4 x2, Internet on)

```bash
git clone https://github.com/Ngoc-LM/efficient-vla && cd efficient-vla
bash scripts/kaggle_setup.sh
PYTHONPATH=src python -m evla.profile --out results/profile_t4.json
bash scripts/launch_eval.sh spatial_baseline 10 --suite libero_spatial --method baseline
python scripts/monitor.py /kaggle/working/results/raw
```
