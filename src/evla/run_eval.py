"""Resumable, sharded LIBERO evaluation of SmolVLA variants (one row per episode).

Each episode is pinned to (task_id, init_id), so the same episode can be replayed by any method,
on any machine or Kaggle account, and results from several sessions can simply be concatenated.

Example (two GPUs, two shards):
  python -m evla.run_eval --suite libero_spatial --tasks 0-9 --inits 0-49 --method baseline \
      --out results/raw/spatial_baseline.csv --shard 0/2 --device cuda:0
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

from evla.common import episode_key, load_done, parse_range, select_shard

FIELDS = [
    "method",
    "suite",
    "task_id",
    "init_id",
    "seed",
    "success",
    "n_steps",
    "n_chunk_calls",
    "n_vision_calls",
    "policy_s",
    "policy_ms_p50",
    "wall_s",
    "refresh_init",
    "refresh_period",
    "refresh_image",
    "refresh_gripper",
    "n_warm",
    "n_cold",
    "n_events",
    "gpu",
    "finished_at",
]


def log(msg: str, fh) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    fh.write(line + "\n")
    fh.flush()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--tasks", default="0-9")
    ap.add_argument("--inits", default="0-49")
    ap.add_argument(
        "--method", choices=["baseline", "open_loop", "flow", "stale", "warm", "stale_warm"], default="baseline"
    )
    ap.add_argument("--k", type=int, default=5, help="open_loop: n_action_steps; stale: refresh period")
    ap.add_argument("--warm-t0", type=float, default=0.5, help="warm: flow time of the noised prior")
    ap.add_argument("--warm-steps", type=int, default=3, help="warm: Euler steps from t0 to 0")
    ap.add_argument("--flow-steps", type=int, default=None, help="Euler steps (default: checkpoint, 10)")
    ap.add_argument("--img-tau", type=float, default=None, help="stale: image-change refresh threshold")
    ap.add_argument("--no-gripper-refresh", action="store_true")
    ap.add_argument("--dtype", choices=["native", "fp32", "fp16", "visfp16"], default="native")
    ap.add_argument("--event-trigger", action="store_true", help="re-plan at predicted gripper switches")
    ap.add_argument("--graph", action="store_true", help="replay the flow integration from a CUDA graph")
    ap.add_argument("--hold", type=int, default=5, help="event trigger: steps without a new trigger")
    ap.add_argument("--fixed-noise", action="store_true", help="same flow noise for every re-plan of an episode")
    ap.add_argument("--policy", default="HuggingFaceVLA/smolvla_libero")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--skip-from", default=None, help="glob of CSVs whose finished episodes are skipped")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-hours", type=float, default=11.0)
    ap.add_argument("--save-actions", action="store_true", help="store executed actions per episode")
    ap.add_argument("--videos", type=int, default=0, help="save up to N failure + N success videos")
    args = ap.parse_args(argv)

    import numpy as np
    import torch

    from evla import loading, methods

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    logf = open(out.with_suffix(".log"), "a")
    act_dir = out.with_suffix("")
    act_dir = act_dir.parent / (act_dir.name + "_actions")
    vid_dir = out.parent / (out.stem + "_videos")

    tasks = parse_range(args.tasks)
    inits = parse_range(args.inits)
    episodes = select_shard([(t, i) for t in tasks for i in inits], args.shard)
    done = load_done(out)
    if args.skip_from:
        import glob

        for f in glob.glob(args.skip_from):
            done |= load_done(f)
    todo = [e for e in episodes if e not in done]
    log(f"start {vars(args)} | shard episodes={len(episodes)} done={len(episodes) - len(todo)}", logf)
    if not todo:
        return 0

    n_action_steps = args.k if args.method == "open_loop" else None
    if args.device.startswith("cuda:"):
        # make cuda:N the current device, so that streams and CUDA graphs are created on it
        import torch

        torch.cuda.set_device(args.device)
    b = loading.build(
        args.suite,
        policy_path=args.policy,
        device=args.device,
        n_action_steps=n_action_steps,
        num_steps=args.flow_steps,
        dtype=args.dtype,
    )
    policy = b.policy
    log(f"policy cfg: n_action_steps={policy.config.n_action_steps} chunk={policy.config.chunk_size} "
        f"num_steps={policy.config.num_steps} dtypes={loading.param_dtypes(policy)} gpu={loading.gpu_name()}", logf)

    stats = methods.CallStats()
    crn = methods.CRNNoise(fixed=args.fixed_noise)
    sampler = None
    if args.method in ("stale", "warm", "stale_warm"):
        sampler = methods.install_sampler(
            policy,
            stats,
            refresh_every=args.k if "stale" in args.method else 1,
            img_tau=args.img_tau,
            refresh_on_gripper=not args.no_gripper_refresh,
            warm_t0=args.warm_t0 if "warm" in args.method else None,
            warm_steps=args.warm_steps if "warm" in args.method else None,
        )
    if args.graph:
        if sampler is not None:
            raise SystemExit("--graph cannot be combined with stale/warm samplers")
        methods.install_graph(policy, stats)
    methods.instrument(policy, stats, crn=crn, sampler=sampler)
    trigger = None
    if args.event_trigger:
        trigger = methods.install_event_trigger(policy, b.postprocessor, hold=args.hold)

    def post(action):
        a = b.postprocessor(action)
        if sampler is not None:
            sampler.note_action(a)
        return a

    from lerobot.scripts.lerobot_eval import rollout

    frames: list = []

    def render_cb(env):
        f = env.call("render")[0]
        frames.append(f[::2, ::2].copy())

    write_header = not out.exists() or out.stat().st_size == 0
    fcsv = open(out, "a", newline="")
    writer = csv.DictWriter(fcsv, fieldnames=FIELDS)
    if write_header:
        writer.writeheader()
        fcsv.flush()

    t_start = time.time()
    envs: dict[int, object] = {}
    n_succ = n_run = 0
    vids = {"success": 0, "failure": 0}
    for task_id, init_id in todo:
        if (time.time() - t_start) / 3600 > args.max_hours:
            log("max-hours reached, stopping cleanly", logf)
            break
        if task_id not in envs:
            for e in envs.values():
                e.close()
            envs = {task_id: loading.make_task_env(b.env_cfg, task_id)}
        env = envs[task_id]
        loading.set_init_state(env, init_id)
        key = episode_key(task_id, init_id)
        seed = args.seed_base + key
        crn.set_episode(key)
        stats.reset()
        frames.clear()
        want_video = args.videos > 0 and min(vids.values()) < args.videos
        t0 = time.time()
        with torch.inference_mode():
            data = rollout(
                env=env,
                policy=policy,
                env_preprocessor=b.env_preprocessor,
                env_postprocessor=b.env_postprocessor,
                preprocessor=b.preprocessor,
                postprocessor=post,
                seeds=[seed],
                render_callback=render_cb if want_video else None,
            )
        wall = time.time() - t0
        done_mask = data["done"][0].numpy().astype(bool)
        n_steps = int(np.argmax(done_mask)) + 1 if done_mask.any() else int(done_mask.shape[0])
        success = bool(data["success"][0].any().item())
        trig = sampler.n_refresh_trigger if sampler is not None else {}
        row = {
            "method": method_name(args),
            "suite": args.suite,
            "task_id": task_id,
            "init_id": init_id,
            "seed": seed,
            "success": int(success),
            "n_steps": n_steps,
            "n_chunk_calls": stats.n_chunk_calls,
            "n_vision_calls": stats.n_vision_calls,
            "policy_s": round(stats.policy_s, 4),
            "policy_ms_p50": round(float(np.median(stats.per_call_ms)), 3) if stats.per_call_ms else "",
            "wall_s": round(wall, 2),
            "refresh_init": trig.get("init", ""),
            "refresh_period": trig.get("period", ""),
            "refresh_image": trig.get("image", ""),
            "refresh_gripper": trig.get("gripper", ""),
            "n_warm": sampler.n_warm if sampler is not None else "",
            "n_cold": sampler.n_cold if sampler is not None else "",
            "n_events": trigger.n_events if trigger is not None else "",
            "gpu": loading.gpu_name(),
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        writer.writerow(row)
        fcsv.flush()
        os.fsync(fcsv.fileno())
        if args.save_actions:
            act_dir.mkdir(parents=True, exist_ok=True)
            np.save(act_dir / f"t{task_id}_i{init_id}.npy", data["action"][0, :n_steps].numpy().astype(np.float32))
        if want_video and frames:
            kind = "success" if success else "failure"
            if vids[kind] < args.videos:
                save_video(vid_dir / f"{kind}_t{task_id}_i{init_id}.mp4", frames)
                vids[kind] += 1
        n_run += 1
        n_succ += int(success)
        el = time.time() - t_start
        eta = el / n_run * (len(todo) - n_run) / 3600
        log(
            f"t{task_id} i{init_id} success={int(success)} steps={n_steps} calls={stats.n_chunk_calls} "
            f"vision={stats.n_vision_calls} policy={stats.policy_s:.1f}s wall={wall:.1f}s | "
            f"{n_run}/{len(todo)} SR={n_succ / n_run:.3f} ETA={eta:.2f}h",
            logf,
        )
    fcsv.close()
    with open(out.with_suffix(".meta.json"), "w") as fm:
        json.dump({"args": vars(args), "python": sys.version, "torch": torch.__version__}, fm, indent=1)
    return 0


def method_name(args) -> str:
    if args.method == "baseline":
        name = "baseline"
    elif args.method == "open_loop":
        name = f"open_loop_k{args.k}"
    elif args.method == "flow":
        name = f"flow_s{args.flow_steps}"
    else:
        parts = []
        if "stale" in args.method:
            parts.append(f"stale_k{args.k}" + (f"_tau{args.img_tau}" if args.img_tau is not None else ""))
        if "warm" in args.method:
            parts.append(f"warm_t{args.warm_t0}_m{args.warm_steps}")
        name = "+".join(parts) + ("_nogrip" if args.no_gripper_refresh else "")
    if args.event_trigger:
        name += f"_evt_h{args.hold}"
    if args.method != "flow" and args.flow_steps is not None:
        name += f"_s{args.flow_steps}"
    if args.dtype != "native":
        name += f"_{args.dtype}"
    if args.fixed_noise:
        name += "_fixnoise"
    if args.graph:
        name += "_graph"
    return name


def save_video(path: Path, frames: list) -> None:
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=20)


if __name__ == "__main__":
    raise SystemExit(main())
