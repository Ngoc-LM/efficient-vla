"""How much does the executed action depend on the flow noise, for S = 1, 3 and 10 Euler steps?

The float16 policy with S = 3 and common random numbers drives the robot (the configuration of the
closed-loop runs). Every `--every` control steps, the same observation and prefix are reused to
integrate the flow from K different noises for each S, and the first (executed) action is compared
across noises, in the policy's normalised action space:

  spread       mean distance of the K actions from their mean (sampling spread for this observation)
  to_mean10    distance of the mean action for S from the mean of the K actions for S = 10
  grip_split   fraction of the K actions whose gripper sign differs from the majority sign

With LeRobot's convention x_tau = tau * eps + (1 - tau) * A and Euler integration from tau = 1, one
step returns eps - v(eps, 1 | o); at the optimum of the flow-matching loss this equals E[A | o] for
any eps, so the spread for S = 1 measures how far the trained network is from that limit.

Usage (GPU):
  python -m evla.noise_spread --suite libero_spatial --tasks 0-4 --inits 0 --out results/spread_a.json
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np


def first_action_stats(acts: np.ndarray, ref_mean: np.ndarray | None) -> dict:
    """acts: (K, adim) first actions for K noises; the last dimension is the gripper."""
    mean = acts.mean(axis=0)
    spread = float(np.linalg.norm(acts - mean, axis=1).mean())
    signs = np.sign(acts[:, -1])
    majority = 1.0 if (signs >= 0).sum() * 2 >= len(signs) else -1.0
    grip_split = float((signs != majority).mean())
    out = {"spread": spread, "grip_split": grip_split}
    if ref_mean is not None:
        out["to_mean10"] = float(np.linalg.norm(mean - ref_mean))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--tasks", default="0-9")
    ap.add_argument("--inits", default="0")
    ap.add_argument("--steps", default="1,3,10", help="Euler steps compared")
    ap.add_argument("--k", type=int, default=8, help="noises per observation")
    ap.add_argument("--every", type=int, default=6, help="probe every n-th control step")
    ap.add_argument("--max-steps", type=int, default=150, help="cap control steps per episode")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    import torch
    from lerobot.policies.common.flow_matching import euler_integrate
    from lerobot.scripts.lerobot_eval import rollout

    from evla import loading, methods
    from evla.common import episode_key, parse_range

    if args.device.startswith("cuda:"):
        torch.cuda.set_device(args.device)
    steps = [int(x) for x in args.steps.split(",")]
    b = loading.build(args.suite, device=args.device, dtype="fp16")
    policy, m = b.policy, b.policy.model
    crn = methods.CRNNoise()
    stats = methods.CallStats()
    helper = methods.FastSampler(m, stats)  # only for its prefix routine
    adim = b.policy.config.action_feature.shape[0]
    rows: list[dict] = []
    state = {"k": 0}

    @torch.no_grad()
    def sample_actions(images, img_masks, lang_tokens, lang_masks, st, noise=None, **kw):
        bsize, device = st.shape[0], st.device
        shape = (bsize, m.config.chunk_size, m.config.max_action_dim)
        if noise is None:
            noise = m.sample_noise(shape, device)
        pad, cache = helper._prefix_full(images, img_masks, lang_tokens, lang_masks, st)

        def denoise(x_t, t):
            return m.denoise_step(x_t=x_t, prefix_pad_masks=pad, past_key_values=cache, timestep=t)

        drive = euler_integrate(denoise, noise, 3)  # the trajectory follows fp16, S = 3, CRN noise
        if state["k"] % args.every == 0:
            g = torch.Generator(device="cpu")
            g.manual_seed(crn.episode_key * 7919 + crn.step + 17)
            noises = [torch.randn(shape, generator=g).to(device) for _ in range(args.k)]
            firsts = {}
            for s in sorted(steps, reverse=True):  # S = 10 first: its mean is the reference
                acts = torch.cat([euler_integrate(denoise, z, s)[:, 0, :adim] for z in noises]).float()
                firsts[s] = acts.cpu().numpy()
            ref = firsts[max(steps)].mean(axis=0)
            row = {"episode": crn.episode_key, "step": state["k"]}
            for s in steps:
                for key, val in first_action_stats(firsts[s], None if s == max(steps) else ref).items():
                    row[f"S{s}_{key}"] = val
            rows.append(row)
        state["k"] += 1
        return drive

    m.sample_actions = sample_actions
    methods.instrument(policy, stats, crn=crn)
    episodes = [(t, i) for t in parse_range(args.tasks) for i in parse_range(args.inits)]
    t_start = time.time()
    for task_id, init_id in episodes:
        env = loading.make_task_env(b.env_cfg, task_id)
        env.envs[0].unwrapped._max_episode_steps = args.max_steps
        loading.set_init_state(env, init_id)
        crn.set_episode(episode_key(task_id, init_id))
        state["k"] = 0
        with torch.inference_mode():
            data = rollout(
                env=env, policy=policy, env_preprocessor=b.env_preprocessor,
                env_postprocessor=b.env_postprocessor, preprocessor=b.preprocessor,
                postprocessor=b.postprocessor, seeds=[1000 + episode_key(task_id, init_id)],
            )
        env.close()
        print(f"[{time.time() - t_start:6.0f}s] t{task_id} i{init_id} success={bool(data['success'][0].any())} "
              f"probes={len(rows)}", flush=True)

    keys = [k for k in rows[0] if k.startswith("S")] if rows else []
    summary = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    with open(args.out, "w") as f:
        json.dump({"args": vars(args), "n_probes": len(rows), "summary": summary, "rows": rows}, f, indent=1)
    for k in sorted(summary):
        print(f"{k:16s} {summary[k]:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
