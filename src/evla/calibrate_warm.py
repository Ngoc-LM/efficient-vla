"""Offline calibration of the warm-start time t0 on a suite that is NOT used for the final evaluation.

The released policy (cold start, 10 Euler steps, re-planning every control step) drives the robot.
At every control step the same prefix and the same flow noise are reused to also compute warm-started
chunks (previous chunk shifted by one action, placed at time t0, integrated with M steps) and cold
chunks with fewer steps. Reported: error of the executed (first) action and of the first 5 actions
against the 10-step cold chunk, in the policy's normalised action space. A second cold 10-step chunk
with different noise gives the policy's own sampling spread (the floor below which errors do not matter).

Usage:
  python -m evla.calibrate_warm --suite libero_object --tasks 0-9 --inits 0 --out results/calib_object.json
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_object")
    ap.add_argument("--tasks", default="0-9")
    ap.add_argument("--inits", default="0")
    ap.add_argument("--t0s", default="0.2,0.3,0.4,0.5,0.6,0.8")
    ap.add_argument("--ms", default="2,3,5")
    ap.add_argument("--max-steps", type=int, default=None, help="cap control steps per episode")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    import torch
    from lerobot.policies.common.flow_matching import euler_integrate
    from lerobot.scripts.lerobot_eval import rollout

    from evla import loading, methods
    from evla.common import episode_key, parse_range

    t0s = [float(x) for x in args.t0s.split(",")]
    ms = [int(x) for x in args.ms.split(",")]
    b = loading.build(args.suite, device=args.device)
    policy, m = b.policy, b.policy.model
    crn = methods.CRNNoise()
    stats = methods.CallStats()
    helper = methods.FastSampler(m, stats)  # only for its prefix routine
    adim = b.policy.config.action_feature.shape[0]
    errs: dict[str, list[tuple[float, float]]] = {}
    state = {"prev": None}

    def add(name, x, ref):
        e1 = float((x[:, 0, :adim] - ref[:, 0, :adim]).norm(dim=-1).mean())
        e5 = float((x[:, :5, :adim] - ref[:, :5, :adim]).norm(dim=-1).mean())
        errs.setdefault(name, []).append((e1, e5))

    @torch.no_grad()
    def sample_actions(images, img_masks, lang_tokens, lang_masks, st, noise=None, **kw):
        bsize, device = st.shape[0], st.device
        shape = (bsize, m.config.chunk_size, m.config.max_action_dim)
        if noise is None:
            noise = m.sample_noise(shape, device)
        pad, cache = helper._prefix_full(images, img_masks, lang_tokens, lang_masks, st)

        def denoise(x_t, t):
            return m.denoise_step(x_t=x_t, prefix_pad_masks=pad, past_key_values=cache, timestep=t)

        ref = euler_integrate(denoise, noise, 10)
        g = torch.Generator(device="cpu")
        g.manual_seed(crn.episode_key * 7919 + crn.step + 1)
        other = torch.randn(shape, generator=g).to(device)
        add("cold10_other_noise", euler_integrate(denoise, other, 10), ref)
        for s in ms:
            add(f"cold{s}", euler_integrate(denoise, noise, s), ref)
        if state["prev"] is not None:
            prior = methods.shift_chunk(state["prev"], 1)
            add("prior_only", prior, ref)
            for t0 in t0s:
                for s in ms:
                    add(f"warm_t{t0}_m{s}", methods.warm_integrate(denoise, noise, prior, t0, s), ref)
        state["prev"] = ref
        return ref

    m.sample_actions = sample_actions
    methods.instrument(policy, stats, crn=crn)
    episodes = [(t, i) for t in parse_range(args.tasks) for i in parse_range(args.inits)]
    succ = []
    t_start = time.time()
    for task_id, init_id in episodes:
        env = loading.make_task_env(b.env_cfg, task_id)
        if args.max_steps:
            env.envs[0].unwrapped._max_episode_steps = args.max_steps
        loading.set_init_state(env, init_id)
        crn.set_episode(episode_key(task_id, init_id))
        state["prev"] = None
        with torch.inference_mode():
            data = rollout(
                env=env, policy=policy, env_preprocessor=b.env_preprocessor,
                env_postprocessor=b.env_postprocessor, preprocessor=b.preprocessor,
                postprocessor=b.postprocessor, seeds=[1000 + episode_key(task_id, init_id)],
            )
        succ.append(bool(data["success"][0].any()))
        env.close()
        print(f"[{time.time() - t_start:6.0f}s] t{task_id} i{init_id} success={succ[-1]} "
              f"n={len(errs.get('cold3', []))}", flush=True)

    summary = {
        k: {"n": len(v), "err_first": float(np.mean([a for a, _ in v])), "err_first5": float(np.mean([c for _, c in v]))}
        for k, v in sorted(errs.items())
    }
    res = {"args": vars(args), "episodes": episodes, "success": succ, "summary": summary}
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    for k, v in sorted(summary.items(), key=lambda kv: kv[1]["err_first"]):
        print(f"{k:24s} err_first={v['err_first']:.4f} err_first5={v['err_first5']:.4f} (n={v['n']})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
