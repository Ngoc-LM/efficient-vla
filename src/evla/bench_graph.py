"""Latency of one re-plan with and without CUDA-graph replay of the flow integration.

For each precision mode and number of Euler steps: eager `sample_actions` (released code path) vs
`GraphedFlowSampler` (prefix eager, integration replayed from a CUDA graph). Same observation and
same flow noise; reports p50/p90 latency (CUDA events, warm-up, interleaved rounds), graph capture
time, and the max/mean absolute action difference to the eager result.

Usage:
  python -m evla.bench_graph --out results/profile/graph_t4.json
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--modes", default="native,fp16")
    ap.add_argument("--steps", default="10,3,2,1")
    ap.add_argument("--repeats", type=int, default=60)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    import torch
    from lerobot.envs import preprocess_observation
    from lerobot.envs.utils import NEW_ROLLOUT_OPTION
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    from evla import loading, methods

    steps_list = [int(s) for s in args.steps.split(",")]
    res: dict = {"gpu": None, "results": {}}

    def ev():
        return torch.cuda.Event(enable_timing=True)

    def time_fn(fn, n):
        out = []
        for _ in range(n):
            s, e = ev(), ev()
            s.record()
            fn()
            e.record()
            torch.cuda.synchronize()
            out.append(s.elapsed_time(e))
        return out

    for mode in args.modes.split(","):
        b = loading.build(args.suite, device=args.device, dtype=mode)
        res["gpu"] = loading.gpu_name()
        pol, m = b.policy, b.policy.model
        env = loading.make_task_env(b.env_cfg, args.task)
        loading.set_init_state(env, 0)
        raw, _ = env.reset(seed=[0], options={NEW_ROLLOUT_OPTION: True})
        o = preprocess_observation(raw)
        o["task"] = list(env.call("task_description"))
        batch = b.preprocessor(b.env_preprocessor(o))
        env.close()
        with torch.inference_mode():
            images, img_masks = pol.prepare_images(batch)
            state = pol.prepare_state(batch)
            lt, lm = batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK]
            g = torch.Generator(device="cpu")
            g.manual_seed(0)
            noise = torch.randn(1, m.config.chunk_size, m.config.max_action_dim, generator=g).to(state.device)

        eager = m.sample_actions  # released implementation (bound method)
        refs = {}
        with torch.inference_mode():
            for s in steps_list:
                m.config.num_steps = s
                refs[s] = eager(images, img_masks, lt, lm, state, noise=noise).float().cpu()

        sampler = methods.GraphedFlowSampler(m)  # also swaps in the graph-safe embed_suffix
        prefix_t, eager_t, graph_t, capture_s, diffs = {}, {}, {}, {}, {}
        with torch.inference_mode():
            for s in steps_list:
                m.config.num_steps = s
                a_patched = eager(images, img_masks, lt, lm, state, noise=noise).float().cpu()
                t0 = time.perf_counter()
                a_graph = sampler.sample_actions(images, img_masks, lt, lm, state, noise=noise).float().cpu()
                torch.cuda.synchronize()
                capture_s[s] = time.perf_counter() - t0
                a_graph2 = sampler.sample_actions(images, img_masks, lt, lm, state, noise=noise).float().cpu()
                adim = pol.config.action_feature.shape[0]
                diffs[s] = {
                    "eager_patched_vs_released_max": float((a_patched - refs[s])[..., :adim].abs().max()),
                    "graph_vs_released_max": float((a_graph - refs[s])[..., :adim].abs().max()),
                    "graph_vs_released_mean": float((a_graph - refs[s])[..., :adim].abs().mean()),
                    "graph_replay_repeatable_max": float((a_graph2 - a_graph)[..., :adim].abs().max()),
                    "finite": bool(torch.isfinite(a_graph).all()),
                }

            def f_prefix():
                sampler.prefix(images, img_masks, lt, lm, state)

            for s in steps_list:
                m.config.num_steps = s
                time_fn(lambda: eager(images, img_masks, lt, lm, state, noise=noise), args.warmup)
                time_fn(lambda: sampler.sample_actions(images, img_masks, lt, lm, state, noise=noise), args.warmup)
            time_fn(f_prefix, args.warmup)
            per = max(1, args.repeats // args.rounds)
            for _ in range(args.rounds):
                prefix_t.setdefault("all", []).extend(time_fn(f_prefix, per))
                for s in steps_list:
                    m.config.num_steps = s
                    eager_t.setdefault(s, []).extend(
                        time_fn(lambda: eager(images, img_masks, lt, lm, state, noise=noise), per))
                    graph_t.setdefault(s, []).extend(
                        time_fn(lambda: sampler.sample_actions(images, img_masks, lt, lm, state, noise=noise), per))

        def q(x):
            return {"p50": float(np.percentile(x, 50)), "p90": float(np.percentile(x, 90)), "n": len(x)}

        res["results"][mode] = {
            "prefix_ms": q(prefix_t["all"]),
            "per_steps": {
                str(s): {
                    "eager_ms": q(eager_t[s]),
                    "graph_ms": q(graph_t[s]),
                    "speedup_p50": float(np.percentile(eager_t[s], 50) / np.percentile(graph_t[s], 50)),
                    "capture_s": capture_s[s],
                    **diffs[s],
                }
                for s in steps_list
            },
        }
        r = res["results"][mode]
        print(f"[{mode}] prefix p50 {r['prefix_ms']['p50']:.1f} ms", flush=True)
        for s in steps_list:
            x = r["per_steps"][str(s)]
            print(f"[{mode}] steps={s:2d} eager {x['eager_ms']['p50']:7.1f} ms | graph {x['graph_ms']['p50']:7.1f} ms "
                  f"| x{x['speedup_p50']:.2f} | max|da| {x['graph_vs_released_max']:.2e} | capture {x['capture_s']:.2f}s",
                  flush=True)
        del sampler, b, pol, m
        torch.cuda.empty_cache()

    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
