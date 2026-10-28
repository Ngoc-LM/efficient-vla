"""Latency, memory and FLOPs breakdown of SmolVLA on the current GPU (no rollout, fixed inputs).

Measures, with CUDA events, after warm-up, interleaving precision modes round by round so that
GPU clock drift affects all modes equally:

  vision      SigLIP encoder + connector for both cameras
  prefix      image/language/state embedding + 32-layer VLM prefill (includes `vision`)
  expert1     one flow-matching denoise step of the action expert (reads the prefix KV cache)
  chunk       full `sample_actions` = prefix + num_steps expert steps (one re-plan)
  stale       stale-vision step without refresh = state token through the VLM + num_steps expert steps
  e2e         preprocessor + select_action + postprocessor for one control step (n_action_steps=1)
  sim         one LIBERO env.step (physics + rendering of two 360x360 cameras), CPU/GPU mixed

Usage:
  python -m evla.profile --out results/profile_t4.json --repeats 100
"""

from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext

import numpy as np


def pct(xs):
    a = np.asarray(xs, dtype=float)
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "p50": float(np.percentile(a, 50)),
        "p90": float(np.percentile(a, 90)),
        "p99": float(np.percentile(a, 99)),
        "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--init", type=int, default=0)
    ap.add_argument("--modes", default="native,fp16,visfp16")
    ap.add_argument("--repeats", type=int, default=100)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--flow-steps", default="10,3")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    import torch
    from lerobot.envs import preprocess_observation
    from lerobot.envs.utils import NEW_ROLLOUT_OPTION
    from lerobot.policies.common.vla_utils import make_att_2d_masks
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
    from torch.utils.flop_counter import FlopCounterMode

    from evla import loading, methods

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    base_modes = sorted({m.split("_")[0] for m in modes})
    bundles = {bm: loading.build(args.suite, device=args.device, dtype=bm) for bm in base_modes}
    any_b = bundles[base_modes[0]]

    # one fixed observation from the real environment
    env = loading.make_task_env(any_b.env_cfg, args.task)
    loading.set_init_state(env, args.init)
    raw_obs, _ = env.reset(seed=[0], options={NEW_ROLLOUT_OPTION: True})
    task_desc = list(env.call("task_description"))

    def make_batch(b):
        o = preprocess_observation(raw_obs)
        o["task"] = task_desc
        o = b.env_preprocessor(o)
        return b.preprocessor(o)

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

    flow_list = [int(x) for x in args.flow_steps.split(",")]
    samples: dict[str, list[float]] = {}
    peak_mem: dict[str, float] = {}
    flops: dict[str, float] = {}
    info: dict = {"gpu": loading.gpu_name(), "torch": torch.__version__}

    def components(b, amp: bool):
        pol = b.policy
        m = pol.model
        vwe = m.vlm_with_expert
        batch = make_batch(b)
        ctx = (lambda: torch.autocast("cuda", dtype=torch.float16)) if amp else nullcontext
        with torch.inference_mode(), ctx():
            images, img_masks = pol.prepare_images(batch)
            state = pol.prepare_state(batch)
            lang_tokens = batch[OBS_LANGUAGE_TOKENS]
            lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
            g = torch.Generator(device="cpu")
            g.manual_seed(0)
            noise = torch.randn(1, m.config.chunk_size, m.config.max_action_dim, generator=g).to(state.device)

        def f_vision():
            with torch.inference_mode(), ctx():
                for im in images:
                    vwe.embed_image(im)

        def prefix():
            embs, pad, att = m.embed_prefix(images, img_masks, lang_tokens, lang_masks, state=state)
            att2d = make_att_2d_masks(pad, att)
            pos = torch.cumsum(pad, dim=1) - 1
            _, kv = vwe.forward(
                attention_mask=att2d, position_ids=pos, past_key_values=None,
                inputs_embeds=[embs, None], use_cache=True,
            )
            return pad, kv

        def f_prefix():
            with torch.inference_mode(), ctx():
                prefix()

        with torch.inference_mode(), ctx():
            pad0, kv0 = prefix()

        def f_expert1():
            with torch.inference_mode(), ctx():
                m.denoise_step(
                    x_t=noise, prefix_pad_masks=pad0, past_key_values=kv0,
                    timestep=torch.ones(1, device=noise.device),
                )

        def f_chunk():
            with torch.inference_mode(), ctx():
                m.sample_actions(images, img_masks, lang_tokens, lang_masks, state, noise=noise)

        stats = methods.CallStats()
        sampler = methods.StaleVisionSampler(m, stats, refresh_every=10**9, refresh_on_gripper=False)
        with torch.inference_mode(), ctx():
            sampler.sample_actions(images, img_masks, lang_tokens, lang_masks, state, noise=noise)

        def f_stale():
            with torch.inference_mode(), ctx():
                sampler.sample_actions(images, img_masks, lang_tokens, lang_masks, state, noise=noise)

        def f_e2e():
            with torch.inference_mode(), ctx():
                pol.reset()
                o = make_batch(b)
                a = pol.select_action(o)
                b.postprocessor(a)

        # equivalence check of the stale path (refresh at every step == baseline chunk)
        s1 = methods.StaleVisionSampler(m, methods.CallStats(), refresh_every=1, refresh_on_gripper=False)
        with torch.inference_mode(), ctx():
            a_ref = m.sample_actions(images, img_masks, lang_tokens, lang_masks, state, noise=noise)
            a_st = s1.sample_actions(images, img_masks, lang_tokens, lang_masks, state, noise=noise)
            a_st2 = sampler.sample_actions(images, img_masks, lang_tokens, lang_masks, state, noise=noise)
        eq = {
            "actions_ref": a_ref.float().cpu(),
            "max_abs_diff_refresh_every_step": float((a_ref - a_st).abs().max()),
            "max_abs_diff_cached_same_image": float((a_ref - a_st2).abs().max()),
            "action_abs_mean": float(a_ref.abs().mean()),
        }
        return {
            "vision": f_vision, "prefix": f_prefix, "expert1": f_expert1,
            "chunk": f_chunk, "stale": f_stale, "e2e": f_e2e,
        }, eq, m

    comp = {}
    for mode in modes:
        base, amp = mode.split("_")[0], mode.endswith("amp16")
        comp[mode] = components(bundles[base], amp)
        info[f"param_dtypes_{base}"] = loading.param_dtypes(bundles[base].policy)
    # same observation and same flow noise: deviation of each precision mode from the native chunk
    ref = comp[modes[0]][1]["actions_ref"]
    for mode in modes:
        eq = comp[mode][1]
        a = eq.pop("actions_ref")
        adim = any_b.policy.config.action_feature.shape[0]
        eq[f"chunk_max_abs_diff_vs_{modes[0]}"] = float((a - ref)[..., :adim].abs().max())
        eq[f"chunk_mean_abs_diff_vs_{modes[0]}"] = float((a - ref)[..., :adim].abs().mean())
        eq["finite"] = bool(torch.isfinite(a).all())
        info[f"equivalence_{mode}"] = eq

    # parameter counts per block
    m0 = any_b.policy.model
    vlm = m0.vlm_with_expert.get_vlm_model()
    info["params"] = {
        "vision_encoder": sum(p.numel() for p in vlm.vision_model.parameters()),
        "connector": sum(p.numel() for p in vlm.connector.parameters()),
        "text_model": sum(p.numel() for p in vlm.text_model.parameters()),
        "action_expert": sum(p.numel() for p in m0.vlm_with_expert.lm_expert.parameters()),
        "total": sum(p.numel() for p in any_b.policy.parameters()),
    }

    # FLOPs (counted once, native mode semantics)
    fns0 = comp[modes[0]][0]
    for name in ["vision", "prefix", "expert1", "chunk", "stale"]:
        fc = FlopCounterMode(display=False)
        with fc:
            fns0[name]()
        flops[name] = fc.get_total_flops() / 1e9

    # warm-up
    for mode in modes:
        for name, fn in comp[mode][0].items():
            time_fn(fn, args.warmup)

    # interleaved timing rounds
    per_round = max(1, args.repeats // args.rounds)
    for r in range(args.rounds):
        for mode in modes:
            fns, _, m = comp[mode]
            for name, fn in fns.items():
                if name == "chunk" or name == "stale":
                    for s in flow_list:
                        m.config.num_steps = s
                        samples.setdefault(f"{mode}/{name}_s{s}", []).extend(time_fn(fn, per_round))
                    m.config.num_steps = 10
                else:
                    samples.setdefault(f"{mode}/{name}", []).extend(time_fn(fn, per_round))
        print(f"round {r + 1}/{args.rounds} done", flush=True)

    # memory per mode: weight bytes of that model + activation peak of one chunk call
    for mode in modes:
        fns, _, _ = comp[mode]
        pol = bundles[mode.split("_")[0]].policy
        weights = sum(p.numel() * p.element_size() for p in pol.parameters()) / 2**20
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        fns["chunk"]()
        torch.cuda.synchronize()
        act = (torch.cuda.max_memory_allocated() - before) / 2**20
        peak_mem[mode] = {"weights_mib": weights, "activation_peak_mib": act, "total_mib": weights + act}

    # simulator step time (zero action), independent of the policy
    sim = []
    a0 = np.zeros((1, 7), dtype=np.float32)
    for _ in range(60):
        t0 = time.perf_counter()
        env.step(a0)
        sim.append((time.perf_counter() - t0) * 1e3)
    samples["sim/env_step"] = sim[10:]

    res = {
        "info": info,
        "flops_gflop": flops,
        "peak_mem_mib_chunk": peak_mem,
        "latency_ms": {k: pct(v) for k, v in samples.items()},
    }
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    with open(args.out.replace(".json", "_samples.json"), "w") as f:
        json.dump(samples, f)
    for k, v in res["latency_ms"].items():
        print(f"{k:32s} p50={v['p50']:8.2f}  p90={v['p90']:8.2f}  mean={v['mean']:8.2f} ms", flush=True)
    print(json.dumps({"flops_gflop": flops, "peak_mem": peak_mem, "info": info}, indent=1), flush=True)
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
