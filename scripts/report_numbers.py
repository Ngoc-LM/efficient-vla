"""Numbers quoted in the report, recomputed from the episode CSVs and latency runs.

  PYTHONPATH=src python scripts/report_numbers.py --raw results/t4/raw --latency results/t4/latency \
      --out results/t4/report_numbers.json

1. Table of closed-loop results: every variant against the released policy on initial states 0-19
   (200 pairs), Holm adjustment over the comparisons in the table.
2. Tests on more pairs: fp16 S=3 on initial states 0-49 (500 pairs; states 20-49 were added after the
   200-pair lower bound missed the margin, so they are also reported alone) and fp16 S=1 on the fresh
   initial states 20-49 only (300 pairs, a confirmation fixed before the run, not pooled with the
   exploratory states 0-19).
3. Fixed noise against fresh noise (both fp16, S=3), states 0-19.
4. In-loop latency with one process per GPU, and whether CUDA-graph replay changes trajectories
   compared with the run-to-run variation of eager execution.
5. Failure modes of the released policy (states 0-19) from the executed gripper commands: segments of
   at most 3 steps are merged into their neighbour (chatter), then failures are classed as closing the
   gripper three or more times, else still closed at the step limit, else reopened after 1-2 closures.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from evla import stats
from evla.analyze import load, summarise

REF = "baseline"
TABLE = [  # (label, method) in report order
    ("open_loop_k5", "open_loop_k5"),
    ("open_loop_k10", "open_loop_k10"),
    ("open_loop_k5_events", "open_loop_k5_evt_h5"),
    ("stale_image", "stale_k5_s3"),
    ("bf16_s3", "flow_s3"),
    ("fp16_s3", "flow_s3_fp16"),
    ("fp16_s2", "flow_s2_fp16"),
    ("fp16_s1", "flow_s1_fp16"),
    ("fp16_s3_fixed_noise", "flow_s3_fp16_fixnoise"),
]


def subset(rows, inits):
    return [r for r in rows if int(r["init_id"]) in inits]


def cmp(rows_ref, rows_var):
    c = stats.compare(rows_ref, rows_var)
    c.pop("per_task", None)
    return c


def actions(run_glob):
    out = {}
    for d in glob.glob(run_glob):
        for f in glob.glob(os.path.join(d, "*.npy")):
            out[os.path.basename(f)] = np.load(f)
    return out


def gripper_segments(actions, short=3):
    """[(sign, length)] of the gripper command (+1 closed), segments of <= `short` steps merged."""
    segs: list[list[int]] = []
    for v in np.where(actions[:, -1] > 0, 1, -1):
        if segs and segs[-1][0] == v:
            segs[-1][1] += 1
        else:
            segs.append([int(v), 1])
    while len(segs) > 1:
        idx = [i for i, (_, n) in enumerate(segs) if n <= short]
        if not idx:
            break
        i = idx[0]
        segs[i - 1 if i > 0 else 1][1] += segs[i][1]
        del segs[i]
        merged: list[list[int]] = []
        for v, n in segs:
            if merged and merged[-1][0] == v:
                merged[-1][1] += n
            else:
                merged.append([v, n])
        segs = merged
    return [tuple(x) for x in segs]


def failure_modes(raw_dir, run="spatial_baseline", inits=range(20)):
    import csv

    counts = {"closes_3_or_more": 0, "closed_at_limit": 0, "reopened_after_1_2_closures": 0, "no_closure": 0}
    for path in sorted(glob.glob(os.path.join(raw_dir, f"{run}_s[0-9].csv"))):
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                if int(r["init_id"]) not in inits or r["success"] == "1":
                    continue
                a = np.load(os.path.join(path[:-4] + "_actions", f"t{r['task_id']}_i{r['init_id']}.npy"))
                segs = gripper_segments(a)
                closes = sum(1 for v, _ in segs if v > 0)
                if closes == 0:
                    counts["no_closure"] += 1
                elif closes >= 3:
                    counts["closes_3_or_more"] += 1
                elif segs[-1][0] > 0:
                    counts["closed_at_limit"] += 1
                else:
                    counts["reopened_after_1_2_closures"] += 1
    return counts


def same_trajectory(a, b):
    return len(a) == len(b) and bool(np.array_equal(a, b))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="results/t4/raw")
    ap.add_argument("--latency", default="results/t4/latency")
    ap.add_argument("--out", default="results/t4/report_numbers.json")
    args = ap.parse_args()

    runs = load(args.raw)
    s0 = set(range(20))
    ref0 = subset(runs[REF], s0)
    out: dict = {"table_states_0_19": {}, "pre_specified": {}, "latency_in_loop": {}}

    rows = []
    for label, m in TABLE:
        if m not in runs:
            continue
        var = subset(runs[m], s0)
        c = cmp(ref0, var)
        c.update({k: v for k, v in summarise(var).items() if k in ("chatter_per_100", "calls_per_100_steps")})
        rows.append((label, c))
    adj = stats.holm([c["mcnemar_p"] for _, c in rows])
    for (label, c), p in zip(rows, adj):
        c["mcnemar_p_holm"] = p
        c["holm_family_size"] = len(rows)
        out["table_states_0_19"][label] = c
    out["table_states_0_19"]["released"] = {"n": len(ref0), **{k: v for k, v in summarise(ref0).items()
                                                                 if k in ("sr", "chatter_per_100")}}

    s_all, s_new = set(range(50)), set(range(20, 50))
    out["pre_specified"]["fp16_s3_states_0_49"] = cmp(subset(runs[REF], s_all), subset(runs["flow_s3_fp16"], s_all))
    # states 20-49 were added after the 200-pair lower bound (-5.8) missed the margin: report them alone too
    out["pre_specified"]["fp16_s3_states_20_49"] = cmp(subset(runs[REF], s_new), subset(runs["flow_s3_fp16"], s_new))
    if "flow_s1_fp16" in runs:
        new = subset(runs["flow_s1_fp16"], s_new)
        if new:
            out["pre_specified"]["fp16_s1_states_20_49"] = cmp(subset(runs[REF], s_new), new)
            out["pre_specified"]["fp16_s1_states_0_49_descriptive"] = cmp(subset(runs[REF], s_all),
                                                                          subset(runs["flow_s1_fp16"], s_all))
    if "flow_s3_fp16_fixnoise" in runs:
        out["fixed_vs_fresh_noise_fp16_s3"] = cmp(subset(runs["flow_s3_fp16"], s0),
                                                  subset(runs["flow_s3_fp16_fixnoise"], s0))

    if os.path.isdir(args.latency):
        lat = load(args.latency)
        for m, r in lat.items():
            ms = [float(x["policy_ms_p50"]) for x in r]
            out["latency_in_loop"][m] = {"episodes": len(r), "replan_ms_median_of_episode_p50": float(np.median(ms)),
                                         "min": float(min(ms)), "max": float(max(ms))}
        eager = actions(os.path.join(args.latency, "lat_fp16_s3_s[0-9]_actions"))
        graph = actions(os.path.join(args.latency, "lat_fp16_s3_graph_s[0-9]_actions"))
        eager_again = actions(os.path.join(args.raw, "spatial_cold3_fp16_s[0-9]_actions"))
        eq = {}
        for name, other in (("graph_vs_eager", graph), ("eager_vs_eager_other_run", eager_again)):
            keys = sorted(set(eager) & set(other))
            eq[name] = {"episodes": len(keys),
                        "identical_trajectories": sum(same_trajectory(eager[k], other[k]) for k in keys),
                        "identical_first_action": sum(bool(np.array_equal(eager[k][0], other[k][0])) for k in keys)}
        out["latency_in_loop"]["trajectory_equality"] = eq

    out["failure_modes_released_states_0_19"] = failure_modes(args.raw)

    # noise-spread probe (src/evla/noise_spread.py): pooled over the two GPU shards, weighted by probes
    probe = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(os.path.dirname(args.raw), "probe", "spread_*.json")))]
    if probe:
        n = sum(p["n_probes"] for p in probe)
        keys = probe[0]["summary"].keys()
        out["noise_spread_probe"] = {"n_probes": n, **{k: sum(p["summary"][k] * p["n_probes"] for p in probe) / n for k in keys}}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)

    print(f"{'variant (states 0-19)':24s} {'SR':>6s} {'dSR':>7s} {'95% CI':>17s} {'p_holm':>7s} {'chatter':>7s}")
    print(f"{'released':24s} {out['table_states_0_19']['released']['sr']:6.3f}")
    for label, c in rows:
        lo, hi = c["ci95_newcombe"]
        print(f"{label:24s} {c['sr_b']:6.3f} {c['diff_b_minus_a']:+7.3f} [{lo:+.3f},{hi:+.3f}] "
              f"{c['mcnemar_p_holm']:7.3f} {c.get('chatter_per_100', float('nan')):7.2f}")
    for k, c in out["pre_specified"].items():
        lo, hi = c["ci95_newcombe"]
        print(f"{k:34s} n={c['n_pairs']} {c['sr_a']:.3f}->{c['sr_b']:.3f} {c['diff_b_minus_a']:+.3f} "
              f"[{lo:+.3f},{hi:+.3f}] p={c['mcnemar_p']:.3g} boot={np.round(c['ci95_task_bootstrap'], 3)}")
    if "fixed_vs_fresh_noise_fp16_s3" in out:
        c = out["fixed_vs_fresh_noise_fp16_s3"]
        print(f"fixed vs fresh noise: {c['sr_a']:.3f}->{c['sr_b']:.3f} discordant {c['a_only']}/{c['b_only']} "
              f"p={c['mcnemar_p']:.2g}")
    print(json.dumps(out["latency_in_loop"], indent=1))
    print("failure modes (released, states 0-19):", out["failure_modes_released_states_0_19"])
    if "noise_spread_probe" in out:
        print("noise-spread probe:", {k: round(v, 3) for k, v in out["noise_spread_probe"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
