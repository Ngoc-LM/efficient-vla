"""Numbers quoted in the report, recomputed from the episode CSVs and latency runs.

  PYTHONPATH=src python scripts/report_numbers.py --raw results/t4/raw --latency results/t4/latency \
      --out results/t4/report_numbers.json

1. Table of closed-loop results: every variant against the released policy on initial states 0-19
   (200 pairs), Holm adjustment over the comparisons in the table.
2. Pre-specified tests on more pairs: fp16 S=3 on initial states 0-49 (500 pairs, primary
   non-inferiority test) and fp16 S=1 on the fresh initial states 20-49 only (300 pairs,
   confirmation of the S-curve result, not pooled with the exploratory states 0-19).
3. Fixed noise against fresh noise (both fp16, S=3), states 0-19.
4. In-loop latency with one process per GPU, and whether CUDA-graph replay changes trajectories
   compared with the run-to-run variation of eager execution.
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
