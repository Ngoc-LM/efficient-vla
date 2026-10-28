"""Summary table of all evaluated configurations against a reference run (paired on (task, init)).

Per configuration: episodes, success rate (Wilson 95% CI), paired difference to the reference with
Newcombe 95% CI and task-cluster bootstrap CI, exact McNemar p, non-inferiority at the margin,
policy calls per episode, policy GPU-seconds per episode and per control step, and gripper chatter
(short gripper segments of <= 3 steps per 100 control steps) from the executed actions.

Usage:
  python -m evla.analyze --root results/t4/raw --ref baseline --out results/t4/summary.json
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np

from evla import stats


def chatter_per_100(actions: np.ndarray, max_len: int = 3) -> tuple[float, float]:
    """(short gripper segments per 100 steps, gripper switches per 100 steps)."""
    g = np.sign(actions[:, -1])
    idx = np.where(g[1:] != g[:-1])[0] + 1
    if len(g) == 0:
        return 0.0, 0.0
    seg = np.diff(np.r_[idx, len(g)]) if len(idx) else np.array([], dtype=int)
    return float((seg <= max_len).sum() / len(g) * 100), float(len(idx) / len(g) * 100)


def load(root: str) -> dict[str, list[dict]]:
    by_method: dict[str, dict[tuple[int, int], dict]] = defaultdict(dict)
    for path in sorted(glob.glob(os.path.join(root, "*.csv"))):
        act_dir = path[: -len(".csv")] + "_actions"
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                key = (int(r["task_id"]), int(r["init_id"]))
                a = os.path.join(act_dir, f"t{key[0]}_i{key[1]}.npy")
                if os.path.exists(a):
                    r["chatter"], r["switches"] = chatter_per_100(np.load(a))
                by_method[r["method"]][key] = r  # last row wins if an episode was re-run
    return {m: list(v.values()) for m, v in by_method.items()}


def summarise(rows: list[dict]) -> dict:
    n = len(rows)
    x = sum(int(r["success"]) for r in rows)
    lo, hi = stats.wilson(x, n)
    steps = np.array([int(r["n_steps"]) for r in rows], dtype=float)
    calls = np.array([int(r["n_chunk_calls"]) for r in rows], dtype=float)
    pol = np.array([float(r["policy_s"]) for r in rows])
    out = {
        "n": n,
        "sr": x / n if n else float("nan"),
        "sr_ci95": [lo, hi],
        "calls_per_episode": float(calls.mean()),
        "calls_per_100_steps": float(calls.sum() / steps.sum() * 100),
        "policy_s_per_episode": float(pol.mean()),
        "policy_ms_per_step": float(pol.sum() / steps.sum() * 1e3),
        "policy_ms_per_call_p50": float(np.median([float(r["policy_ms_p50"]) for r in rows if r["policy_ms_p50"]])),
        "steps_success_median": float(np.median([int(r["n_steps"]) for r in rows if r["success"] == "1"] or [np.nan])),
    }
    ch = [r["chatter"] for r in rows if "chatter" in r]
    if ch:
        out["chatter_per_100"] = float(np.mean(ch))
        out["chatter_per_100_failures"] = float(np.mean([r["chatter"] for r in rows if "chatter" in r and r["success"] == "0"] or [np.nan]))
        out["switches_per_100"] = float(np.mean([r["switches"] for r in rows if "switches" in r]))
    ev = [int(r["n_events"]) for r in rows if r.get("n_events") not in (None, "")]
    if ev:
        out["events_per_episode"] = float(np.mean(ev))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/t4/raw")
    ap.add_argument("--ref", default="baseline")
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    data = load(args.root)
    ref = data.get(args.ref, [])
    table = {}
    for m, rows in sorted(data.items()):
        s = summarise(rows)
        if ref and m != args.ref:
            c = stats.compare(ref, rows, args.margin)
            s["paired_vs_ref"] = {k: c[k] for k in (
                "n_pairs", "sr_a", "sr_b", "a_only", "b_only", "diff_b_minus_a",
                "ci95_newcombe", "ci95_task_bootstrap", "mcnemar_p", "noninferior")}
        table[m] = s
    ref_ms = table.get(args.ref, {}).get("policy_ms_per_step")
    hdr = f"{'method':34s} {'n':>4s} {'SR':>6s} {'dSR vs ref [95% CI]':>24s} {'p':>6s} {'calls/100st':>11s} {'ms/step':>8s} {'speedup':>7s} {'chatter':>7s}"
    print(hdr)
    for m, s in table.items():
        p = s.get("paired_vs_ref")
        d = f"{p['diff_b_minus_a']:+.3f} [{p['ci95_newcombe'][0]:+.3f},{p['ci95_newcombe'][1]:+.3f}]" if p else "-"
        pv = f"{p['mcnemar_p']:.3f}" if p else "-"
        sp = f"{ref_ms / s['policy_ms_per_step']:.1f}x" if ref_ms else "-"
        print(f"{m:34s} {s['n']:4d} {s['sr']:6.3f} {d:>24s} {pv:>6s} {s['calls_per_100_steps']:11.1f} "
              f"{s['policy_ms_per_step']:8.0f} {sp:>7s} {s.get('chatter_per_100', float('nan')):7.2f}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(table, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
