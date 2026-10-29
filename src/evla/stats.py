"""Paired statistics for success rates measured on the same (task, init state) episodes.

- McNemar exact test on discordant pairs.
- Newcombe (1998) method 10 confidence interval for a difference of paired proportions.
- Cluster bootstrap over tasks (episodes of one task are correlated).
- Non-inferiority decision against a pre-registered margin.

Usage:
  python -m evla.stats --a results/raw/spatial_baseline.csv --b results/raw/spatial_stale_k5.csv
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict

from scipy import stats as st


def wilson(x: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = x / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def newcombe_paired(a: int, b: int, c: int, d: int, z: float = 1.959963984540054) -> tuple[float, float, float]:
    """CI for p1 - p2 with p1 = (a+b)/n (method A success), p2 = (a+c)/n (method B success).

    a: both succeed, b: A only, c: B only, d: both fail. Newcombe 1998, method 10.
    Returns (diff, lower, upper) for diff = p1 - p2.
    """
    n = a + b + c + d
    if n == 0:
        return (0.0, -1.0, 1.0)
    p1, p2 = (a + b) / n, (a + c) / n
    l1, u1 = wilson(a + b, n, z)
    l2, u2 = wilson(a + c, n, z)
    diff = p1 - p2
    denom = (a + b) * (c + d) * (a + c) * (b + d)
    if denom == 0:
        phi = 0.0
    else:
        num = a * d - b * c
        # Newcombe (1998), method 10: continuity correction of the numerator only when ad > bc
        if num > 0:
            num = max(num - n / 2, 0)
        phi = num / math.sqrt(denom)
    dl = math.sqrt(max(0.0, (p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2))
    du = math.sqrt(max(0.0, (u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2))
    return (diff, max(-1.0, diff - dl), min(1.0, diff + du))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value on discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    return float(st.binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue)


def pair_rows(rows_a: list[dict], rows_b: list[dict]) -> list[tuple[int, int, int]]:
    """[(task_id, success_a, success_b)] on the (task, init) keys present in both runs."""
    ka = {(int(r["task_id"]), int(r["init_id"])): int(r["success"]) for r in rows_a}
    kb = {(int(r["task_id"]), int(r["init_id"])): int(r["success"]) for r in rows_b}
    keys = sorted(set(ka) & set(kb))
    return [(k[0], ka[k], kb[k]) for k in keys]


def table(pairs: list[tuple[int, int, int]]) -> tuple[int, int, int, int]:
    a = sum(1 for _, x, y in pairs if x and y)
    b = sum(1 for _, x, y in pairs if x and not y)
    c = sum(1 for _, x, y in pairs if y and not x)
    d = sum(1 for _, x, y in pairs if not x and not y)
    return a, b, c, d


def cluster_bootstrap(pairs, n_boot: int = 10_000, seed: int = 0) -> tuple[float, float]:
    """95% percentile CI for SR_b - SR_a, resampling tasks with replacement."""
    import numpy as np

    rng = np.random.default_rng(seed)
    by_task: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for t, x, y in pairs:
        by_task[t].append((x, y))
    tasks = sorted(by_task)
    sums = np.array([[sum(y - x for x, y in by_task[t]), len(by_task[t])] for t in tasks], dtype=float)
    idx = rng.integers(0, len(tasks), size=(n_boot, len(tasks)))
    tot = sums[idx].sum(axis=1)
    diffs = tot[:, 0] / tot[:, 1]
    return (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))


def compare(rows_a, rows_b, margin: float = 0.05) -> dict:
    pairs = pair_rows(rows_a, rows_b)
    a, b, c, d = table(pairs)
    n = len(pairs)
    # report B - A (positive = B better)
    diff, lo, hi = newcombe_paired(a, c, b, d)
    blo, bhi = cluster_bootstrap(pairs) if pairs else (float("nan"), float("nan"))
    per_task: dict[int, list[int]] = defaultdict(lambda: [0, 0, 0])
    for t, x, y in pairs:
        per_task[t][0] += x
        per_task[t][1] += y
        per_task[t][2] += 1
    return {
        "n_pairs": n,
        "sr_a": (a + b) / n if n else float("nan"),
        "sr_b": (a + c) / n if n else float("nan"),
        "both": a,
        "a_only": b,
        "b_only": c,
        "neither": d,
        "diff_b_minus_a": diff,
        "ci95_newcombe": [lo, hi],
        "ci95_task_bootstrap": [blo, bhi],
        "mcnemar_p": mcnemar_exact(b, c),
        "noninferiority_margin": margin,
        "noninferior": lo > -margin,
        "per_task": {t: {"sr_a": v[0] / v[2], "sr_b": v[1] / v[2], "n": v[2]} for t, v in sorted(per_task.items())},
    }



def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (step-down, monotone, capped at 1), in the input order."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def main(argv=None) -> int:
    from evla.common import read_rows

    ap = argparse.ArgumentParser()
    ap.add_argument("--a", nargs="+", required=True, help="reference run CSV(s)")
    ap.add_argument("--b", nargs="+", required=True, help="compared run CSV(s)")
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    res = compare(read_rows(args.a), read_rows(args.b), args.margin)
    txt = json.dumps(res, indent=1)
    print(txt)
    if args.out:
        with open(args.out, "w") as f:
            f.write(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
