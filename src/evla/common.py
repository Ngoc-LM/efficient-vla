"""Small helpers shared by the runner and the analysis scripts (no GPU or LeRobot imports)."""

from __future__ import annotations

import csv
from pathlib import Path


def parse_range(spec: str) -> list[int]:
    """'0-3,7,9-10' -> [0, 1, 2, 3, 7, 9, 10] (order kept, duplicates removed)."""
    out: list[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            lo_i, hi_i = int(lo), int(hi)
            if hi_i < lo_i:
                raise ValueError(f"bad range {part!r}")
            out.extend(range(lo_i, hi_i + 1))
        else:
            out.append(int(part))
    seen: set[int] = set()
    return [x for x in out if not (x in seen or seen.add(x))]


def select_shard(items: list, shard: str) -> list:
    """'i/n' -> every n-th item starting at i (interleaved, so shards see all tasks)."""
    i, n = (int(x) for x in shard.split("/"))
    if not 0 <= i < n:
        raise ValueError(f"bad shard {shard!r}")
    return items[i::n]


def episode_key(task_id: int, init_id: int) -> int:
    return int(task_id) * 1000 + int(init_id)


def load_done(path: str | Path) -> set[tuple[int, int]]:
    """(task_id, init_id) pairs already written to a results CSV."""
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return set()
    with open(p, newline="") as f:
        return {(int(r["task_id"]), int(r["init_id"])) for r in csv.DictReader(f)}


def read_rows(paths: list[str | Path]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        with open(path, newline="") as f:
            rows.extend(csv.DictReader(f))
    return rows
