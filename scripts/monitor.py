"""Print progress of running shards every N seconds (keeps an interactive Kaggle session busy
and makes the logs visible while the run is in progress).

Usage: python scripts/monitor.py /kaggle/working/results/raw --every 120
"""

import argparse
import csv
import glob
import os
import subprocess
import time


def summary(out_dir: str) -> str:
    lines = []
    for path in sorted(glob.glob(os.path.join(out_dir, "*.csv"))):
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        sr = sum(int(r["success"]) for r in rows) / len(rows)
        wall = sum(float(r["wall_s"]) for r in rows) / len(rows)
        lines.append(f"{os.path.basename(path):40s} n={len(rows):4d} SR={sr:.3f} wall/ep={wall:6.1f}s")
    return "\n".join(lines)


def running() -> int:
    out = subprocess.run(["pgrep", "-f", "evla.run_eval"], capture_output=True, text=True).stdout
    return len([x for x in out.split() if x.strip()])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--every", type=int, default=120)
    args = ap.parse_args()
    while True:
        print(time.strftime("%H:%M:%S"), f"running={running()}", flush=True)
        print(summary(args.out_dir), flush=True)
        for p in sorted(glob.glob(os.path.join(args.out_dir, "*.stdout"))):
            tail = subprocess.run(["tail", "-n", "1", p], capture_output=True, text=True).stdout.strip()
            print("  ", os.path.basename(p), "|", tail[-160:], flush=True)
        if running() == 0:
            print("no evla.run_eval process left", flush=True)
            break
        time.sleep(args.every)
