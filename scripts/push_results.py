"""Push result files from a Kaggle session to the GitHub repo (results/<dest>/), as the repo owner.

The GitHub token is read from the Kaggle secret `GH_TOKEN` inside the session and is never printed.
Usage (in a Kaggle cell):
  python efficient-vla/scripts/push_results.py --src /kaggle/working/results/raw --dest raw \
      --message "Spatial baseline: shard progress"
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import shutil
import subprocess
import time

REPO = "github.com/Ngoc-LM/efficient-vla.git"
NAME = "Minh-Ngoc Luong"
EMAIL = "70001123+Ngoc-LM@users.noreply.github.com"
PATTERNS = ["*.csv", "*.json", "*.log", "*.npy", "*.mp4", "*.png", "*.txt"]


def run(cmd: list[str], token: str, cwd: str | None = None, check: bool = True) -> str:
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    out = (p.stdout + p.stderr).replace(token, "***")
    if check and p.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd[:3])} failed:\n{out}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dest", required=True, help="sub-directory under results/")
    ap.add_argument("--message", default="Add Kaggle results")
    ap.add_argument("--work", default="/kaggle/working/_push_repo")
    ap.add_argument("--max-mb", type=float, default=20.0, help="skip single files larger than this")
    args = ap.parse_args()

    from kaggle_secrets import UserSecretsClient

    token = UserSecretsClient().get_secret("GH_TOKEN")
    url = f"https://x-access-token:{token}@{REPO}"

    if not os.path.isdir(os.path.join(args.work, ".git")):
        run(["git", "clone", "-q", "--depth", "1", url, args.work], token)
    run(["git", "config", "user.name", NAME], token, cwd=args.work)
    run(["git", "config", "user.email", EMAIL], token, cwd=args.work)
    run(["git", "remote", "set-url", "origin", url], token, cwd=args.work)

    dest_root = os.path.join(args.work, "results", args.dest)
    n = 0
    for root, _, files in os.walk(args.src):
        for f in files:
            if not any(fnmatch.fnmatch(f, p) for p in PATTERNS):
                continue
            src = os.path.join(root, f)
            if os.path.getsize(src) > args.max_mb * 2**20:
                print(f"skip (too large): {src}")
                continue
            rel = os.path.relpath(src, args.src)
            dst = os.path.join(dest_root, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            n += 1
    run(["git", "add", "-A", "results"], token, cwd=args.work)
    status = run(["git", "status", "--porcelain"], token, cwd=args.work)
    if not status.strip():
        print(f"nothing new to push ({n} files checked)")
        return 0
    run(["git", "commit", "-q", "-m", args.message], token, cwd=args.work)
    for attempt in range(4):
        out = run(["git", "pull", "-q", "--rebase", "origin", "main"], token, cwd=args.work, check=False)
        out = run(["git", "push", "-q", "origin", "HEAD:main"], token, cwd=args.work, check=False)
        if "rejected" not in out and "error" not in out.lower():
            sha = run(["git", "rev-parse", "--short", "HEAD"], token, cwd=args.work).strip()
            print(f"pushed {n} files -> results/{args.dest} @ {sha}")
            return 0
        print(f"push attempt {attempt + 1} failed, retrying:\n{out}")
        time.sleep(5)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
