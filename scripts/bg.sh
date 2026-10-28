#!/usr/bin/env bash
# Run any command detached from the notebook cell (survives the end of the cell).
# Usage: bash scripts/bg.sh <logfile> <command...>
LOG=$1; shift
mkdir -p "$(dirname "$LOG")"
export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd)/src:${PYTHONPATH:-}"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
# setsid -f forks into a new session before this script returns, so the job cannot be caught by
# the notebook killing the cell's process group (a plain `cmd &` loses that race).
setsid -f nohup "$@" > "$LOG" 2>&1 < /dev/null
sleep 1
echo "started -> $LOG"; pgrep -af "$2 $3 $4" | head -3
