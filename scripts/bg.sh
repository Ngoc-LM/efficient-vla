#!/usr/bin/env bash
# Run any command detached from the notebook cell (survives the end of the cell).
# Usage: bash scripts/bg.sh <logfile> <command...>
LOG=$1; shift
mkdir -p "$(dirname "$LOG")"
export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd)/src:${PYTHONPATH:-}"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
setsid nohup "$@" > "$LOG" 2>&1 < /dev/null &
echo "started pid $! -> $LOG"
