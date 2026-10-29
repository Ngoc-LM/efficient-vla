#!/usr/bin/env bash
# Run a list of evaluation configurations one after another on a Kaggle session and push the
# results to GitHub after each one. Start it detached:
#   bash scripts/bg.sh /kaggle/working/results/queue.log bash scripts/run_queue.sh scripts/queue_spatial.txt
# Each line of the queue file: <run_name> <run_eval args...>   (lines starting with # are skipped)
# Re-running the same queue resumes: finished episodes are skipped by run_eval.
set -uo pipefail
QUEUE=$1
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT=${OUT_DIR:-/kaggle/working/results/raw}
COMMON=${COMMON_ARGS:---suite libero_spatial --tasks 0-9 --inits 0-19 --save-actions --videos 2}
busy() { pgrep -f -- "-m evla.run_eval" > /dev/null; }
# never start while another evaluation is still running (e.g. a previous queue)
while busy; do sleep 60; done
while read -r NAME ARGS; do
  [[ -z "${NAME:-}" || "$NAME" == \#* ]] && continue
  if [[ "$NAME" == "!" ]]; then   # raw command line, e.g. a profiling run
    echo "[$(date +%T)] run: $ARGS"
    (cd "$HERE/.." && PYTHONPATH=src MUJOCO_GL=egl PYOPENGL_PLATFORM=egl bash -c "$ARGS")
    echo "[$(date +%T)] done: exit $?"
    python "$HERE/push_results.py" --src "$(dirname "$OUT")" --dest t4 --message "T4 results: ${ARGS:0:60}" 2>&1 | tail -1
    continue
  fi
  echo "[$(date +%T)] start $NAME: $ARGS"
  SHARDS=${SHARDS:-4} bash "$HERE/launch_eval.sh" "$NAME" "${MAX_HOURS:-6}" $COMMON $ARGS
  sleep 30
  t=0
  while busy; do   # push partial results regularly so a lost session costs little
    sleep 60; t=$((t + 1))
    if (( t % ${PUSH_EVERY_MIN:-20} == 0 )); then
      python "$HERE/push_results.py" --src "$(dirname "$OUT")" --dest t4 --message "T4 results: $NAME (in progress)" 2>&1 | tail -1
    fi
  done
  n=$(cat "$OUT/${NAME}"_s*.csv 2>/dev/null | grep -vc '^method')
  echo "[$(date +%T)] done  $NAME: $n episodes"
  python "$HERE/push_results.py" --src "$(dirname "$OUT")" --dest t4 --message "T4 results: $NAME ($n episodes)" 2>&1 | tail -1
done < "$QUEUE"
echo "[$(date +%T)] queue finished"
