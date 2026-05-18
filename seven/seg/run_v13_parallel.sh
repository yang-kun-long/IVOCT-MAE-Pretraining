#!/bin/bash
# Parallel launcher for v13 LOPO training.
#
# Usage:
#   bash run_v13_parallel.sh <fold1> <fold2> ...
#
# Launches one independent python process per fold concurrently, redirecting
# each to its own timestamped log and per-fold pid file. Waits for all to
# finish before returning, so the caller can chain multiple rounds.
#
# Each invocation produces its own experiment_id / results_*.json /
# audit_*.json / progress_*.json files (timestamps differ). Use
# aggregate_v13_results.py afterwards to fuse them into a single LOPO result.

set -u
cd "$(dirname "$0")"

if [ "$#" -eq 0 ]; then
    echo "Usage: $0 <fold1> [fold2 ...]"
    exit 1
fi

TS=$(date +%Y%m%d_%H%M%S)
PYTHON=/root/miniconda3/bin/python
SCRIPT=train_clean_weighted_lopo_v13_two_stage.py
PIDS=()

echo "[$(date +%H:%M:%S)] Launching ${#@} parallel folds: $@"
for fold in "$@"; do
    LOG="train_lopo_v13_two_stage_fold${fold}_${TS}.log"
    PIDFILE="train_lopo_v13_two_stage_fold${fold}.pid"
    nohup "$PYTHON" "$SCRIPT" --fold "$fold" > "$LOG" 2>&1 &
    PID=$!
    PIDS+=($PID)
    echo $PID > "$PIDFILE"
    echo "  fold $fold -> pid=$PID log=$LOG"
    sleep 3
done

echo "[$(date +%H:%M:%S)] All ${#PIDS[@]} workers launched. Waiting..."
EXIT_CODE=0
for pid in "${PIDS[@]}"; do
    if wait "$pid"; then
        echo "[$(date +%H:%M:%S)] pid $pid completed cleanly"
    else
        rc=$?
        EXIT_CODE=$rc
        echo "[$(date +%H:%M:%S)] pid $pid exited with code $rc"
    fi
done

echo "[$(date +%H:%M:%S)] All workers done. Round exit_code=$EXIT_CODE"
exit $EXIT_CODE
