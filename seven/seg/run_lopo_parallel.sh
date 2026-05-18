#!/bin/bash
# Generic parallel launcher for LOPO single-fold trainings.
#
# Usage:
#   bash run_lopo_parallel.sh <python_script> <fold1> [fold2 ...]
#
# Behaviour:
#   - Generates a BATCH_ID from script name + timestamp (or honours the
#     existing $BATCH_ID env var) and exports it so each worker's MonitorRun
#     writes it into its progress/result JSON. The monitor's /api/batches
#     endpoint groups all workers with the same BATCH_ID.
#   - Launches one nohup python worker per fold, redirecting stdout/stderr
#     to a per-fold timestamped log.
#   - Waits for all workers to complete, returning non-zero if any failed.
#
# Example:
#   bash run_lopo_parallel.sh train_clean_weighted_lopo_lipid_warm_sentinel.py 12 7 11 5

set -u
cd "$(dirname "$0")"

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <python_script_name> <fold1> [fold2 ...]"
    echo "Example: bash $0 train_clean_weighted_lopo_lipid_warm_sentinel.py 12 7 11 5"
    exit 1
fi

SCRIPT="$1"
shift

if [ ! -f "$SCRIPT" ]; then
    echo "Script not found: $SCRIPT"
    exit 2
fi

TS=$(date +%Y%m%d_%H%M%S)
SCRIPT_STEM="${SCRIPT%.py}"
BATCH_ID="${BATCH_ID:-${SCRIPT_STEM}_${TS}}"
export BATCH_ID

PYTHON=/root/miniconda3/bin/python
PIDS=()

echo "[$(date +%H:%M:%S)] BATCH_ID=$BATCH_ID"
echo "[$(date +%H:%M:%S)] script=$SCRIPT folds=$@ launching ${#@} workers..."
for fold in "$@"; do
    LOG="${SCRIPT_STEM}_fold${fold}_${TS}.log"
    PIDFILE="${SCRIPT_STEM}_fold${fold}.pid"
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
        echo "[$(date +%H:%M:%S)] pid $pid done"
    else
        rc=$?
        EXIT_CODE=$rc
        echo "[$(date +%H:%M:%S)] pid $pid exit=$rc"
    fi
done

echo "[$(date +%H:%M:%S)] Round complete. BATCH_ID=$BATCH_ID exit=$EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "[$(date +%H:%M:%S)] Aggregating batch into synthetic v12-style result..."
    "$PYTHON" aggregate_batch.py --batch-id "$BATCH_ID" || \
        echo "[$(date +%H:%M:%S)] WARN: aggregator failed (workers still finished OK)"
fi

exit $EXIT_CODE
