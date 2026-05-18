#!/bin/bash
# Sequentially launch batches of N folds in parallel, sharing one BATCH_ID.
#
# Usage:
#   bash run_lopo_chunked.sh <max_parallel> <python_script> <fold1> [fold2 ...]
#
# Example (Round 3 full_remaining10, 4-way parallel, 10 folds = 3 chunks):
#   bash run_lopo_chunked.sh 4 train_clean_weighted_lopo_lipid_warm_full_remaining10.py \
#       0 1 2 3 4 6 10 15 16 17
#
# Behaviour:
#   - Picks one BATCH_ID up-front (env var or auto), exports for every chunk
#   - Chunks the fold list into groups of <max_parallel>, runs each chunk
#     via `run_lopo_parallel.sh` (which already handles per-worker logs,
#     pid files, and post-chunk aggregator invocation)
#   - Because all chunks share BATCH_ID, the per-chunk aggregator call
#     overwrites the shared synthetic each time, so the monitor card grows
#     fold-by-fold instead of producing N separate batches
#   - Final explicit aggregation at the end as a safety net

set -u
cd "$(dirname "$0")"

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <max_parallel> <python_script> <fold1> [fold2 ...]"
    exit 1
fi

MAX_PARALLEL=$1
SCRIPT=$2
shift 2

TS=$(date +%Y%m%d_%H%M%S)
SCRIPT_STEM="${SCRIPT%.py}"
BATCH_ID="${BATCH_ID:-${SCRIPT_STEM}_${TS}}"
export BATCH_ID

PYTHON=/root/miniconda3/bin/python
FOLDS=("$@")
TOTAL=${#FOLDS[@]}

echo "[$(date +%H:%M:%S)] BATCH_ID=$BATCH_ID"
echo "[$(date +%H:%M:%S)] Script=$SCRIPT total_folds=$TOTAL max_parallel=$MAX_PARALLEL"

i=0
chunk=1
while [ $i -lt $TOTAL ]; do
    SLICE=("${FOLDS[@]:$i:$MAX_PARALLEL}")
    echo ""
    echo "============================================================"
    echo "[$(date +%H:%M:%S)] Chunk $chunk: folds=${SLICE[*]}"
    echo "============================================================"
    bash run_lopo_parallel.sh "$SCRIPT" "${SLICE[@]}"
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "[$(date +%H:%M:%S)] Chunk $chunk exited with code $rc; continuing with remaining chunks"
    fi
    i=$((i + MAX_PARALLEL))
    chunk=$((chunk + 1))
done

echo ""
echo "[$(date +%H:%M:%S)] All chunks done. Final aggregation for BATCH_ID=$BATCH_ID"
"$PYTHON" aggregate_batch.py --batch-id "$BATCH_ID" || \
    echo "[$(date +%H:%M:%S)] WARN: final aggregator failed (per-chunk aggregations already ran)"

echo "[$(date +%H:%M:%S)] Done."
