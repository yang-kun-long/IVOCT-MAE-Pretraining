#!/bin/bash
# Monitor training progress

LOG_FILE="/root/CN_seg/seven/logs_v2/train_20260504_153219.log"

echo "=========================================="
echo "MAE Training Monitor"
echo "=========================================="
echo ""

# Check if training is running
if ps aux | grep -q "[p]ython train_mae_v2.py"; then
    echo "✓ Training is running"
else
    echo "✗ Training is not running"
    exit 1
fi

echo ""
echo "Latest progress:"
echo "----------------------------------------"

# Show last 20 lines with epoch info
tail -100 "$LOG_FILE" | grep -E "(Epoch|total_loss)" | tail -20

echo ""
echo "----------------------------------------"
echo "GPU Status:"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader

echo ""
echo "To see real-time logs:"
echo "  tail -f $LOG_FILE"
