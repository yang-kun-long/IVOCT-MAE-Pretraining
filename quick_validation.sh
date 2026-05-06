#!/bin/bash
# Quick validation: Run 1-3 folds to verify the approach

echo "=========================================="
echo "Quick Validation (1-3 Folds)"
echo "=========================================="
echo ""

# Activate conda
source /root/miniconda3/bin/activate
cd /root/CN_seg/seven/seg

# Check prerequisites
echo "[1/2] Checking prerequisites..."
if [ ! -f "/root/CN_seg/seven/checkpoints_v2/mae_v2_best.pth" ]; then
    echo "✗ MAE checkpoint not found!"
    exit 1
fi
echo "✓ MAE checkpoint found"
echo ""

# Show configuration
echo "[2/2] Configuration:"
python -c "
import config_seg as config
print(f'  - Patients: {len(config.PATIENTS)} ({config.PATIENTS[0]}-{config.PATIENTS[-1]})')
print(f'  - Patch size: {config.PATCH_SIZE}')
print(f'  - Freeze encoder: {config.FREEZE_ENCODER}')
print(f'  - Epochs: {config.EPOCHS}')
print(f'  - Batch size: {config.BATCH_SIZE}')
"
echo ""

# Ask how many folds to run
echo "How many folds to run for quick validation?"
echo "  1 - Run fold 0 only (~10-15 min)"
echo "  3 - Run folds 0-2 (~30-45 min)"
echo "  5 - Run folds 0-4 (~50-75 min)"
echo ""
read -p "Enter number (1/3/5): " NUM_FOLDS

case $NUM_FOLDS in
    1)
        echo ""
        echo "=========================================="
        echo "Running Fold 0 (Val: P001)"
        echo "=========================================="
        python train_seg.py --fold 0
        ;;
    3)
        echo ""
        echo "=========================================="
        echo "Running Folds 0-2"
        echo "=========================================="
        for i in 0 1 2; do
            echo ""
            echo "--- Fold $i ---"
            python train_seg.py --fold $i
        done
        ;;
    5)
        echo ""
        echo "=========================================="
        echo "Running Folds 0-4"
        echo "=========================================="
        for i in 0 1 2 3 4; do
            echo ""
            echo "--- Fold $i ---"
            python train_seg.py --fold $i
        done
        ;;
    *)
        echo "Invalid input. Exiting."
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Quick Validation Completed!"
echo "=========================================="
echo ""
echo "Check results:"
echo "  - Checkpoints: seven/seg/checkpoints/"
echo "  - Logs: seven/seg/logs/"
echo ""
echo "To run all 19 folds:"
echo "  cd /root/CN_seg"
echo "  bash start_segmentation.sh"
echo ""
