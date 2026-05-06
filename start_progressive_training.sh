#!/bin/bash
# Progressive unfreezing training pipeline for segmentation

echo "=========================================="
echo "Progressive Unfreezing Training Pipeline"
echo "=========================================="
echo ""

# Activate conda
source /root/miniconda3/bin/activate
cd /root/CN_seg/seven/seg

# Check prerequisites
echo "[1/5] Checking prerequisites..."
if [ ! -f "/root/CN_seg/seven/checkpoints_v2/mae_v2_best.pth" ]; then
    echo "✗ MAE checkpoint not found!"
    exit 1
fi
echo "✓ MAE checkpoint found"
echo ""

# Stage 1: Freeze encoder (train decoder only)
echo "=========================================="
echo "Stage 1: Freeze Encoder"
echo "=========================================="
echo "Training decoder only with frozen encoder"
echo "Expected time: ~3-5 hours (19 folds)"
echo ""
read -p "Start Stage 1? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python train_seg_progressive.py --stage freeze_all
    echo "✓ Stage 1 completed"
else
    echo "Skipped Stage 1"
fi
echo ""

# Stage 2: Unfreeze adapters
echo "=========================================="
echo "Stage 2: Unfreeze Adapters"
echo "=========================================="
echo "Fine-tuning adapters + decoder"
echo "Expected time: ~2-3 hours (19 folds)"
echo ""
read -p "Start Stage 2? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Find best checkpoint from stage 1
    STAGE1_DIR="/root/CN_seg/seven/seg/checkpoints/stage_freeze_all"
    if [ -d "$STAGE1_DIR" ]; then
        python train_seg_progressive.py --stage unfreeze_adapters --resume "$STAGE1_DIR/fold0_best.pth"
        echo "✓ Stage 2 completed"
    else
        echo "✗ Stage 1 checkpoints not found. Please run Stage 1 first."
    fi
else
    echo "Skipped Stage 2"
fi
echo ""

# Stage 3: Unfreeze top layers
echo "=========================================="
echo "Stage 3: Unfreeze Top Layers"
echo "=========================================="
echo "Fine-tuning top 3 layers + adapters + decoder"
echo "Expected time: ~2-3 hours (19 folds)"
echo ""
read -p "Start Stage 3? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    STAGE2_DIR="/root/CN_seg/seven/seg/checkpoints/stage_unfreeze_adapters"
    if [ -d "$STAGE2_DIR" ]; then
        python train_seg_progressive.py --stage unfreeze_top_layers --resume "$STAGE2_DIR/fold0_best.pth"
        echo "✓ Stage 3 completed"
    else
        echo "✗ Stage 2 checkpoints not found. Please run Stage 2 first."
    fi
else
    echo "Skipped Stage 3"
fi
echo ""

# Stage 4: Unfreeze all (optional)
echo "=========================================="
echo "Stage 4: Unfreeze All (Optional)"
echo "=========================================="
echo "Full model fine-tuning with very small LR"
echo "Expected time: ~1-2 hours (19 folds)"
echo ""
read -p "Start Stage 4? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    STAGE3_DIR="/root/CN_seg/seven/seg/checkpoints/stage_unfreeze_top_layers"
    if [ -d "$STAGE3_DIR" ]; then
        python train_seg_progressive.py --stage unfreeze_all --resume "$STAGE3_DIR/fold0_best.pth"
        echo "✓ Stage 4 completed"
    else
        echo "✗ Stage 3 checkpoints not found. Please run Stage 3 first."
    fi
else
    echo "Skipped Stage 4"
fi
echo ""

# Summary
echo "=========================================="
echo "Training Pipeline Completed!"
echo "=========================================="
echo ""
echo "Results saved in:"
echo "  - Checkpoints: seven/seg/checkpoints/stage_*/"
echo "  - Logs: seven/seg/logs/"
echo ""
echo "To evaluate results:"
echo "  cd /root/CN_seg/seven/seg"
echo "  python eval_seg.py"
echo ""
