#!/bin/bash
# Start segmentation training with pretrained MAE encoder

echo "=========================================="
echo "IVOCT Segmentation Training"
echo "=========================================="
echo ""

# Activate conda
source /root/miniconda3/bin/activate

# Check environment
echo "[1/3] Checking environment..."
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
fi
echo ""

# Check MAE checkpoint
echo "[2/3] Checking MAE checkpoint..."
if [ -f "/root/CN_seg/seven/checkpoints_v2/mae_v2_best.pth" ]; then
    echo "✓ MAE checkpoint found"
    ls -lh /root/CN_seg/seven/checkpoints_v2/mae_v2_best.pth
else
    echo "✗ MAE checkpoint not found!"
    echo "Please ensure pretraining is completed first."
    exit 1
fi
echo ""

# Show configuration
echo "[3/3] Configuration:"
cd /root/CN_seg/seven/seg
python -c "
import config_seg as config
print(f'  - Patients: {len(config.PATIENTS)} ({config.PATIENTS[0]}-{config.PATIENTS[-1]})')
print(f'  - Split mode: {config.SPLIT_MODE}')
print(f'  - Patch size: {config.PATCH_SIZE}')
print(f'  - Use adapter: {config.USE_ADAPTER}')
print(f'  - Freeze encoder: {config.FREEZE_ENCODER}')
print(f'  - Epochs: {config.EPOCHS}')
print(f'  - Batch size: {config.BATCH_SIZE}')
print(f'  - Learning rate: {config.BASE_LR}')
print(f'  - Loss mode: {config.LOSS_MODE}')
"
echo ""

echo "=========================================="
echo "Starting segmentation training..."
echo "=========================================="
echo ""

# Start training
cd /root/CN_seg/seven/seg
python train_seg.py

echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="
