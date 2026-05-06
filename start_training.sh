#!/bin/bash
# Quick start script for training MAE with adapter tuning on remote server

set -e  # Exit on error

echo "=========================================="
echo "IVOCT MAE Adapter Tuning - Quick Start"
echo "=========================================="

# Check if we're in the right directory
if [ ! -d "seven" ]; then
    echo "Error: Please run this script from the project root directory"
    exit 1
fi

# Activate conda environment
echo ""
echo "[1/5] Activating conda environment..."
source /root/miniconda3/bin/activate
conda activate base

# Check Python and PyTorch
echo ""
echo "[2/5] Checking environment..."
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Install tqdm if not present
echo ""
echo "[3/5] Checking dependencies..."
python -c "import tqdm" 2>/dev/null || pip install tqdm

# Test adapter implementation
echo ""
echo "[4/5] Testing adapter implementation..."
cd seven
python test_adapter.py

# Confirm before training
echo ""
echo "[5/5] Ready to start training!"
echo ""
echo "Configuration:"
echo "  - Dataset: 4,554 images (19 patients)"
echo "  - Mode: Adapter Tuning"
echo "  - Batch size: 64"
echo "  - Epochs: 200"
echo "  - Estimated time: ~16 hours on RTX 5090"
echo ""
read -p "Start training? (y/n) " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Starting training..."
    echo "Logs will be saved to: seven/logs_v2/"
    echo "Checkpoints will be saved to: seven/checkpoints_v2/"
    echo ""
    python train_mae_v2.py
else
    echo "Training cancelled."
    exit 0
fi
