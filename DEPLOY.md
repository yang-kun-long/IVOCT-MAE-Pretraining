# IVOCT Segmentation Deployment Guide

## Quick Start

### 1. Fast Validation (Single Holdout)
Test the pipeline on one fold (~30-60 minutes):
```bash
cd seven/seg
python train_seg.py --split single_holdout --holdout P004
```

### 2. Full Evaluation (LOPO-CV)
Run 4-fold cross-validation (~2-4 hours):
```bash
python train_seg.py --split lopo
```

## Outputs

### Checkpoints
- `seg_checkpoints/fold_X_best.pth` - Best model for each fold (by validation Dice)
- `seg_checkpoints/fold_X_last.pth` - Last epoch checkpoint

### Logs
- `seg_logs/fold_X_train.log` - Training logs per fold
- `seg_logs/lopo_summary.json` - Aggregated metrics across all folds

### Visualizations
- `seg_vis/fold_X_epoch_Y.png` - 3-column layout (image | GT | prediction)
- Generated every 10 epochs during training

## Evaluation

After training, evaluate a specific fold:
```bash
python eval_seg.py --fold 0 --checkpoint seg_checkpoints/fold_0_best.pth
```

Metrics saved to `seg_logs/fold_0_eval.json`:
- Dice coefficient
- IoU (Intersection over Union)
- Sensitivity (Recall)
- Specificity

## Configuration

Edit `config_seg.py` to adjust:
- Learning rate (default: 1e-4)
- Batch size (default: 8)
- Epochs (default: 100)
- Loss weights (Dice: 0.7, BCE: 0.3)

## Data Split

**LOPO-CV** (Leave-One-Patient-Out):
- Fold 0: Train on P001,P002,P003 | Val on P004 (11 images)
- Fold 1: Train on P001,P002,P004 | Val on P003 (8 images)
- Fold 2: Train on P001,P003,P004 | Val on P002 (14 images)
- Fold 3: Train on P002,P003,P004 | Val on P001 (10 images)

Total: 43 annotated images across 4 patients

## Requirements

- PyTorch with CUDA support
- Pretrained MAE checkpoint at `results/pretrain/v4/mae_v2_best.pth`
- Dataset at `DATA/` with annotations in `DATA/annotations/`
