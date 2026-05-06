#!/usr/bin/env python
"""
Progressive unfreezing training script for segmentation.

Usage:
    # Stage 1: Freeze encoder
    python train_seg_progressive.py --stage freeze_all

    # Stage 2: Unfreeze adapters
    python train_seg_progressive.py --stage unfreeze_adapters --resume checkpoints/stage1_best.pth

    # Stage 3: Unfreeze top layers
    python train_seg_progressive.py --stage unfreeze_top_layers --resume checkpoints/stage2_best.pth

    # Stage 4: Unfreeze all
    python train_seg_progressive.py --stage unfreeze_all --resume checkpoints/stage3_best.pth
"""

import sys
import argparse
import json
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.amp import autocast, GradScaler

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(1, str(Path(__file__).parent.parent))

import config_seg as config
from datasets import IVOCTSegDataset
from models import MAESegmenter
from utils import seg_loss, compute_metrics, aggregate_metrics, save_seg_visualization
from utils.progressive_unfreezing import (
    setup_progressive_unfreezing,
    get_recommended_lr,
    get_recommended_epochs
)


def train_one_epoch(model, loader, optimizer, scaler, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    amp_enabled = config.USE_AMP and device.type == 'cuda'

    for batch_idx, batch in enumerate(loader):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad()

        with autocast(device_type='cuda', enabled=amp_enabled):
            logits = model(images)
            loss = seg_loss(
                logits, masks,
                lambda_dice=config.LAMBDA_DICE,
                lambda_bce=config.LAMBDA_BCE,
                loss_mode=config.LOSS_MODE,
                tversky_alpha=config.TVERSKY_ALPHA,
                tversky_beta=config.TVERSKY_BETA,
                focal_tversky_gamma=config.FOCAL_TVERSKY_GAMMA,
            )

        if config.USE_AMP:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if (batch_idx + 1) % 5 == 0:
            print(f"  Batch {batch_idx + 1}/{len(loader)}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / num_batches
    return avg_loss


def evaluate(model, loader, device, threshold):
    """Evaluate on validation set"""
    model.eval()
    metrics_list = []
    amp_enabled = config.USE_AMP and device.type == 'cuda'

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            with autocast(device_type='cuda', enabled=amp_enabled):
                logits = model(images)

            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()

            for i in range(len(images)):
                metrics = compute_metrics(
                    preds[i].cpu().numpy(),
                    masks[i].cpu().numpy()
                )
                metrics_list.append(metrics)

    return aggregate_metrics(metrics_list)


def train_stage(stage, resume_from=None, fold_idx=0, train_patients=None, val_patients=None):
    """Train one stage of progressive unfreezing"""

    print(f"\n{'='*80}")
    print(f"Training Stage: {stage}")
    print(f"Fold: {fold_idx}")
    print(f"{'='*80}\n")

    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

    # Create datasets
    train_dataset = IVOCTSegDataset(
        config.DATA_DIR, train_patients,
        img_size=config.IMG_SIZE,
        roi_crop_ratio=config.ROI_CROP_RATIO,
        augment=True
    )
    val_dataset = IVOCTSegDataset(
        config.DATA_DIR, val_patients,
        img_size=config.IMG_SIZE,
        roi_crop_ratio=config.ROI_CROP_RATIO,
        augment=False
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE,
        shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE,
        shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True
    )

    # Create or load model
    if resume_from and Path(resume_from).exists():
        print(f"Loading checkpoint from {resume_from}")
        checkpoint = torch.load(resume_from, map_location='cpu', weights_only=False)

        # Create model
        model = MAESegmenter(
            config.MAE_CHECKPOINT,
            patch_size=config.PATCH_SIZE,
            freeze_encoder=False,  # Will be set by setup_progressive_unfreezing
            use_adapter=config.USE_ADAPTER,
            adapter_bottleneck=config.ADAPTER_BOTTLENECK,
        ).to(device)

        # Load weights
        model.load_state_dict(checkpoint['model'])
        print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        # Create new model
        model = MAESegmenter(
            config.MAE_CHECKPOINT,
            patch_size=config.PATCH_SIZE,
            freeze_encoder=False,
            use_adapter=config.USE_ADAPTER,
            adapter_bottleneck=config.ADAPTER_BOTTLENECK,
        ).to(device)

    # Setup progressive unfreezing
    setup_progressive_unfreezing(model, stage=stage, num_layers=3)

    # Get recommended hyperparameters
    base_lr = get_recommended_lr(stage)
    epochs = get_recommended_epochs(stage)

    print(f"\nHyperparameters:")
    print(f"  Learning rate: {base_lr}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {config.BATCH_SIZE}")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")

    # Optimizer
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=base_lr,
        weight_decay=config.WEIGHT_DECAY
    )

    # Scheduler with warmup
    warmup_epochs = min(5, epochs // 4)
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler('cuda', enabled=config.USE_AMP and device.type == 'cuda')

    # Training loop
    best_dice = 0
    best_epoch = 0
    best_metrics = None
    patience_counter = 0

    stage_checkpoint_dir = config.SEG_CHECKPOINT_DIR / f"stage_{stage}"
    stage_checkpoint_dir.mkdir(exist_ok=True)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, epoch)
        val_metrics = evaluate(model, val_loader, device, config.EVAL_THRESHOLD)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Metrics: Dice={val_metrics['dice']:.4f}, IoU={val_metrics['iou']:.4f}, "
              f"Precision={val_metrics['precision']:.4f}, Recall={val_metrics['recall']:.4f}")

        scheduler.step()

        # Save best model
        if val_metrics['dice'] > best_dice:
            best_dice = val_metrics['dice']
            best_epoch = epoch
            best_metrics = val_metrics
            patience_counter = 0

            checkpoint_path = stage_checkpoint_dir / f"fold{fold_idx}_best.pth"
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'metrics': val_metrics,
                'stage': stage,
            }, checkpoint_path)
            print(f"✓ New best model saved: {checkpoint_path}")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    print(f"\n{'='*80}")
    print(f"Stage {stage} completed!")
    print(f"Best Dice: {best_dice:.4f} at epoch {best_epoch + 1}")
    print(f"{'='*80}\n")

    return best_metrics, stage_checkpoint_dir / f"fold{fold_idx}_best.pth"


def main():
    parser = argparse.ArgumentParser(description="Progressive unfreezing training")
    parser.add_argument("--stage", type=str, required=True,
                       choices=["freeze_all", "unfreeze_adapters", "unfreeze_top_layers", "unfreeze_all"],
                       help="Training stage")
    parser.add_argument("--resume", type=str, default=None,
                       help="Resume from checkpoint")
    parser.add_argument("--fold", type=int, default=None,
                       help="Specific fold to train (default: all folds)")
    args = parser.parse_args()

    # Get splits
    patients = config.PATIENTS
    if config.SPLIT_MODE == "lopo":
        splits = [(
            [p for p in patients if p != val_patient],
            [val_patient]
        ) for val_patient in patients]
    else:
        raise ValueError(f"Unsupported split mode: {config.SPLIT_MODE}")

    # Train specific fold or all folds
    if args.fold is not None:
        folds_to_train = [args.fold]
    else:
        folds_to_train = range(len(splits))

    all_results = []

    for fold_idx in folds_to_train:
        train_patients, val_patients = splits[fold_idx]
        print(f"\n{'='*80}")
        print(f"Fold {fold_idx}: Val patient = {val_patients[0]}")
        print(f"{'='*80}")

        metrics, checkpoint_path = train_stage(
            stage=args.stage,
            resume_from=args.resume,
            fold_idx=fold_idx,
            train_patients=train_patients,
            val_patients=val_patients
        )

        all_results.append({
            'fold': fold_idx,
            'val_patient': val_patients[0],
            'metrics': metrics,
            'checkpoint': str(checkpoint_path)
        })

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = config.SEG_LOG_DIR / f"results_{args.stage}_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    print(f"\n{'='*80}")
    print(f"Stage {args.stage} - All Folds Summary")
    print(f"{'='*80}")
    avg_dice = np.mean([r['metrics']['dice'] for r in all_results])
    avg_iou = np.mean([r['metrics']['iou'] for r in all_results])
    print(f"Average Dice: {avg_dice:.4f}")
    print(f"Average IoU: {avg_iou:.4f}")
    print(f"Results saved to: {results_file}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
