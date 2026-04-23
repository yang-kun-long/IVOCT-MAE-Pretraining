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

sys.path.insert(0, str(Path(__file__).parent))        # seven/seg/ first
sys.path.insert(1, str(Path(__file__).parent.parent))  # seven/ second

import config_seg as config
from datasets import IVOCTSegDataset
from models import MAESegmenter
from utils import seg_loss, compute_metrics, aggregate_metrics, save_seg_visualization


def get_splits(split_mode, holdout_patient):
    """
    Generate train/val splits.

    Returns:
        List of (train_patients, val_patients) tuples
    """
    patients = config.PATIENTS

    if split_mode == "lopo":
        # Leave-one-patient-out cross-validation
        splits = []
        for val_patient in patients:
            train_patients = [p for p in patients if p != val_patient]
            splits.append((train_patients, [val_patient]))
        return splits
    elif split_mode == "single_holdout":
        # Single holdout split
        train_patients = [p for p in patients if p != holdout_patient]
        return [(train_patients, [holdout_patient])]
    else:
        raise ValueError(f"Unknown split_mode: {split_mode}")


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
                lambda_bce=config.LAMBDA_BCE
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


def evaluate(model, loader, device, threshold, vis_dir=None, epoch=None):
    """Evaluate on validation set"""
    model.eval()
    metrics_list = []
    amp_enabled = config.USE_AMP and device.type == 'cuda'

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            with autocast(device_type='cuda', enabled=amp_enabled):
                logits = model(images)

            metrics = compute_metrics(logits, masks, threshold=threshold)
            metrics_list.append(metrics)

            if batch_idx == 0:
                probs = torch.sigmoid(logits)
                pred_ratio = (probs > threshold).float().mean()
                print(
                    f"  [diag] prob mean={probs.mean():.3f} "
                    f"max={probs.max():.3f} pred>{threshold:.1f}={pred_ratio:.3f}"
                )

            # Save visualization for first batch
            if batch_idx == 0 and vis_dir is not None and epoch is not None:
                vis_path = vis_dir / f"epoch_{epoch:03d}.png"
                save_seg_visualization(images, logits, masks, vis_path)

    # Aggregate metrics
    agg_metrics = aggregate_metrics(metrics_list)
    return agg_metrics


def train_fold(fold_idx, train_patients, val_patients, device):
    """Train a single fold"""
    print(f"\n{'='*70}")
    print(f"Fold {fold_idx + 1}: Train={train_patients}, Val={val_patients}")
    print(f"{'='*70}")

    # Create datasets
    train_dataset = IVOCTSegDataset(
        config.DATA_DIR, train_patients,
        img_size=config.IMG_SIZE,
        crop_ratio=config.ROI_CROP_RATIO,
        is_train=True
    )
    val_dataset = IVOCTSegDataset(
        config.DATA_DIR, val_patients,
        img_size=config.IMG_SIZE,
        crop_ratio=config.ROI_CROP_RATIO,
        is_train=False
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE,
        shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE,
        shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True
    )

    # Create model
    model = MAESegmenter(
        config.MAE_CHECKPOINT,
        freeze_encoder=config.FREEZE_ENCODER
    ).to(device)

    # Optimizer with differential learning rates
    encoder_params = []
    decoder_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'encoder' in name:
                encoder_params.append(param)
            else:
                decoder_params.append(param)

    optimizer = AdamW([
        {'params': encoder_params, 'lr': config.BASE_LR * config.ENCODER_LR_SCALE},
        {'params': decoder_params, 'lr': config.BASE_LR}
    ], weight_decay=config.WEIGHT_DECAY)

    # Scheduler with warmup
    def lr_lambda(epoch):
        if epoch < config.WARMUP_EPOCHS:
            return (epoch + 1) / config.WARMUP_EPOCHS
        else:
            return 0.5 * (1 + np.cos(np.pi * (epoch - config.WARMUP_EPOCHS) / (config.EPOCHS - config.WARMUP_EPOCHS)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    scaler = GradScaler('cuda', enabled=config.USE_AMP and device.type == 'cuda')

    # Training loop
    best_dice = 0
    best_epoch = 0
    best_metrics = None
    patience_counter = 0
    fold_vis_dir = config.SEG_VIS_DIR / f"fold_{fold_idx}"
    fold_vis_dir.mkdir(exist_ok=True)

    for epoch in range(config.EPOCHS):
        print(f"\nEpoch {epoch + 1}/{config.EPOCHS}")
        print(f"LR: encoder={optimizer.param_groups[0]['lr']:.2e}, decoder={optimizer.param_groups[1]['lr']:.2e}")

        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, epoch)
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            threshold=config.EVAL_THRESHOLD,
            vis_dir=fold_vis_dir,
            epoch=epoch,
        )

        scheduler.step()

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Dice: {val_metrics['dice_mean']:.4f} ± {val_metrics['dice_std']:.4f}")
        print(f"Val IoU:  {val_metrics['iou_mean']:.4f} ± {val_metrics['iou_std']:.4f}")

        # Save best model
        if val_metrics['dice_mean'] > best_dice:
            best_dice = val_metrics['dice_mean']
            best_epoch = epoch
            best_metrics = val_metrics
            patience_counter = 0
            ckpt_path = config.SEG_CHECKPOINT_DIR / f"seg_fold{fold_idx}_best.pth"
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'metrics': val_metrics,
                'fold': fold_idx,
                'train_patients': train_patients,
                'val_patients': val_patients
            }, ckpt_path)
            print(f"  → Saved best model (Dice={best_dice:.4f})")
        else:
            patience_counter += 1

        if epoch + 1 >= config.MIN_EPOCHS and patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(
                f"  → Early stopping at epoch {epoch + 1} "
                f"(best Dice={best_dice:.4f}, best epoch={best_epoch + 1})"
            )
            break

    print(f"\nFold {fold_idx + 1} finished. Best Dice: {best_dice:.4f} at epoch {best_epoch + 1}")
    return best_dice, best_metrics if best_metrics is not None else val_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default=None,
                        help='Split mode: lopo or single_holdout (overrides config)')
    parser.add_argument('--holdout', type=str, default=None,
                        help='Holdout patient for single_holdout mode (overrides config)')
    args = parser.parse_args()

    # Override config if specified
    split_mode = args.split if args.split else config.SPLIT_MODE
    holdout_patient = args.holdout if args.holdout else config.HOLDOUT_PATIENT

    print(f"Split mode: {split_mode}")
    if split_mode == "single_holdout":
        print(f"Holdout patient: {holdout_patient}")

    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.SEED)

    # Get splits
    splits = get_splits(split_mode, holdout_patient)
    print(f"Total folds: {len(splits)}")

    # Train each fold
    fold_results = []
    for fold_idx, (train_patients, val_patients) in enumerate(splits):
        best_dice, val_metrics = train_fold(fold_idx, train_patients, val_patients, device)
        fold_results.append({
            'fold': fold_idx,
            'train_patients': train_patients,
            'val_patients': val_patients,
            'best_dice': best_dice,
            'metrics': val_metrics
        })

    # Summary
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")

    all_dice = [r['best_dice'] for r in fold_results]
    mean_dice = np.mean(all_dice)
    std_dice = np.std(all_dice)

    print(f"Mean Dice across {len(splits)} fold(s): {mean_dice:.4f} ± {std_dice:.4f}")
    print("\nPer-fold results:")
    for r in fold_results:
        print(f"  Fold {r['fold'] + 1} (val={r['val_patients']}): Dice={r['best_dice']:.4f}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = config.SEG_LOG_DIR / f"results_{split_mode}_{timestamp}.json"
    with open(results_path, 'w') as f:
        json.dump({
            'split_mode': split_mode,
            'holdout_patient': holdout_patient if split_mode == 'single_holdout' else None,
            'mean_dice': float(mean_dice),
            'std_dice': float(std_dice),
            'fold_results': fold_results
        }, f, indent=2)

    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
