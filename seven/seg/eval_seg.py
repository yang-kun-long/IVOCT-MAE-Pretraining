import sys
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config_seg as config
from datasets import IVOCTSegDataset
from models import MAESegmenter
from utils import compute_metrics, aggregate_metrics, save_seg_visualization


def evaluate_checkpoint(checkpoint_path, device):
    """
    Evaluate a single checkpoint on its validation set.

    Args:
        checkpoint_path: path to checkpoint file
        device: torch device
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract metadata
    fold_idx = ckpt.get('fold', 0)
    val_patients = ckpt['val_patients']
    train_patients = ckpt['train_patients']
    saved_metrics = ckpt.get('metrics', {})

    print(f"Fold {fold_idx + 1}")
    print(f"  Train patients: {train_patients}")
    print(f"  Val patients: {val_patients}")
    print(f"  Saved metrics: Dice={saved_metrics.get('dice_mean', 'N/A'):.4f}")

    # Create validation dataset
    val_dataset = IVOCTSegDataset(
        config.DATA_DIR, val_patients,
        img_size=config.IMG_SIZE,
        crop_ratio=config.ROI_CROP_RATIO,
        is_train=False
    )

    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE,
        shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True
    )

    # Create model and load weights
    model = MAESegmenter(
        config.MAE_CHECKPOINT,
        freeze_encoder=config.FREEZE_ENCODER
    ).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    print(f"\nEvaluating on {len(val_dataset)} samples...")

    # Evaluate
    metrics_list = []
    vis_dir = config.SEG_VIS_DIR / f"eval_fold{fold_idx}"
    vis_dir.mkdir(exist_ok=True)

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            with autocast(enabled=config.USE_AMP):
                logits = model(images)

            metrics = compute_metrics(logits, masks)
            metrics_list.append(metrics)

            # Save visualization for all batches
            vis_path = vis_dir / f"batch_{batch_idx:03d}.png"
            save_seg_visualization(images, logits, masks, vis_path)

    # Aggregate metrics
    agg_metrics = aggregate_metrics(metrics_list)

    print(f"\n{'='*70}")
    print("EVALUATION RESULTS")
    print(f"{'='*70}")
    print(f"Dice:        {agg_metrics['dice_mean']:.4f} ± {agg_metrics['dice_std']:.4f}")
    print(f"IoU:         {agg_metrics['iou_mean']:.4f} ± {agg_metrics['iou_std']:.4f}")
    print(f"Sensitivity: {agg_metrics['sensitivity_mean']:.4f} ± {agg_metrics['sensitivity_std']:.4f}")
    print(f"Specificity: {agg_metrics['specificity_mean']:.4f} ± {agg_metrics['specificity_std']:.4f}")
    print(f"{'='*70}")

    # Save results
    results_path = vis_dir / "eval_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            'checkpoint': str(checkpoint_path),
            'fold': fold_idx,
            'train_patients': train_patients,
            'val_patients': val_patients,
            'metrics': agg_metrics
        }, f, indent=2)

    print(f"\nResults saved to: {results_path}")
    print(f"Visualizations saved to: {vis_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file (relative to SEG_CHECKPOINT_DIR or absolute)')
    args = parser.parse_args()

    # Resolve checkpoint path
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = config.SEG_CHECKPOINT_DIR / ckpt_path

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    evaluate_checkpoint(ckpt_path, device)


if __name__ == "__main__":
    main()
