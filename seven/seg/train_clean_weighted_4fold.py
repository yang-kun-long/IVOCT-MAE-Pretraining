"""Clean weighted 4-fold segmentation training.

This entry point excludes the exact P010/P011 duplicate by keeping P010 and
dropping P011, validates the split before training, and applies capped sample
weights during training. Validation metrics remain unweighted.
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(1, str(Path(__file__).parent.parent))

import config_seg as config
from datasets import IVOCTSegDataset
from models import MAESegmenter
from utils import compute_metrics, aggregate_metrics, save_seg_visualization, write_final_result
from utils.progress_tracker import ProgressTracker


EXCLUDED_PATIENTS = ["P011"]
DUPLICATE_GROUPS = [["P010", "P011"]]

CLEAN_STRATIFIED_FOLDS = [
    {
        "fold": 0,
        "val_patients": ["P014", "P002", "P007", "P013"],
    },
    {
        "fold": 1,
        "val_patients": ["P010", "P003", "P018", "P005", "P015"],
    },
    {
        "fold": 2,
        "val_patients": ["P008", "P004", "P012", "P019"],
    },
    {
        "fold": 3,
        "val_patients": ["P009", "P006", "P016", "P001", "P017"],
    },
]


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def discover_patients(data_dir=None):
    data_dir = Path(data_dir if data_dir is not None else config.DATA_DIR)
    return sorted([p.name for p in data_dir.iterdir() if p.is_dir() and p.name.startswith("P")])


def clean_patients(all_patients=None, excluded=None):
    all_patients = list(all_patients if all_patients is not None else discover_patients())
    excluded = set(excluded if excluded is not None else EXCLUDED_PATIENTS)
    return [p for p in all_patients if p not in excluded]


def build_clean_folds(all_patients=None):
    patients = clean_patients(all_patients)
    folds = []
    for fold in CLEAN_STRATIFIED_FOLDS:
        val_patients = list(fold["val_patients"])
        train_patients = [p for p in patients if p not in val_patients]
        folds.append({
            "fold": fold["fold"],
            "train_patients": train_patients,
            "val_patients": val_patients,
        })
    validate_folds(folds, patients=patients, duplicate_groups=DUPLICATE_GROUPS)
    return folds


def validate_folds(folds, patients, duplicate_groups):
    patient_set = set(patients)
    val_seen = []
    for fold in folds:
        train = set(fold["train_patients"])
        val = set(fold["val_patients"])
        if train & val:
            raise ValueError(f"Fold {fold['fold']} has train/val overlap: {sorted(train & val)}")
        unknown = (train | val) - patient_set
        if unknown:
            raise ValueError(f"Fold {fold['fold']} contains unknown/excluded patients: {sorted(unknown)}")
        missing = patient_set - (train | val)
        if missing:
            raise ValueError(f"Fold {fold['fold']} is missing patients: {sorted(missing)}")
        for group in duplicate_groups:
            members = [p for p in group if p in train or p in val]
            if not members:
                continue
            in_train = [p for p in members if p in train]
            in_val = [p for p in members if p in val]
            if in_train and in_val:
                raise ValueError(f"Duplicate group split across train/val: {group}")
        val_seen.extend(fold["val_patients"])
    if sorted(val_seen) != sorted(patient_set):
        raise ValueError("Validation folds must cover each clean patient exactly once")
    if len(val_seen) != len(set(val_seen)):
        raise ValueError("A clean patient appears in validation more than once")


def mask_foreground_ratio(mask_path):
    from PIL import Image

    mask = np.array(Image.open(mask_path).convert("L"))
    return float((mask > 0).mean())


def collect_patient_stats(data_dir, patients):
    data_dir = Path(data_dir)
    stats = {}
    for patient in patients:
        mask_dir = data_dir / patient / "mask"
        samples = []
        ratios = []
        for mask_path in sorted(mask_dir.glob("*_mask.png")):
            image_name = mask_path.stem.replace("_mask", "") + ".jpg"
            image_path = data_dir / patient / "Data" / image_name
            if not image_path.exists():
                continue
            ratio = mask_foreground_ratio(mask_path)
            samples.append({
                "image": str(image_path),
                "mask": str(mask_path),
                "foreground_ratio": ratio,
            })
            ratios.append(ratio)
        stats[patient] = {
            "num_masks": len(samples),
            "foreground_mean": float(np.mean(ratios)) if ratios else 0.0,
            "foreground_min": float(np.min(ratios)) if ratios else 0.0,
            "foreground_max": float(np.max(ratios)) if ratios else 0.0,
            "samples": samples,
        }
    return stats


def compute_sample_weights(patient_stats, patients, min_weight=0.5, max_weight=2.0):
    counts = np.array([patient_stats[p]["num_masks"] for p in patients if patient_stats[p]["num_masks"] > 0])
    foregrounds = np.array([
        sample["foreground_ratio"]
        for p in patients
        for sample in patient_stats[p]["samples"]
        if sample["foreground_ratio"] > 0
    ])
    median_count = float(np.median(counts)) if len(counts) else 1.0
    median_fg = float(np.median(foregrounds)) if len(foregrounds) else 1.0

    weights = {}
    raw_values = []
    for patient in patients:
        count = max(patient_stats[patient]["num_masks"], 1)
        patient_factor = np.sqrt(median_count / count)
        patient_factor = float(np.clip(patient_factor, 0.75, 1.50))
        for sample in patient_stats[patient]["samples"]:
            fg = max(sample["foreground_ratio"], 1e-8)
            foreground_factor = np.sqrt(fg / median_fg)
            foreground_factor = float(np.clip(foreground_factor, 0.70, 1.30))
            raw = float(np.clip(patient_factor * foreground_factor, min_weight, max_weight))
            weights[sample["image"]] = raw
            raw_values.append(raw)

    mean_weight = float(np.mean(raw_values)) if raw_values else 1.0
    if mean_weight <= 0:
        mean_weight = 1.0
    return {path: float(weight / mean_weight) for path, weight in weights.items()}


def weighted_dice_bce_loss(pred_logit, target, weights):
    pred = torch.sigmoid(pred_logit)
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    dice_loss = 1.0 - ((2.0 * intersection + 1e-6) / (union + 1e-6))
    bce = F.binary_cross_entropy_with_logits(pred_logit, target, reduction="none")
    bce = bce.view(bce.size(0), -1).mean(dim=1)
    per_sample = config.LAMBDA_DICE * dice_loss + config.LAMBDA_BCE * bce
    return (per_sample * weights).sum() / weights.sum().clamp_min(1e-6)


def train_one_epoch_weighted(model, loader, optimizer, scaler, device, sample_weights):
    model.train()
    total_loss = 0.0
    num_batches = 0
    amp_enabled = config.USE_AMP and device.type == "cuda"

    for batch_idx, batch in enumerate(loader):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        paths = batch["path"]
        weights = torch.tensor(
            [sample_weights.get(path, 1.0) for path in paths],
            dtype=torch.float32,
            device=device,
        )

        optimizer.zero_grad()
        with autocast(device_type="cuda", enabled=amp_enabled):
            logits = model(images)
            loss = weighted_dice_bce_loss(logits, masks, weights)

        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        num_batches += 1
        if (batch_idx + 1) % 5 == 0:
            print(f"  Batch {batch_idx + 1}/{len(loader)}, Weighted Loss: {loss.item():.4f}")

    return total_loss / max(num_batches, 1)


def evaluate(model, loader, device, threshold, vis_dir=None, epoch=None):
    model.eval()
    metrics_list = []
    amp_enabled = config.USE_AMP and device.type == "cuda"

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            with autocast(device_type="cuda", enabled=amp_enabled):
                logits = model(images)
            metrics_list.append(compute_metrics(logits, masks, threshold=threshold))
            if batch_idx == 0 and vis_dir is not None and epoch is not None:
                save_seg_visualization(images, logits, masks, vis_dir / f"epoch_{epoch:03d}.png")

    return aggregate_metrics(metrics_list)


def make_optimizer(model):
    encoder_params = []
    decoder_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "encoder" in name:
                encoder_params.append(param)
            else:
                decoder_params.append(param)
    return AdamW([
        {"params": encoder_params, "lr": config.BASE_LR * config.ENCODER_LR_SCALE},
        {"params": decoder_params, "lr": config.BASE_LR},
    ], weight_decay=config.WEIGHT_DECAY)


def make_scheduler(optimizer, epochs):
    def lr_lambda(epoch):
        if epoch < config.WARMUP_EPOCHS:
            return (epoch + 1) / config.WARMUP_EPOCHS
        return 0.5 * (1 + np.cos(np.pi * (epoch - config.WARMUP_EPOCHS) / (epochs - config.WARMUP_EPOCHS)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_fold(fold_info, device, patient_stats, sample_weights, tracker=None, max_epochs=None):
    fold_idx = fold_info["fold"]
    train_patients = fold_info["train_patients"]
    val_patients = fold_info["val_patients"]
    epochs = max_epochs or config.EPOCHS

    print(f"\n{'=' * 80}")
    print(f"Clean weighted fold {fold_idx}: Train={train_patients}, Val={val_patients}")
    print(f"{'=' * 80}")

    if tracker:
        tracker.start_fold(fold_idx, epochs, train_patients, val_patients)

    train_dataset = IVOCTSegDataset(config.DATA_DIR, train_patients, config.IMG_SIZE, config.ROI_CROP_RATIO, True)
    val_dataset = IVOCTSegDataset(config.DATA_DIR, val_patients, config.IMG_SIZE, config.ROI_CROP_RATIO, False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    fold_weights = [sample_weights.get(str(sample["image"]), 1.0) for sample in train_dataset.samples]
    print(
        f"Train samples={len(train_dataset)}, Val samples={len(val_dataset)}, "
        f"weight min/mean/max={min(fold_weights):.3f}/{np.mean(fold_weights):.3f}/{max(fold_weights):.3f}"
    )

    model = MAESegmenter(
        config.MAE_CHECKPOINT,
        patch_size=config.PATCH_SIZE,
        freeze_encoder=config.FREEZE_ENCODER,
        use_adapter=config.USE_ADAPTER,
        adapter_bottleneck=config.ADAPTER_BOTTLENECK,
    ).to(device)
    optimizer = make_optimizer(model)
    scheduler = make_scheduler(optimizer, epochs)
    scaler = GradScaler("cuda", enabled=config.USE_AMP and device.type == "cuda")

    best_dice = 0.0
    best_epoch = 0
    best_metrics = None
    patience_counter = 0
    fold_vis_dir = config.SEG_VIS_DIR / f"clean_weighted_fold_{fold_idx}"
    fold_vis_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print(f"LR: encoder={optimizer.param_groups[0]['lr']:.2e}, decoder={optimizer.param_groups[1]['lr']:.2e}")
        train_loss = train_one_epoch_weighted(model, train_loader, optimizer, scaler, device, sample_weights)
        val_metrics = evaluate(model, val_loader, device, config.EVAL_THRESHOLD, fold_vis_dir, epoch)
        scheduler.step()

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Dice: {val_metrics['dice_mean']:.4f} +/- {val_metrics['dice_std']:.4f}")
        print(f"Val IoU:  {val_metrics['iou_mean']:.4f} +/- {val_metrics['iou_std']:.4f}")

        is_best = bool(val_metrics["dice_mean"] > best_dice)
        if tracker:
            tracker.update_epoch(
                fold_idx,
                epoch + 1,
                float(train_loss),
                float(val_metrics["dice_mean"]),
                float(val_metrics["iou_mean"]),
                is_best,
            )

        if is_best:
            best_dice = float(val_metrics["dice_mean"])
            best_epoch = epoch
            best_metrics = val_metrics
            patience_counter = 0
            ckpt_path = config.SEG_CHECKPOINT_DIR / f"clean_weighted_fold{fold_idx}_best.pth"
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "metrics": json_safe(val_metrics),
                "fold": fold_idx,
                "train_patients": train_patients,
                "val_patients": val_patients,
                "excluded_patients": EXCLUDED_PATIENTS,
                "duplicate_groups": DUPLICATE_GROUPS,
                "patient_stats": json_safe({p: {k: v for k, v in patient_stats[p].items() if k != "samples"} for p in train_patients + val_patients}),
            }, ckpt_path)
            print(f"  -> Saved best model (Dice={best_dice:.4f})")
        else:
            patience_counter += 1

        if epoch + 1 >= config.MIN_EPOCHS and patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"  -> Early stopping at epoch {epoch + 1} (best Dice={best_dice:.4f}, best epoch={best_epoch + 1})")
            break

    if tracker:
        tracker.finish_fold(fold_idx, float(best_dice), json_safe(best_metrics if best_metrics is not None else val_metrics))

    return {
        "fold": fold_idx,
        "train_patients": train_patients,
        "val_patients": val_patients,
        "best_dice": best_dice,
        "best_epoch": best_epoch + 1,
        "metrics": json_safe(best_metrics if best_metrics is not None else val_metrics),
    }


def write_audit(logs_dir, experiment_id, folds, patient_stats, sample_weights):
    audit = {
        "experiment_id": experiment_id,
        "created_at": datetime.now().isoformat(),
        "excluded_patients": EXCLUDED_PATIENTS,
        "duplicate_groups": DUPLICATE_GROUPS,
        "folds": json_safe(folds),
        "patient_stats": json_safe({p: {k: v for k, v in stats.items() if k != "samples"} for p, stats in patient_stats.items()}),
        "sample_weight_summary": {
            "min": float(min(sample_weights.values())),
            "mean": float(np.mean(list(sample_weights.values()))),
            "max": float(max(sample_weights.values())),
        },
        "weighting": {
            "patient_factor": "sqrt(median_patient_mask_count / patient_mask_count), clipped to [0.75, 1.50]",
            "foreground_factor": "sqrt(sample_foreground_ratio / median_foreground_ratio), clipped to [0.70, 1.30]",
            "final_weight": "patient_factor * foreground_factor clipped to [0.5, 2.0], then normalized to mean 1",
        },
    }
    path = Path(logs_dir) / f"audit_{experiment_id}.json"
    path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=None, help="Run one fold only")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs for smoke tests")
    args = parser.parse_args()

    folds = build_clean_folds()
    if args.fold is not None:
        selected = [fold for fold in folds if fold["fold"] == args.fold]
        if not selected:
            raise ValueError(f"Unknown fold: {args.fold}")
        folds = selected

    patients = clean_patients()
    patient_stats = collect_patient_stats(config.DATA_DIR, patients)
    sample_weights = compute_sample_weights(patient_stats, patients)

    experiment_id = f"clean_weighted_4fold_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    tracker = ProgressTracker(experiment_id=experiment_id, logs_dir=config.SEG_LOG_DIR)
    tracker.plan_folds([
        {
            "fold": fold["fold"],
            "total_epochs": args.epochs or config.EPOCHS,
            "train_patients": fold["train_patients"],
            "val_patients": fold["val_patients"],
        }
        for fold in folds
    ])
    audit_path = write_audit(config.SEG_LOG_DIR, experiment_id, folds, patient_stats, sample_weights)

    print("=" * 80)
    print("Clean Weighted 4-Fold Segmentation Training")
    print("=" * 80)
    print(f"Experiment ID: {experiment_id}")
    print(f"Audit: {audit_path}")
    print(f"Progress: {config.SEG_LOG_DIR / f'progress_{experiment_id}.json'}")
    print(f"Excluded patients: {EXCLUDED_PATIENTS}")
    print(f"Duplicate groups: {DUPLICATE_GROUPS}")
    for fold in folds:
        val_count = sum(patient_stats[p]["num_masks"] for p in fold["val_patients"])
        print(f"Fold {fold['fold']} val={fold['val_patients']} masks={val_count}")

    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.SEED)
    print(f"Device: {device}")

    results = []
    try:
        for fold in folds:
            results.append(train_fold(fold, device, patient_stats, sample_weights, tracker, max_epochs=args.epochs))

        mean_dice = float(np.mean([r["best_dice"] for r in results]))
        std_dice = float(np.std([r["best_dice"] for r in results]))
        tracker.finish_experiment(mean_dice, results)

        output_file = write_final_result(
            logs_dir=config.SEG_LOG_DIR,
            result_prefix="results_clean_weighted",
            split_mode="clean_weighted_4fold",
            experiment_id=experiment_id,
            mean_dice=mean_dice,
            std_dice=std_dice,
            fold_results=results,
            extra={
                "excluded_patients": EXCLUDED_PATIENTS,
                "duplicate_groups": DUPLICATE_GROUPS,
                "audit_file": str(audit_path),
            },
        )

        print("\n" + "=" * 80)
        print(f"Mean Dice: {mean_dice:.4f} +/- {std_dice:.4f}")
        print(f"Results saved to: {output_file}")
    except Exception as exc:
        tracker.mark_error(str(exc))
        print(f"\nTraining failed: {exc}")
        raise


if __name__ == "__main__":
    main()
