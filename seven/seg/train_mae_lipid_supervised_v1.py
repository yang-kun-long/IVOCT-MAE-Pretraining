"""Supervised pretraining on lipid masks.

Trains MAESkipSegmenter on the 4831 lipid masks in LIPID_DATA_SEG/, then saves
the encoder weights (only) for downstream calcification fine-tuning. Purpose:
inject IVOCT-specific tissue-segmentation prior into the encoder before the
small (268-mask) calc fine-tune. The decoder/head is throwaway here.

Design choices:
- Reuses IVOCTSegDataset (same dir structure works) — no new dataset class.
- Reuses MAESkipSegmenter (v7's model) and v5's loss (dice + focal BCE).
- Single train pass (no LOPO), 10-patient held-out val for convergence sanity.
- Saves encoder-only ckpt every 10 epochs + final. The "best" encoder is the
  one whose downstream calc Dice is highest, decided in a later experiment.
- Starts from mae_v2_best.pth so this is *continued* training, not from scratch.
"""

import argparse
import json
import random
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
from models import MAESkipSegmenter
from utils import compute_metrics, aggregate_metrics


# ---------------------------------------------------------------------------
# Lipid pretrain configuration
# ---------------------------------------------------------------------------

LIPID_DATA_DIR = Path("/root/CN_seg/LIPID_DATA_SEG")
PRETRAIN_CHECKPOINT_DIR = Path("/root/CN_seg/seven/seg/checkpoints_lipid_supervised_v1")
PRETRAIN_LOG_DIR = Path("/root/CN_seg/seven/seg/logs")
PRETRAIN_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
PRETRAIN_LOG_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS = 60
BATCH_SIZE = 16              # bigger than v7's 4 — lipid pretrain doesn't need v7 baseline parity
NUM_WORKERS = 8
DECODER_LR = 1e-3
ENCODER_LR = 5e-5
FREEZE_EPOCHS = 15           # shorter than v7's 40 — lipid task converges faster
UNFREEZE_TOP_LAYERS = 4
WEIGHT_DECAY = 0.05
WARMUP_EPOCHS = 3
SAVE_EVERY = 10
VAL_PATIENT_COUNT = 10
SEED = 42

EXPERIMENT_PREFIX = "lipid_supervised_pretrain_v1"


# ---------------------------------------------------------------------------
# Loss (matches v5/v7's weighted dice + focal BCE, but with uniform weights)
# ---------------------------------------------------------------------------

def dice_focal_bce_loss(pred_logit, target, alpha=0.75, gamma=2.0,
                        lambda_dice=1.0, lambda_bce=1.0):
    pred = torch.sigmoid(pred_logit)
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    dice_loss = 1.0 - ((2.0 * intersection + 1e-6) / (union + 1e-6))

    bce = F.binary_cross_entropy_with_logits(pred_logit, target, reduction="none")
    p_t = pred * target + (1.0 - pred) * (1.0 - target)
    alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)
    focal_bce = alpha_t * (1.0 - p_t).pow(gamma) * bce
    focal_bce = focal_bce.view(focal_bce.size(0), -1).mean(dim=1)

    per_sample = lambda_dice * dice_loss + lambda_bce * focal_bce
    return per_sample.mean()


# ---------------------------------------------------------------------------
# Training stage / optimizer (same shape as v5)
# ---------------------------------------------------------------------------

def stage_for_epoch(epoch):
    return "freeze_all" if epoch <= FREEZE_EPOCHS else "unfreeze_top_layers"


def apply_training_stage(model, stage, num_layers=UNFREEZE_TOP_LAYERS):
    for param in model.encoder.parameters():
        param.requires_grad = False
    for param in model.decoder.parameters():
        param.requires_grad = True

    if stage == "freeze_all":
        return
    if stage != "unfreeze_top_layers":
        raise ValueError(f"Unknown stage: {stage}")

    total_blocks = len(model.encoder.blocks)
    unfreeze_from = max(0, total_blocks - num_layers)
    for idx, block in enumerate(model.encoder.blocks):
        if idx >= unfreeze_from:
            for param in block.parameters():
                param.requires_grad = True
    for param in model.encoder.norm.parameters():
        param.requires_grad = True


def make_optimizer_for_stage(model):
    encoder_params, decoder_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("encoder."):
            encoder_params.append(param)
        else:
            decoder_params.append(param)
    groups = []
    if encoder_params:
        groups.append({"params": encoder_params, "lr": ENCODER_LR, "name": "encoder"})
    if decoder_params:
        groups.append({"params": decoder_params, "lr": DECODER_LR, "name": "decoder"})
    return AdamW(groups, weight_decay=WEIGHT_DECAY)


def make_scheduler(optimizer, epochs):
    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / WARMUP_EPOCHS
        return 0.5 * (1 + np.cos(np.pi * (epoch - WARMUP_EPOCHS) / (epochs - WARMUP_EPOCHS)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, scaler, device, use_amp):
    model.train()
    total_loss, num_batches = 0.0, 0
    for batch_idx, batch in enumerate(loader):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        optimizer.zero_grad()
        with autocast(device_type="cuda", enabled=use_amp):
            logits = model(images)
            loss = dice_focal_bce_loss(logits, masks)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += float(loss.item())
        num_batches += 1
        if (batch_idx + 1) % 20 == 0:
            print(f"  batch {batch_idx + 1}/{len(loader)} loss={loss.item():.4f}")
    return total_loss / max(num_batches, 1)


def evaluate(model, loader, device, use_amp, threshold=0.3):
    model.eval()
    metrics_list = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            with autocast(device_type="cuda", enabled=use_amp):
                logits = model(images)
            metrics_list.append(compute_metrics(logits, masks, threshold=threshold))
    if not metrics_list:
        return {"dice_mean": 0.0, "dice_std": 0.0, "iou_mean": 0.0, "iou_std": 0.0,
                "sensitivity_mean": 0.0, "sensitivity_std": 0.0,
                "specificity_mean": 0.0, "specificity_std": 0.0}
    return aggregate_metrics(metrics_list)


# ---------------------------------------------------------------------------
# Encoder-only checkpoint saving
# ---------------------------------------------------------------------------

def save_encoder_only(model, path, epoch, train_loss, val_metrics):
    encoder_state = {
        k: v for k, v in model.state_dict().items() if k.startswith("encoder.")
    }
    torch.save({
        "epoch": epoch,
        "encoder": encoder_state,
        "train_loss": float(train_loss),
        "val_metrics": {k: float(v) for k, v in val_metrics.items()},
        "source": "lipid_supervised_pretrain_v1",
        "note": "Use this to initialize MAESkipSegmenter.encoder for calc fine-tuning.",
    }, path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def discover_lipid_patients():
    return sorted([p.name for p in LIPID_DATA_DIR.iterdir() if p.is_dir() and p.name.startswith("P")])


def split_patients(patients, val_count=VAL_PATIENT_COUNT, seed=SEED):
    rng = random.Random(seed)
    shuffled = patients[:]
    rng.shuffle(shuffled)
    val = sorted(shuffled[:val_count])
    train = sorted(shuffled[val_count:])
    return train, val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_id = f"{EXPERIMENT_PREFIX}_{timestamp}"
    print(f"Experiment ID: {experiment_id}")

    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = config.USE_AMP and device.type == "cuda"
    print(f"Device: {device}, AMP: {use_amp}")

    patients = discover_lipid_patients()
    train_patients, val_patients = split_patients(patients)
    print(f"Lipid patients: {len(patients)} (train={len(train_patients)}, val={len(val_patients)})")
    print(f"Val patients: {val_patients}")

    train_dataset = IVOCTSegDataset(
        LIPID_DATA_DIR, train_patients, config.IMG_SIZE, config.ROI_CROP_RATIO, True
    )
    val_dataset = IVOCTSegDataset(
        LIPID_DATA_DIR, val_patients, config.IMG_SIZE, config.ROI_CROP_RATIO, False
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=NUM_WORKERS, pin_memory=True,
                              persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=NUM_WORKERS, pin_memory=True,
                            persistent_workers=True)
    print(f"Train samples={len(train_dataset)}, Val samples={len(val_dataset)}")

    model = MAESkipSegmenter(
        config.MAE_CHECKPOINT,
        patch_size=config.PATCH_SIZE,
        freeze_encoder=False,
        use_adapter=config.USE_ADAPTER,
        adapter_bottleneck=config.ADAPTER_BOTTLENECK,
        decoder_norm="batch",
    ).to(device)

    scaler = GradScaler("cuda", enabled=use_amp)
    optimizer = None
    scheduler = None
    current_stage = None

    history = []
    best_val_dice = 0.0
    best_path = PRETRAIN_CHECKPOINT_DIR / "encoder_best.pth"
    final_path = PRETRAIN_CHECKPOINT_DIR / "encoder_final.pth"

    for epoch in range(args.epochs):
        stage = stage_for_epoch(epoch + 1)
        if stage != current_stage:
            current_stage = stage
            apply_training_stage(model, stage)
            optimizer = make_optimizer_for_stage(model)
            scheduler = make_scheduler(optimizer, args.epochs)
            print(f"\nSwitched training stage to {stage}")

        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        lr_desc = ", ".join(f"{g.get('name', i)}={g['lr']:.2e}" for i, g in enumerate(optimizer.param_groups))
        print(f"LR: {lr_desc}")

        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, use_amp)
        val_metrics = evaluate(model, val_loader, device, use_amp, threshold=0.3)
        scheduler.step()

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Dice : {val_metrics['dice_mean']:.4f} +/- {val_metrics['dice_std']:.4f}")

        history.append({
            "epoch": epoch + 1,
            "stage": current_stage,
            "train_loss": train_loss,
            "val_dice_mean": float(val_metrics["dice_mean"]),
            "val_dice_std": float(val_metrics["dice_std"]),
            "val_iou_mean": float(val_metrics["iou_mean"]),
        })

        if val_metrics["dice_mean"] > best_val_dice:
            best_val_dice = float(val_metrics["dice_mean"])
            save_encoder_only(model, best_path, epoch + 1, train_loss, val_metrics)
            print(f"  -> Saved best encoder (val Dice={best_val_dice:.4f})")

        if (epoch + 1) % SAVE_EVERY == 0 or (epoch + 1) == args.epochs:
            ckpt_path = PRETRAIN_CHECKPOINT_DIR / f"encoder_epoch_{epoch + 1}.pth"
            save_encoder_only(model, ckpt_path, epoch + 1, train_loss, val_metrics)
            print(f"  -> Saved encoder at epoch {epoch + 1}")

    save_encoder_only(model, final_path, args.epochs, train_loss, val_metrics)

    history_path = PRETRAIN_LOG_DIR / f"{experiment_id}.json"
    history_path.write_text(json.dumps({
        "experiment_id": experiment_id,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "train_patients": train_patients,
        "val_patients": val_patients,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "best_val_dice": best_val_dice,
        "history": history,
    }, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"Done. Best val Dice = {best_val_dice:.4f}")
    print(f"Best encoder : {best_path}")
    print(f"Final encoder: {final_path}")
    print(f"History JSON : {history_path}")


if __name__ == "__main__":
    main()
