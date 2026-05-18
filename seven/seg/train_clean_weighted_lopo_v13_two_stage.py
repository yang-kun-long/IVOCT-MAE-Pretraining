"""V13 LOPO training: two-stage detect-then-segment with shared MAE skip encoder.

Builds on the v5 base training pipeline (loss/sampler/schedule), but:
  1. Uses ``MAESkipSegmenterV13`` which adds a frame-level CA classifier head
     on the encoder CLS token.
  2. Uses ``IVOCTSegDatasetV13`` which surfaces both positive (mask) frames and
     subsampled negative (no-mask) frames so the cls head has real supervision.
  3. Replaces the seg-only loss with a joint loss:
        L = (dice + focal_bce) over positive frames
          + lambda_neg_bce * BCE over negative frames (seg signal only)
          + lambda_cls    * BCE over all frames (cls signal)
  4. Performs a 2-D (cls_threshold, seg_threshold) sweep at evaluation and
     reports detection-gated metrics in addition to legacy seg Dice on positive
     frames.

This file follows the same monkey-patch-the-v5-base pattern as
``train_clean_weighted_lopo_v7_skip.py``: only the things that change are
overridden, everything else (sampler, audit, monitor, checkpoint layout) is
reused.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(1, str(Path(__file__).parent.parent))

import config_seg as config
import train_clean_weighted_4fold_v5 as base
from datasets import IVOCTSegDatasetV13
from models import MAESkipSegmenterV13
from utils import compute_metrics, aggregate_metrics, save_seg_visualization


# ---------------------------------------------------------------------------
# Experiment identity and tuning knobs
# ---------------------------------------------------------------------------

base.EXPERIMENT_PREFIX = "clean_weighted_lopo_v13_two_stage"
base.RESULT_PREFIX = "results_clean_weighted_lopo_v13_two_stage"
base.CHECKPOINT_PREFIX = "clean_weighted_lopo_v13_two_stage"
base.VIS_PREFIX = "clean_weighted_lopo_v13_two_stage"

LAMBDA_CLS = 0.5            # weight of the frame-level classification BCE
LAMBDA_NEG_BCE = 0.3        # weight of seg BCE on negative frames
NEG_POS_RATIO = 2.0
CLS_THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70]


# ---------------------------------------------------------------------------
# LOPO fold builder (same as v7)
# ---------------------------------------------------------------------------

def build_lopo_folds(all_patients=None):
    patients = base.clean_patients(all_patients)
    folds = []
    for idx, patient in enumerate(patients):
        folds.append({
            "fold": idx,
            "train_patients": [p for p in patients if p != patient],
            "val_patients": [patient],
        })
    base.validate_folds(folds, patients=patients, duplicate_groups=base.DUPLICATE_GROUPS)
    return folds


# ---------------------------------------------------------------------------
# Joint loss (seg + cls) with negative-frame handling
# ---------------------------------------------------------------------------

def joint_loss(model_out, masks, has_calc, weights):
    seg_logit = model_out["seg"]
    cls_logit = model_out["cls"]

    cls_loss = F.binary_cross_entropy_with_logits(cls_logit, has_calc, reduction="mean")

    pos_mask = has_calc > 0.5
    seg_loss_total = seg_logit.new_zeros(())
    contributions = 0

    if pos_mask.any():
        pos_idx = pos_mask.nonzero(as_tuple=True)[0]
        seg_loss_pos = base.weighted_dice_focal_bce_loss(
            seg_logit[pos_idx], masks[pos_idx], weights[pos_idx]
        )
        seg_loss_total = seg_loss_total + seg_loss_pos
        contributions += 1

    neg_mask = ~pos_mask
    if neg_mask.any():
        neg_idx = neg_mask.nonzero(as_tuple=True)[0]
        neg_bce = F.binary_cross_entropy_with_logits(
            seg_logit[neg_idx], masks[neg_idx], reduction="none"
        )
        neg_bce = neg_bce.view(neg_bce.size(0), -1).mean(dim=1)
        w = weights[neg_idx]
        neg_seg = (neg_bce * w).sum() / w.sum().clamp_min(1e-6)
        seg_loss_total = seg_loss_total + LAMBDA_NEG_BCE * neg_seg
        contributions += 1

    if contributions == 0:
        seg_loss_total = seg_logit.new_zeros(())

    return seg_loss_total + LAMBDA_CLS * cls_loss


# ---------------------------------------------------------------------------
# Train / eval loops aware of the dict model output
# ---------------------------------------------------------------------------

def train_one_epoch_weighted_v13(model, loader, optimizer, scaler, device, sample_weights):
    model.train()
    total_loss = 0.0
    num_batches = 0
    amp_enabled = config.USE_AMP and device.type == "cuda"

    for batch_idx, batch in enumerate(loader):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        has_calc = batch["has_calc"].to(device)
        paths = batch["path"]
        weights = torch.tensor(
            [sample_weights.get(path, 1.0) for path in paths],
            dtype=torch.float32, device=device,
        )

        optimizer.zero_grad()
        with autocast(device_type="cuda", enabled=amp_enabled):
            out = model(images)
            loss = joint_loss(out, masks, has_calc, weights)

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
            print(f"  Batch {batch_idx + 1}/{len(loader)}, Loss: {loss.item():.4f}")

    return total_loss / max(num_batches, 1)


def _gated_metrics_for_threshold(seg_probs, cls_probs, masks, has_calc,
                                   seg_threshold, cls_threshold):
    """Compute compute_metrics-compatible Dice ONLY on positive frames after
    gating; also return a few aggregate detection counters across all frames."""
    pos_idx = (has_calc > 0.5).nonzero(as_tuple=True)[0]
    metrics_pos = None
    if pos_idx.numel() > 0:
        gate_pos = (cls_probs[pos_idx] >= cls_threshold).float().view(-1, 1, 1, 1)
        seg_logit_eff = torch.logit(
            (seg_probs[pos_idx] * gate_pos).clamp(1e-6, 1 - 1e-6)
        )
        metrics_pos = compute_metrics(seg_logit_eff, masks[pos_idx], threshold=seg_threshold)

    pred_any = ((seg_probs >= seg_threshold) &
                (cls_probs.view(-1, 1, 1, 1) >= cls_threshold)).any(dim=(1, 2, 3))
    tp = int(((has_calc > 0.5) & pred_any).sum().item())
    fp = int(((has_calc < 0.5) & pred_any).sum().item())
    fn = int(((has_calc > 0.5) & ~pred_any).sum().item())
    tn = int(((has_calc < 0.5) & ~pred_any).sum().item())

    return metrics_pos, {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def collect_validation_probs_v13(model, loader, device):
    model.eval()
    seg_probs_all = []
    cls_probs_all = []
    masks_all = []
    has_calc_all = []
    amp_enabled = config.USE_AMP and device.type == "cuda"
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"]
            has_calc = batch["has_calc"]
            with autocast(device_type="cuda", enabled=amp_enabled):
                out = model(images)
            seg_probs_all.append(torch.sigmoid(out["seg"]).float().cpu())
            cls_probs_all.append(torch.sigmoid(out["cls"]).float().cpu())
            masks_all.append(masks.cpu())
            has_calc_all.append(has_calc.cpu())
    return {
        "seg_probs": torch.cat(seg_probs_all, dim=0),
        "cls_probs": torch.cat(cls_probs_all, dim=0),
        "masks": torch.cat(masks_all, dim=0),
        "has_calc": torch.cat(has_calc_all, dim=0),
    }


def evaluate_v13(model, loader, device, threshold, vis_dir=None, epoch=None):
    """Evaluate seg Dice on positive frames at fixed seg_threshold, cls gate disabled.

    Used inside the per-epoch loop to drive checkpoint selection (kept comparable
    to v7's "Val Dice"). Cls-gated sweep is run at end-of-fold separately.
    """
    model.eval()
    metrics_list = []
    amp_enabled = config.USE_AMP and device.type == "cuda"
    first_batch_for_vis = None

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            has_calc = batch["has_calc"].to(device)
            with autocast(device_type="cuda", enabled=amp_enabled):
                out = model(images)
            seg_logits = out["seg"]
            pos_idx = (has_calc > 0.5).nonzero(as_tuple=True)[0]
            if pos_idx.numel() > 0:
                metrics_list.append(
                    compute_metrics(seg_logits[pos_idx], masks[pos_idx], threshold=threshold)
                )
            if batch_idx == 0 and vis_dir is not None and epoch is not None:
                first_batch_for_vis = (images, seg_logits, masks)

    if not metrics_list:
        return {"dice_mean": 0.0, "dice_std": 0.0, "iou_mean": 0.0, "iou_std": 0.0,
                "sensitivity_mean": 0.0, "sensitivity_std": 0.0,
                "specificity_mean": 0.0, "specificity_std": 0.0}

    if first_batch_for_vis is not None and vis_dir is not None and epoch is not None:
        save_seg_visualization(*first_batch_for_vis, vis_dir / f"epoch_{epoch:03d}.png")

    return aggregate_metrics(metrics_list)


def threshold_sweep_v13(model, loader, device):
    """2-D sweep over (cls_threshold, seg_threshold).

    Reports best by *positive-frame Dice* (apples-to-apples with v7).
    """
    cache = collect_validation_probs_v13(model, loader, device)
    seg_probs = cache["seg_probs"]
    cls_probs = cache["cls_probs"]
    masks = cache["masks"]
    has_calc = cache["has_calc"]

    rows = []
    for cls_thr in CLS_THRESHOLDS:
        for seg_thr in base.THRESHOLDS:
            metrics_pos, det = _gated_metrics_for_threshold(
                seg_probs, cls_probs, masks, has_calc, seg_thr, cls_thr
            )
            if metrics_pos is None:
                continue
            agg = aggregate_metrics([metrics_pos])
            rows.append({
                "cls_threshold": float(cls_thr),
                "seg_threshold": float(seg_thr),
                "dice_mean": float(agg["dice_mean"]),
                "dice_std": float(agg["dice_std"]),
                "iou_mean": float(agg["iou_mean"]),
                "iou_std": float(agg["iou_std"]),
                "tp": det["tp"], "fp": det["fp"], "fn": det["fn"], "tn": det["tn"],
            })

    if not rows:
        return {"thresholds": [], "best_cls_threshold": 0.5, "best_seg_threshold": 0.3,
                "best_threshold": 0.3, "best_dice": 0.0, "best_iou": 0.0}

    best = max(rows, key=lambda r: r["dice_mean"])
    return {
        "thresholds": rows,
        "best_cls_threshold": best["cls_threshold"],
        "best_seg_threshold": best["seg_threshold"],
        # alias for v5 train_fold print compatibility
        "best_threshold": best["seg_threshold"],
        "best_dice": best["dice_mean"],
        "best_iou": best["iou_mean"],
        "best_detection": {"tp": best["tp"], "fp": best["fp"], "fn": best["fn"], "tn": best["tn"]},
    }


def patient_metrics_v13(model, loader, device, threshold=None):
    """Per-patient metrics with cls gate disabled (legacy comparable view)."""
    cache = collect_validation_probs_v13(model, loader, device)
    seg_probs = cache["seg_probs"]
    masks = cache["masks"]
    has_calc = cache["has_calc"]
    threshold = config.EVAL_THRESHOLD if threshold is None else threshold

    # Recover patient ids by replaying the loader once (cheap)
    patients = []
    for batch in loader:
        patients.extend(batch["patient"])

    pred = (seg_probs >= threshold)
    rows = []
    for idx in range(seg_probs.size(0)):
        if has_calc[idx].item() < 0.5:
            continue  # only positives
        p = pred[idx, 0].numpy().astype(bool)
        t = masks[idx, 0].numpy().astype(bool)
        tp = np.logical_and(p, t).sum()
        fp = np.logical_and(p, ~t).sum()
        fn = np.logical_and(~p, t).sum()
        tn = np.logical_and(~p, ~t).sum()
        rows.append({
            "patient": patients[idx],
            "dice": float((2 * tp) / (2 * tp + fp + fn + 1e-8)),
            "iou": float(tp / (tp + fp + fn + 1e-8)),
            "sensitivity": float(tp / (tp + fn + 1e-8)),
            "specificity": float(tn / (tn + fp + 1e-8)),
            "gt_fg_ratio": float(t.mean()),
            "pred_fg_ratio": float(p.mean()),
        })
    return base.aggregate_patient_metrics(rows)


# ---------------------------------------------------------------------------
# Dataset class swap: v5 train_fold reads ``IVOCTSegDataset`` from its globals,
# so swapping the binding here is enough.
# ---------------------------------------------------------------------------

class _DatasetFactory:
    """Adapter so v5's `IVOCTSegDataset(...)` call returns a V13 dataset
    with negatives for training and full coverage for validation."""

    def __call__(self, data_dir, patient_ids, img_size, crop_ratio, is_train):
        return IVOCTSegDatasetV13(
            data_dir=data_dir,
            patient_ids=patient_ids,
            img_size=img_size,
            crop_ratio=crop_ratio,
            is_train=is_train,
            include_negatives=True,
            neg_pos_ratio=NEG_POS_RATIO,
        )


# ---------------------------------------------------------------------------
# Apply patches at module level
# ---------------------------------------------------------------------------

base.MAESegmenter = MAESkipSegmenterV13
base.IVOCTSegDataset = _DatasetFactory()
base.build_clean_folds = build_lopo_folds
base.train_one_epoch_weighted = train_one_epoch_weighted_v13
base.evaluate = evaluate_v13
base.threshold_sweep = threshold_sweep_v13
base.patient_metrics = patient_metrics_v13


if __name__ == "__main__":
    base.main()
