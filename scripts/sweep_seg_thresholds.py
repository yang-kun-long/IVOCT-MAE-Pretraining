#!/usr/bin/env python
"""Evaluate saved segmentation checkpoints across multiple thresholds.

Run from repository root:

    python scripts/sweep_seg_thresholds.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.3, 0.4, 0.5, 0.6, 0.7],
        help="Thresholds to evaluate.",
    )
    parser.add_argument(
        "--output",
        default="seven/seg/logs/threshold_sweep_summary.json",
        help="Output json path, relative to repo root.",
    )
    return parser.parse_args()


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    seg_dir = root / "seven" / "seg"

    import sys

    sys.path.insert(0, str(seg_dir))
    sys.path.insert(1, str(root / "seven"))

    import config_seg as config
    from datasets import IVOCTSegDataset
    from models import MAESegmenter
    from utils import compute_metrics, aggregate_metrics

    args = parse_args()
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

    summary = {
        "thresholds": args.thresholds,
        "folds": {},
        "global_best_threshold_by_mean_dice": None,
        "global_best_mean_dice": None,
    }

    global_scores = {threshold: [] for threshold in args.thresholds}

    for fold in range(len(config.PATIENTS)):
        ckpt_path = config.SEG_CHECKPOINT_DIR / f"seg_fold{fold}_best.pth"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        val_patients = ckpt["val_patients"]

        dataset = IVOCTSegDataset(
            config.DATA_DIR,
            val_patients,
            img_size=config.IMG_SIZE,
            crop_ratio=config.ROI_CROP_RATIO,
            is_train=False,
        )
        loader = DataLoader(
            dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
        )

        model = MAESegmenter(config.MAE_CHECKPOINT, freeze_encoder=config.FREEZE_ENCODER).to(device)
        model.load_state_dict(ckpt["model"], strict=True)
        model.eval()

        fold_result = {
            "checkpoint": str(ckpt_path.relative_to(root)),
            "val_patients": val_patients,
            "best_epoch": int(ckpt["epoch"]) + 1,
            "thresholds": {},
        }

        with torch.no_grad():
            cached_batches = []
            for batch in loader:
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)
                logits = model(images)
                cached_batches.append((logits.cpu(), masks.cpu()))

        best_threshold = None
        best_dice = -1.0

        for threshold in args.thresholds:
            metrics_list = []
            for logits_cpu, masks_cpu in cached_batches:
                metrics_list.append(compute_metrics(logits_cpu, masks_cpu, threshold=threshold))

            agg = aggregate_metrics(metrics_list)
            fold_result["thresholds"][str(threshold)] = agg
            global_scores[threshold].append(float(agg["dice_mean"]))

            if agg["dice_mean"] > best_dice:
                best_dice = float(agg["dice_mean"])
                best_threshold = threshold

        fold_result["best_threshold"] = best_threshold
        fold_result["best_dice_mean"] = best_dice
        summary["folds"][f"fold_{fold}"] = fold_result

    global_best_threshold = None
    global_best_mean = -1.0
    for threshold, scores in global_scores.items():
        mean_score = sum(scores) / len(scores)
        if mean_score > global_best_mean:
            global_best_mean = mean_score
            global_best_threshold = threshold

    summary["global_best_threshold_by_mean_dice"] = global_best_threshold
    summary["global_best_mean_dice"] = global_best_mean

    output_path = root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
