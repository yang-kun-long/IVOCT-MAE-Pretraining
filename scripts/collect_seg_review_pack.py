#!/usr/bin/env python
"""Collect a lightweight review pack for segmentation runs.

Run from the repository root:

    python scripts/collect_seg_review_pack.py

Optional:

    python scripts/collect_seg_review_pack.py --tag 20260423
    python scripts/collect_seg_review_pack.py --results-json seven/seg/logs/results_lopo_20260423_222429.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import tarfile
from datetime import datetime
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tag",
        default=datetime.now().strftime("%Y%m%d"),
        help="Tag used in result/review_pack_<tag>/ and the output tarball name.",
    )
    parser.add_argument(
        "--results-json",
        default=None,
        help="Optional explicit path to a LOPO results json, relative to repo root.",
    )
    return parser.parse_args()


def find_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def choose_results_json(root: Path, explicit: str | None) -> Path | None:
    if explicit:
        path = root / explicit
        if not path.exists():
            raise FileNotFoundError(f"Results json not found: {path}")
        return path

    candidates = sorted((root / "seven" / "seg" / "logs").glob("results_lopo_*.json"))
    if candidates:
        return candidates[-1]

    baseline = root / "seven" / "seg" / "logs" / "results_lopo_baseline.json"
    if baseline.exists():
        return baseline
    return None


def copy_if_exists(root: Path, out_dir: Path, relative_path: str, collected: list[str]) -> None:
    src = root / relative_path
    if not src.exists():
        return
    dst = out_dir / relative_path
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    collected.append(relative_path)


def main() -> None:
    args = parse_args()
    root = find_repo_root()
    out_dir = root / "result" / f"review_pack_{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict[str, object]] = {}
    collected_files: list[str] = []

    for fold in range(4):
        ckpt_rel = f"seven/seg/checkpoints/seg_fold{fold}_best.pth"
        ckpt_path = root / ckpt_rel
        if not ckpt_path.exists():
            print(f"[warn] Missing checkpoint: {ckpt_rel}")
            continue

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        epoch_idx = int(ckpt["epoch"])
        epoch_human = epoch_idx + 1
        metrics = ckpt["metrics"]

        summary[f"fold_{fold}"] = {
            "checkpoint": ckpt_rel,
            "epoch_index": epoch_idx,
            "epoch_human": epoch_human,
            "dice_mean": float(metrics["dice_mean"]),
            "dice_std": float(metrics["dice_std"]),
            "iou_mean": float(metrics["iou_mean"]),
            "iou_std": float(metrics["iou_std"]),
            "train_patients": ckpt["train_patients"],
            "val_patients": ckpt["val_patients"],
        }

        vis_rel = f"seven/seg/vis/fold_{fold}/epoch_{epoch_idx:03d}.png"
        vis_path = root / vis_rel
        if vis_path.exists():
            vis_dst = out_dir / f"fold_{fold}_best_epoch_{epoch_human:03d}.png"
            shutil.copy2(vis_path, vis_dst)
            collected_files.append(str(vis_dst.relative_to(root)))
        else:
            print(f"[warn] Missing visualization: {vis_rel}")

    summary_rel = f"result/review_pack_{args.tag}/checkpoint_summary.json"
    summary_path = out_dir / "checkpoint_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    collected_files.append(summary_rel)

    results_json = choose_results_json(root, args.results_json)
    if results_json is not None:
        copy_if_exists(root, out_dir, str(results_json.relative_to(root)), collected_files)

    extra_files = [
        "seven/seg/logs/results_lopo_baseline.json",
        "result/seg_lopo_baseline_20260423.md",
        "SEGMENTATION_NOTES.md",
        "seven/seg/config_seg.py",
        "seven/seg/train_seg.py",
        "seven/seg/models/seg_model.py",
        "seven/seg/datasets/ivoct_seg_dataset.py",
        "seven/seg/utils/seg_losses.py",
        "seven/seg/utils/seg_metrics.py",
        "seven/config_v2.py",
        "seven/models/mae_hybrid_v2.py",
    ]
    for rel in extra_files:
        copy_if_exists(root, out_dir, rel, collected_files)

    manifest = {
        "tag": args.tag,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "repo_root": str(root),
        "results_json": str(results_json.relative_to(root)) if results_json else None,
        "collected_files": sorted(collected_files),
    }
    manifest_rel = f"result/review_pack_{args.tag}/manifest.json"
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    tar_path = root / "result" / f"review_pack_{args.tag}.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(out_dir, arcname=out_dir.relative_to(root))

    print(f"Review pack directory: {out_dir.relative_to(root)}")
    print(f"Review pack archive: {tar_path.relative_to(root)}")
    print(f"Summary file: {summary_rel}")
    print(f"Manifest file: {manifest_rel}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
