"""Export hard-patient segmentation review pack for a completed experiment."""
import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(1, str(Path(__file__).parent.parent))

import config_seg as config
from datasets import IVOCTSegDataset
from models import MAESegmenter


DEFAULT_HARD_PATIENTS = ["P014", "P008", "P015", "P007", "P016", "P013"]


def dice_iou(pred, target):
    pred = pred.astype(bool)
    target = target.astype(bool)
    tp = np.logical_and(pred, target).sum()
    fp = np.logical_and(pred, ~target).sum()
    fn = np.logical_and(~pred, target).sum()
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    return float(dice), float(iou), int(tp), int(fp), int(fn)


def latest_result(logs_dir):
    files = sorted(Path(logs_dir).glob("results_clean_weighted_v2_*.json"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError("No results_clean_weighted_v2_*.json files found")
    return files[-1]


def load_model(fold, device):
    ckpt_path = config.SEG_CHECKPOINT_DIR / f"clean_weighted_v2_fold{fold}_best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = MAESegmenter(
        config.MAE_CHECKPOINT,
        patch_size=config.PATCH_SIZE,
        freeze_encoder=False,
        use_adapter=config.USE_ADAPTER,
        adapter_bottleneck=config.ADAPTER_BOTTLENECK,
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def patient_to_fold(results):
    mapping = {}
    for fold in results.get("fold_results", []):
        for patient in fold["val_patients"]:
            mapping[patient] = int(fold["fold"])
    return mapping


def save_case_figure(out_path, image, target, prob, pred, title):
    image = image.squeeze()
    target = target.squeeze().astype(bool)
    prob = prob.squeeze()
    pred = pred.squeeze().astype(bool)

    overlay = np.zeros((*image.shape, 3), dtype=np.float32)
    overlay[..., 0] = np.logical_and(target, ~pred)  # FN red
    overlay[..., 1] = np.logical_and(target, pred)   # TP green
    overlay[..., 2] = np.logical_and(~target, pred)  # FP blue

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Input")
    axes[1].imshow(image, cmap="gray")
    axes[1].imshow(target, cmap="Reds", alpha=0.45)
    axes[1].set_title("GT")
    axes[2].imshow(prob, cmap="jet", vmin=0, vmax=1)
    axes[2].set_title("Pred prob")
    axes[3].imshow(image, cmap="gray")
    axes[3].imshow(overlay, alpha=0.65)
    axes[3].set_title("TP green / FP blue / FN red")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def summarize_rows(rows):
    by_patient = {}
    for row in rows:
        by_patient.setdefault(row["patient"], []).append(row)
    summary = []
    for patient, patient_rows in sorted(by_patient.items()):
        summary.append({
            "patient": patient,
            "n": len(patient_rows),
            "dice_mean": float(np.mean([r["dice"] for r in patient_rows])),
            "iou_mean": float(np.mean([r["iou"] for r in patient_rows])),
            "fg_mean": float(np.mean([r["gt_fg_ratio"] for r in patient_rows])),
            "pred_fg_mean": float(np.mean([r["pred_fg_ratio"] for r in patient_rows])),
        })
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-json", default=None)
    parser.add_argument("--patients", nargs="*", default=DEFAULT_HARD_PATIENTS)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--tag", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    args = parser.parse_args()

    results_path = Path(args.results_json) if args.results_json else latest_result(config.SEG_LOG_DIR)
    results = json.loads(results_path.read_text(encoding="utf-8"))
    fold_by_patient = patient_to_fold(results)
    out_dir = Path("result") / f"hard_patient_review_{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    amp_enabled = config.USE_AMP and device.type == "cuda"
    rows = []

    for patient in args.patients:
        if patient not in fold_by_patient:
            print(f"[warn] patient not in validation folds, skipping: {patient}")
            continue
        fold = fold_by_patient[patient]
        model, checkpoint = load_model(fold, device)
        dataset = IVOCTSegDataset(config.DATA_DIR, [patient], config.IMG_SIZE, config.ROI_CROP_RATIO, False)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        patient_dir = out_dir / patient
        patient_dir.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            for idx, batch in enumerate(loader):
                image = batch["image"].to(device)
                mask = batch["mask"].to(device)
                with autocast(device_type="cuda", enabled=amp_enabled):
                    logits = model(image)
                prob = torch.sigmoid(logits).cpu().numpy()[0, 0]
                target = mask.cpu().numpy()[0, 0]
                img = image.cpu().numpy()[0, 0]
                pred = prob > args.threshold
                dice, iou, tp, fp, fn = dice_iou(pred, target > 0.5)
                path = Path(batch["path"][0])
                stem = path.stem
                row = {
                    "patient": patient,
                    "fold": fold,
                    "sample": stem,
                    "path": str(path),
                    "threshold": args.threshold,
                    "dice": dice,
                    "iou": iou,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "gt_fg_ratio": float((target > 0.5).mean()),
                    "pred_fg_ratio": float(pred.mean()),
                    "prob_mean": float(prob.mean()),
                    "prob_max": float(prob.max()),
                    "checkpoint_epoch": int(checkpoint.get("epoch", -1)) + 1,
                }
                rows.append(row)
                title = f"{patient} fold {fold} {stem} dice={dice:.3f} iou={iou:.3f}"
                save_case_figure(patient_dir / f"{stem}.png", img, target, prob, pred, title)

    summary = summarize_rows(rows)
    (out_dir / "summary.json").write_text(
        json.dumps({
            "created_at": datetime.now().isoformat(),
            "results_json": str(results_path),
            "experiment_id": results.get("experiment_id"),
            "threshold": args.threshold,
            "patients": args.patients,
            "patient_summary": summary,
            "num_cases": len(rows),
        }, indent=2),
        encoding="utf-8",
    )

    with open(out_dir / "per_sample_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["patient"])
        writer.writeheader()
        writer.writerows(rows)

    with open(out_dir / "patient_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()) if summary else ["patient"])
        writer.writeheader()
        writer.writerows(summary)

    print(f"Review directory: {out_dir}")
    print(f"Cases: {len(rows)}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
