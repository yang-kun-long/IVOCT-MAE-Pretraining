"""Aggregate single-fold v13 LOPO results into a unified LOPO-18 result file.

Each `--fold N` invocation of train_clean_weighted_lopo_v13_two_stage.py
produces its own results_clean_weighted_lopo_v13_two_stage_<ts>.json with
fold_results=[1 fold]. After all folds finish this script fuses them into a
single results_clean_weighted_lopo_v13_two_stage_aggregated_<ts>.json with
fold_results=[N folds] and recomputed mean/std.

Usage:
    /root/miniconda3/bin/python aggregate_v13_results.py [--logs DIR] [--since YYYYMMDD_HHMMSS]
"""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np


RESULTS_PATTERN = re.compile(
    r"^results_clean_weighted_lopo_v13_two_stage_(\d{8}_\d{6})\.json$"
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--logs", default="/root/CN_seg/seven/seg/logs",
                   help="Directory containing results_*.json files.")
    p.add_argument("--since", default=None,
                   help="Only aggregate result files with timestamp >= this "
                        "(format YYYYMMDD_HHMMSS). Useful to exclude smoke runs.")
    p.add_argument("--output-prefix",
                   default="results_clean_weighted_lopo_v13_two_stage_aggregated",
                   help="Prefix for the aggregated output file.")
    return p.parse_args()


def load_single_fold_results(logs_dir, since_ts=None):
    logs_dir = Path(logs_dir)
    found = []
    for path in sorted(logs_dir.glob("results_clean_weighted_lopo_v13_two_stage_*.json")):
        m = RESULTS_PATTERN.match(path.name)
        if not m:
            continue
        ts = m.group(1)
        if since_ts is not None and ts < since_ts:
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"  WARN: failed to read {path.name}: {e}")
            continue
        folds = data.get("fold_results", [])
        if not folds:
            continue
        # one entry per fold file (single-fold runs); but accept multi-fold too
        for fr in folds:
            found.append({"source_file": path.name, "timestamp": ts, **fr})
    return found


def deduplicate_by_fold(entries):
    """If the same fold was trained multiple times, keep the most recent."""
    by_fold = {}
    for entry in entries:
        fold = entry["fold"]
        if fold not in by_fold or entry["timestamp"] > by_fold[fold]["timestamp"]:
            by_fold[fold] = entry
    return [by_fold[k] for k in sorted(by_fold)]


def main():
    args = parse_args()
    print(f"Scanning {args.logs} for v13 single-fold results...")
    entries = load_single_fold_results(args.logs, args.since)
    print(f"  found {len(entries)} fold entries across all result files")
    if not entries:
        print("Nothing to aggregate. Exit.")
        return

    folds = deduplicate_by_fold(entries)
    print(f"  unique folds after dedup: {len(folds)}")

    best_dice = [f["best_dice"] for f in folds]
    sweep_dice = [f["threshold_sweep"].get("best_dice", float("nan")) for f in folds]

    mean_dice = float(np.mean(best_dice))
    std_dice = float(np.std(best_dice))
    sweep_mean = float(np.nanmean(sweep_dice))
    sweep_std = float(np.nanstd(sweep_dice))

    print(f"\nLOPO-{len(folds)} aggregated:")
    print(f"  best-checkpoint Dice : mean={mean_dice:.4f} +/- {std_dice:.4f}")
    print(f"  threshold-sweep Dice : mean={sweep_mean:.4f} +/- {sweep_std:.4f}")
    print(f"\nPer-fold (sorted by fold idx):")
    for f in folds:
        cls = f["threshold_sweep"].get("best_cls_threshold", None)
        seg = f["threshold_sweep"].get("best_seg_threshold", None)
        det = f["threshold_sweep"].get("best_detection", {})
        print(
            f"  fold {f['fold']:>2} val={f['val_patients']} "
            f"best={f['best_dice']:.4f} sweep={f['threshold_sweep'].get('best_dice', 0):.4f} "
            f"@ cls={cls} seg={seg} tp/fp/fn/tn={det.get('tp')}/{det.get('fp')}/{det.get('fn')}/{det.get('tn')}"
        )

    out_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {
        "aggregated_at": datetime.now().isoformat(),
        "num_folds": len(folds),
        "experiment_prefix": "clean_weighted_lopo_v13_two_stage",
        "mean_dice": mean_dice,
        "std_dice": std_dice,
        "sweep_mean_dice": sweep_mean,
        "sweep_std_dice": sweep_std,
        "fold_results": [
            {k: v for k, v in f.items() if k not in {"source_file", "timestamp"}}
            for f in folds
        ],
        "source_files": [f["source_file"] for f in folds],
    }
    out_path = Path(args.logs) / f"{args.output_prefix}_{out_ts}.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nAggregated result written to: {out_path}")


if __name__ == "__main__":
    main()
