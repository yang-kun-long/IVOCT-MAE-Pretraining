"""Aggregate parallel single-fold worker results into one synthetic batch result.

When `run_lopo_parallel.sh` fans out N workers each handling one LOPO fold, each
worker writes its own `results_<split_mode>_<ts>.json` containing one
fold_results entry. The training monitor's existing UI was built around v12-style
single-process runs that produced one result JSON with N fold_results inside it,
so these N parallel files look like N separate experiments instead of one.

This script reads all worker outputs sharing a `batch_id`, fuses their
fold_results and epoch_history into one synthetic v12-style result JSON, marks
the originals `monitor_visible: false`, and lets `tools/monitor/app.py`'s
existing composite_config logic merge across rounds (Core-4 → Expanded-8 →
Full LOPO18) without further changes.

Usage:
    /root/miniconda3/bin/python aggregate_batch.py                # aggregate all completed batches
    /root/miniconda3/bin/python aggregate_batch.py --batch-id ID  # aggregate a specific batch
    /root/miniconda3/bin/python aggregate_batch.py --dry-run      # show what would happen
"""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np


SYNTH_SUFFIX_RE = re.compile(r"_aggregated_\d{8}_\d{6}\.json$")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--logs-dir", default="/root/CN_seg/seven/seg/logs")
    p.add_argument("--batch-id", default=None,
                   help="Aggregate just this batch_id (default: aggregate all).")
    p.add_argument("--force", action="store_true",
                   help="Aggregate even if some members are still 'running'.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would happen, write nothing.")
    return p.parse_args()


def is_synthetic(path: Path) -> bool:
    return bool(SYNTH_SUFFIX_RE.search(path.name))


def discover_batches(logs_dir: Path):
    """Return {batch_id: [(path, data), ...]} from result files that declare a batch_id."""
    batches = {}
    for path in sorted(logs_dir.glob("results_*.json")):
        if is_synthetic(path):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  skip {path.name}: parse failed ({e})")
            continue
        batch_id = data.get("batch_id")
        if not batch_id:
            continue
        batches.setdefault(batch_id, []).append((path, data))
    return batches


def all_members_completed(members):
    for path, data in members:
        status = data.get("status")
        if status not in (None, "completed"):
            return False
        if not data.get("fold_results"):
            return False
    return True


def merge_members(members, batch_id):
    """Build the synthetic result payload from sibling worker outputs."""
    members_sorted = sorted(
        members,
        key=lambda pd: (pd[1].get("fold_results", [{}])[0].get("fold", 0)),
    )
    fold_results = []
    epoch_history = []
    split_modes = set()
    source_files = []
    timestamps = []
    for path, data in members_sorted:
        split_modes.add(data.get("split_mode"))
        source_files.append(path.name)
        timestamps.append(data.get("timestamp") or path.stem.split("_")[-2:])
        for fr in data.get("fold_results", []):
            fold_results.append(fr)
        for fh in data.get("epoch_history", []):
            epoch_history.append(fh)

    if len(split_modes) != 1:
        raise ValueError(
            f"batch {batch_id} mixes split_modes {split_modes}; aggregator needs "
            "all members from a single panel"
        )
    split_mode = next(iter(split_modes))

    dices = [
        float(fr["best_dice"]) for fr in fold_results
        if fr.get("best_dice") is not None
    ]
    mean_dice = float(np.mean(dices)) if dices else 0.0
    std_dice = float(np.std(dices)) if dices else 0.0

    sweep_dices = [
        float(fr.get("threshold_sweep", {}).get("best_dice", float("nan")))
        for fr in fold_results
    ]
    sweep_dices = [d for d in sweep_dices if not (d != d)]  # filter NaN
    sweep_mean = float(np.mean(sweep_dices)) if sweep_dices else None

    out_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    synthetic = {
        "split_mode": split_mode,
        "timestamp": out_ts,
        "experiment_id": f"{split_mode}_aggregated_{out_ts}",
        "mean_dice": mean_dice,
        "std_dice": std_dice,
        "fold_results": fold_results,
        "epoch_history": epoch_history,
        "epoch_history_source": "progress_tracker",
        "num_folds": len(fold_results),
        "aggregated_from": source_files,
        "metadata": {
            "synthetic_batch_aggregation": True,
            "source_batch_id": batch_id,
            "members": len(source_files),
            "aggregator_version": "1.0",
            "sweep_mean_dice": sweep_mean,
            "description_cn": (
                f"由 {len(source_files)} 个并发 worker 的单 fold 结果合并而成；"
                f"原始文件保留但已 monitor_visible=false。"
            ),
        },
    }
    return synthetic, split_mode, source_files


def mark_original_hidden(path: Path, data: dict, synthetic_name: str, dry_run: bool):
    metadata = data.setdefault("metadata", {})
    metadata["monitor_visible"] = False
    metadata["aggregated_into"] = synthetic_name
    if dry_run:
        return
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def mark_progress_hidden(logs_dir: Path, experiment_id: str, synthetic_name: str, dry_run: bool):
    """Also hide the matching progress_<id>.json so the worker doesn't reappear
    as a still-running experiment after its result file is hidden."""
    if not experiment_id:
        return None
    progress_path = logs_dir / f"progress_{experiment_id}.json"
    if not progress_path.exists():
        return None
    try:
        data = json.loads(progress_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    metadata = data.setdefault("metadata", {})
    metadata["monitor_visible"] = False
    metadata["aggregated_into"] = synthetic_name
    if dry_run:
        return progress_path
    progress_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return progress_path


def aggregate_batch(batch_id, members, logs_dir, dry_run, force):
    if not force and not all_members_completed(members):
        print(f"[skip] batch {batch_id}: not all members completed; pass --force to override")
        return None

    # Re-aggregating an already-merged batch should replace the previous synthetic
    # (don't leave stale files visible on the monitor).
    existing_synthetics = [
        p for p in logs_dir.glob("results_*_aggregated_*.json")
        if is_synthetic(p) and _peek_batch_id(p) == batch_id
    ]

    synthetic, split_mode, source_files = merge_members(members, batch_id)
    synth_path = logs_dir / f"results_{split_mode}_aggregated_{synthetic['timestamp']}.json"

    print(f"[aggregate] batch {batch_id}")
    print(f"  members        : {len(members)}")
    print(f"  fold_results   : {len(synthetic['fold_results'])} (mean Dice {synthetic['mean_dice']:.4f})")
    print(f"  epoch_history  : {len(synthetic['epoch_history'])} folds")
    print(f"  -> synthetic   : {synth_path.name}")
    if synthetic['metadata'].get("sweep_mean_dice") is not None:
        print(f"  sweep mean Dice: {synthetic['metadata']['sweep_mean_dice']:.4f}")
    if existing_synthetics:
        print(f"  superseding    : {[p.name for p in existing_synthetics]}")

    if dry_run:
        print("  (dry-run: no writes)")
        return synth_path

    for old in existing_synthetics:
        old.unlink()

    synth_path.write_text(json.dumps(synthetic, indent=2), encoding="utf-8")
    for path, data in members:
        mark_original_hidden(path, data, synth_path.name, dry_run=False)
        print(f"  hid original   : {path.name}")
        progress_hidden = mark_progress_hidden(
            logs_dir, data.get("experiment_id"), synth_path.name, dry_run=False
        )
        if progress_hidden:
            print(f"  hid progress   : {progress_hidden.name}")

    return synth_path


def _peek_batch_id(path):
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("batch_id")
    except Exception:
        return None


def main():
    args = parse_args()
    logs_dir = Path(args.logs_dir)
    if not logs_dir.exists():
        raise SystemExit(f"logs_dir not found: {logs_dir}")

    print(f"Scanning {logs_dir} for batched results...")
    batches = discover_batches(logs_dir)

    if args.batch_id:
        if args.batch_id not in batches:
            print(f"No members found for batch_id={args.batch_id}")
            return
        targets = {args.batch_id: batches[args.batch_id]}
    else:
        targets = batches

    if not targets:
        print("No batches to aggregate.")
        return

    print(f"Found {len(targets)} batch(es) with worker results.")
    for batch_id, members in targets.items():
        aggregate_batch(batch_id, members, logs_dir, args.dry_run, args.force)

    if args.dry_run:
        print("\nDry-run complete. Re-run without --dry-run to write files.")
    else:
        print("\nDone.")


if __name__ == "__main__":
    main()
