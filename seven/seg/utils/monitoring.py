"""Shared helpers for the file-based training monitor contract."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .progress_tracker import ProgressTracker


def json_safe(value: Any) -> Any:
    """Convert common NumPy containers/scalars into JSON-serializable values."""
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


def read_progress_history(logs_dir: Path, experiment_id: str) -> list[dict[str, Any]]:
    """Return fold-level epoch history from the matching progress file, if present."""
    progress_file = Path(logs_dir) / f"progress_{experiment_id}.json"
    if not progress_file.exists():
        return []
    data = json.loads(progress_file.read_text(encoding="utf-8"))
    return data.get("folds", [])


def write_final_result(
    *,
    logs_dir: Path,
    result_prefix: str,
    split_mode: str,
    experiment_id: str,
    mean_dice: float,
    fold_results: list[dict[str, Any]],
    std_dice: float | None = None,
    extra: dict[str, Any] | None = None,
    timestamp: str | None = None,
) -> Path:
    """Write a completed result JSON that `/root/monitor` can discover.

    The monitor scans `results_*.json` and expects completed runs to include
    `fold_results` plus optional `epoch_history` copied from the progress file.
    """
    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")

    final: dict[str, Any] = {
        "split_mode": split_mode,
        "timestamp": timestamp,
        "experiment_id": experiment_id,
        "mean_dice": float(mean_dice),
        "fold_results": json_safe(fold_results),
        "epoch_history": json_safe(read_progress_history(logs_dir, experiment_id)),
        "epoch_history_source": "progress_tracker",
    }
    if std_dice is not None:
        final["std_dice"] = float(std_dice)
    if extra:
        final.update(json_safe(extra))

    output_file = logs_dir / f"{result_prefix}_{timestamp}.json"
    output_file.write_text(json.dumps(final, indent=2), encoding="utf-8")
    return output_file


class MonitorRun:
    """Facade for the progress/result files consumed by the monitor UI."""

    def __init__(self, experiment_id: str, logs_dir: Path):
        self.experiment_id = experiment_id
        self.logs_dir = Path(logs_dir)
        self.tracker = ProgressTracker(experiment_id=experiment_id, logs_dir=self.logs_dir)

    @property
    def progress_file(self) -> Path:
        return self.logs_dir / f"progress_{self.experiment_id}.json"

    def plan_folds(self, folds: list[dict[str, Any]]) -> None:
        self.tracker.plan_folds(folds)

    def start_fold(
        self,
        fold_idx: int,
        total_epochs: int,
        train_patients: list[str],
        val_patients: list[str],
    ) -> None:
        self.tracker.start_fold(fold_idx, total_epochs, train_patients, val_patients)

    def update_epoch(
        self,
        fold_idx: int,
        epoch: int,
        train_loss: float,
        val_dice: float,
        val_iou: float,
        is_best: bool = False,
    ) -> None:
        self.tracker.update_epoch(fold_idx, epoch, train_loss, val_dice, val_iou, is_best)

    def finish_fold(self, fold_idx: int, best_dice: float, metrics: dict[str, Any]) -> None:
        self.tracker.finish_fold(fold_idx, float(best_dice), json_safe(metrics))

    def mark_error(self, error: Exception | str) -> None:
        self.tracker.mark_error(str(error))

    def finish(
        self,
        *,
        result_prefix: str,
        split_mode: str,
        mean_dice: float,
        fold_results: list[dict[str, Any]],
        std_dice: float | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        self.tracker.finish_experiment(float(mean_dice), json_safe(fold_results))
        return write_final_result(
            logs_dir=self.logs_dir,
            result_prefix=result_prefix,
            split_mode=split_mode,
            experiment_id=self.experiment_id,
            mean_dice=mean_dice,
            std_dice=std_dice,
            fold_results=fold_results,
            extra=extra,
        )
