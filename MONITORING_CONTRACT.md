# Monitoring Contract

This project uses a file-based contract between training jobs and the monitor UI.
Training code writes JSON files under `seven/seg/logs/`; the Flask monitor reads
those files and never imports training modules.

## Monitor Location

The monitor source is versioned in:

```text
tools/monitor/
```

The server deployment currently lives at:

```text
/root/monitor
```

The monitor reads:

```text
/root/CN_seg/seven/seg/logs/
```

Default monitor app port in code: `6006`. As of 2026-05-06, the monitor process
was not running; port `6007` was TensorBoard.

## File Naming

Running experiments write progress files:

```text
progress_<experiment_id>.json
```

Completed experiments write result files:

```text
results_<name>_<YYYYMMDD>_<HHMMSS>.json
```

The monitor discovers experiments by scanning `progress_*.json` and
`results_*.json`.

## Progress JSON

`ProgressTracker` writes progress files. Minimum shape:

```json
{
  "experiment_id": "clean_weighted_4fold_v2_20260505_185221",
  "status": "running",
  "start_time": "2026-05-05T18:52:21.000000",
  "last_update": "2026-05-05T19:10:00.000000",
  "current_fold": 1,
  "folds": [
    {
      "fold": 0,
      "status": "completed",
      "start_time": "2026-05-05T18:52:21.000000",
      "end_time": "2026-05-05T19:00:00.000000",
      "total_epochs": 180,
      "current_epoch": 180,
      "train_patients": ["P001"],
      "val_patients": ["P002"],
      "epochs": [
        {
          "epoch": 1,
          "timestamp": "2026-05-05T18:53:00.000000",
          "train_loss": 0.5,
          "val_dice": 0.4,
          "val_iou": 0.25,
          "is_best": true
        }
      ],
      "best_dice": 0.4,
      "best_epoch": 1,
      "final_metrics": {}
    }
  ]
}
```

Required top-level fields:

- `experiment_id`
- `status`
- `start_time`
- `last_update`
- `current_fold`
- `folds`

Required fold fields:

- `fold`
- `status`
- `total_epochs`
- `current_epoch`
- `train_patients`
- `val_patients`
- `epochs`
- `best_dice`
- `best_epoch`

Required epoch fields:

- `epoch`
- `timestamp`
- `train_loss`
- `val_dice`
- `val_iou`
- `is_best`

## Result JSON

Use `seven.seg.utils.monitoring.MonitorRun` for new training scripts. It wraps
`ProgressTracker` and `write_final_result(...)` so scripts do not need to know
where progress and completed result files are assembled.

Minimum shape:

```json
{
  "split_mode": "clean_weighted_4fold_v2",
  "timestamp": "20260505_192922",
  "experiment_id": "clean_weighted_4fold_v2_20260505_185221",
  "mean_dice": 0.4714,
  "std_dice": 0.0558,
  "fold_results": [
    {
      "fold": 0,
      "train_patients": ["P001"],
      "val_patients": ["P002"],
      "best_dice": 0.3994,
      "best_epoch": 91
    }
  ],
  "epoch_history": [],
  "epoch_history_source": "progress_tracker"
}
```

Required top-level fields:

- `split_mode`
- `timestamp`
- `experiment_id`
- `mean_dice`
- `fold_results`
- `epoch_history`

Recommended fields:

- `std_dice`
- `epoch_history_source`
- `excluded_patients`
- `duplicate_groups`
- `audit_file`
- `metadata`

## Training Script Pattern

```python
from utils.monitoring import MonitorRun

experiment_id = f"my_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
monitor = MonitorRun(experiment_id=experiment_id, logs_dir=config.SEG_LOG_DIR)
monitor.plan_folds([...])

try:
    results = []
    for fold in folds:
        monitor.start_fold(...)
        ...
        monitor.update_epoch(...)
        ...
        monitor.finish_fold(...)
        results.append(...)

    mean_dice = ...
    std_dice = ...

    output_file = monitor.finish(
        result_prefix="results_my_experiment",
        split_mode="my_experiment",
        mean_dice=mean_dice,
        std_dice=std_dice,
        fold_results=results,
        extra={"audit_file": str(audit_path)},
    )
except Exception as exc:
    monitor.mark_error(exc)
    raise
```

## Boundary

The monitor may depend on JSON field names, but it should not depend on Python
training modules. New training code should depend on `MonitorRun` rather than
calling Flask APIs or hand-writing monitor JSON.
