import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from seven.seg.utils.monitoring import MonitorRun, write_final_result


class MonitoringHelperTest(unittest.TestCase):
    def test_write_final_result_attaches_progress_history_and_numpy_safe_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            logs_dir = Path(tmp)
            progress = {
                "experiment_id": "exp1",
                "folds": [
                    {
                        "fold": 0,
                        "epochs": [{"epoch": 1, "train_loss": 0.5, "val_dice": 0.4}],
                    }
                ],
            }
            (logs_dir / "progress_exp1.json").write_text(json.dumps(progress), encoding="utf-8")

            output = write_final_result(
                logs_dir=logs_dir,
                result_prefix="results_test",
                split_mode="unit_test",
                experiment_id="exp1",
                mean_dice=np.float32(0.4),
                std_dice=np.float64(0.1),
                fold_results=[{"fold": np.int64(0), "best_dice": np.float32(0.4)}],
                extra={"excluded_patients": ("P011",)},
                timestamp="20260506_120000",
            )

            data = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(output.name, "results_test_20260506_120000.json")
            self.assertEqual(data["experiment_id"], "exp1")
            self.assertEqual(data["epoch_history"], progress["folds"])
            self.assertEqual(data["epoch_history_source"], "progress_tracker")
            self.assertEqual(data["fold_results"][0]["fold"], 0)
            self.assertEqual(data["excluded_patients"], ["P011"])

    def test_monitor_run_wraps_progress_and_final_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            monitor = MonitorRun("exp2", Path(tmp))
            monitor.plan_folds([
                {
                    "fold": 0,
                    "total_epochs": 2,
                    "train_patients": ["P001"],
                    "val_patients": ["P002"],
                }
            ])
            monitor.start_fold(0, 2, ["P001"], ["P002"])
            monitor.update_epoch(0, 1, train_loss=0.5, val_dice=0.4, val_iou=0.25, is_best=True)
            monitor.finish_fold(0, 0.4, {"dice_mean": np.float32(0.4)})

            output = monitor.finish(
                result_prefix="results_facade",
                split_mode="facade",
                mean_dice=0.4,
                fold_results=[{"fold": 0, "best_dice": 0.4}],
                std_dice=0.0,
            )

            progress = json.loads(monitor.progress_file.read_text(encoding="utf-8"))
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(progress["status"], "completed")
            self.assertEqual(result["epoch_history"][0]["epochs"][0]["val_iou"], 0.25)


if __name__ == "__main__":
    unittest.main()
