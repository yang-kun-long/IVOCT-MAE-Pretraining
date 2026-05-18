import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


def load_monitor_app():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    spec = importlib.util.spec_from_file_location("monitor_app_batch_under_test", app_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BatchGroupingTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.logs_dir = Path(self.tmp.name)
        self.module = load_monitor_app()
        self.module.LOGS_DIR = self.logs_dir
        self.client = self.module.app.test_client()

    def tearDown(self):
        self.tmp.cleanup()

    def write_result(self, name, payload):
        path = self.logs_dir / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def write_progress(self, name, payload):
        path = self.logs_dir / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def _result_payload(self, *, experiment_id, batch_id, mean_dice, val_patient):
        return {
            "split_mode": "lipid_warm_sentinel",
            "experiment_id": experiment_id,
            "batch_id": batch_id,
            "mean_dice": mean_dice,
            "fold_results": [{
                "fold": 0,
                "best_dice": mean_dice,
                "best_epoch": 50,
                "train_patients": ["P003"],
                "val_patients": [val_patient],
            }],
            "epoch_history": [],
        }

    def test_batches_endpoint_groups_completed_siblings(self):
        ts_base = "20260518_15"
        members = [
            ("a_001", "P014", 0.50),
            ("b_002", "P008", 0.55),
            ("c_003", "P013", 0.60),
            ("d_004", "P006", 0.85),
        ]
        for ts_suffix, patient, dice in [(f"{i:02d}05", p, d) for i, (_, p, d) in enumerate(members)]:
            exp_id = f"lipid_warm_sentinel_{ts_base}{ts_suffix}"
            self.write_result(
                f"results_lipid_warm_sentinel_{ts_base}{ts_suffix}.json",
                self._result_payload(
                    experiment_id=exp_id, batch_id="core4_20260518_150000",
                    mean_dice=dice, val_patient=patient,
                ),
            )

        response = self.client.get("/api/batches")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(len(payload["batches"]), 1)
        self.assertEqual(payload["ungrouped"], [])

        batch = payload["batches"][0]
        self.assertEqual(batch["batch_id"], "core4_20260518_150000")
        self.assertEqual(batch["num_members"], 4)
        self.assertEqual(batch["status"], "completed")
        self.assertEqual(batch["completed_count"], 4)
        self.assertEqual(batch["running_count"], 0)
        self.assertAlmostEqual(batch["mean_dice_completed"], (0.50 + 0.55 + 0.60 + 0.85) / 4, places=4)
        self.assertEqual(len(batch["members"]), 4)

    def test_batches_endpoint_mixes_running_and_completed(self):
        self.write_result(
            "results_lipid_warm_sentinel_20260518_150000.json",
            self._result_payload(
                experiment_id="lipid_warm_sentinel_20260518_150000",
                batch_id="core4_20260518_150000",
                mean_dice=0.55, val_patient="P014",
            ),
        )
        self.write_progress(
            "progress_lipid_warm_sentinel_20260518_150100.json",
            {
                "experiment_id": "lipid_warm_sentinel_20260518_150100",
                "batch_id": "core4_20260518_150000",
                "status": "running",
                "start_time": "2026-05-18T15:01:00",
                "last_update": "2026-05-18T15:30:00",
                "current_fold": 0,
                "folds": [{
                    "fold": 0,
                    "status": "running",
                    "total_epochs": 180,
                    "current_epoch": 90,
                    "train_patients": ["P003"],
                    "val_patients": ["P008"],
                    "epochs": [],
                    "best_dice": 0.30,
                    "best_epoch": 80,
                }],
            },
        )

        payload = self.client.get("/api/batches").get_json()
        self.assertEqual(len(payload["batches"]), 1)
        batch = payload["batches"][0]
        self.assertEqual(batch["num_members"], 2)
        self.assertEqual(batch["status"], "running")
        self.assertEqual(batch["completed_count"], 1)
        self.assertEqual(batch["running_count"], 1)
        # Mean Dice should reflect only completed members
        self.assertAlmostEqual(batch["mean_dice_completed"], 0.55, places=4)

    def test_experiments_without_batch_id_remain_ungrouped(self):
        self.write_result(
            "results_legacy_20260518_140000.json",
            {
                "split_mode": "legacy",
                "experiment_id": "legacy_20260518_140000",
                "mean_dice": 0.60,
                "fold_results": [{
                    "fold": 0, "best_dice": 0.60, "best_epoch": 50,
                    "train_patients": ["P002"], "val_patients": ["P001"],
                }],
                "epoch_history": [],
            },
        )
        payload = self.client.get("/api/batches").get_json()
        self.assertEqual(payload["batches"], [])
        self.assertEqual(len(payload["ungrouped"]), 1)
        self.assertIsNone(payload["ungrouped"][0]["batch_id"])

    def test_experiments_endpoint_still_returns_batch_id_field(self):
        """Backward compat: /api/experiments stays flat but exposes batch_id."""
        self.write_result(
            "results_lipid_warm_sentinel_20260518_150000.json",
            self._result_payload(
                experiment_id="lipid_warm_sentinel_20260518_150000",
                batch_id="core4_20260518_150000",
                mean_dice=0.50, val_patient="P014",
            ),
        )
        payload = self.client.get("/api/experiments").get_json()
        self.assertEqual(len(payload), 1)
        self.assertEqual(payload[0]["batch_id"], "core4_20260518_150000")


if __name__ == "__main__":
    unittest.main()
