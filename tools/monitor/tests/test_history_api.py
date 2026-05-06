import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


def load_monitor_app():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    spec = importlib.util.spec_from_file_location("monitor_app_under_test", app_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class HistoryApiTest(unittest.TestCase):
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

    def test_realtime_endpoint_serves_parsed_epoch_history_for_completed_results(self):
        self.write_result(
            "results_lopo_20260423_222429.json",
            {
                "split_mode": "lopo",
                "mean_dice": 0.51,
                "epoch_history_source": "parsed_from_log",
                "fold_results": [
                    {
                        "fold": 0,
                        "best_dice": 0.55,
                        "best_epoch": 2,
                        "train_patients": ["P002"],
                        "val_patients": ["P001"],
                    }
                ],
                "epoch_history": [
                    {
                        "fold": 0,
                        "current_epoch": 2,
                        "total_epochs": 180,
                        "best_dice": 0.55,
                        "best_epoch": 2,
                        "train_patients": ["P002"],
                        "val_patients": ["P001"],
                        "epochs": [
                            {
                                "epoch": 1,
                                "train_loss": 0.9,
                                "val_dice": 0.1,
                                "val_iou": 0.05,
                                "is_best": True,
                            },
                            {
                                "epoch": 2,
                                "train_loss": 0.8,
                                "val_dice": 0.55,
                                "val_iou": 0.38,
                                "is_best": True,
                            },
                        ],
                    }
                ],
            },
        )

        response = self.client.get("/api/experiment/results_lopo_20260423_222429/realtime")

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "completed")
        self.assertEqual(data["history_source"], "parsed_from_log")
        self.assertEqual(len(data["chart_data"]), 2)
        self.assertEqual(data["chart_data"][1]["val_dice"], 0.55)

    def test_experiment_list_exposes_stable_history_metadata(self):
        self.write_result(
            "results_stratified_20260504_211238.json",
            {
                "split_mode": "stratified_4fold",
                "mean_dice": 0.42,
                "metadata": {
                    "name": "Stratified partial",
                    "strategy": {"split_mode": "stratified_4fold", "num_folds": 4},
                },
                "fold_results": [
                    {"fold": 0, "best_dice": 0.3, "val_patients": ["P014", "P015"]},
                    {"fold": 1, "best_dice": 0.4, "val_patients": ["P008", "P013"]},
                    {"fold": 2, "best_dice": 0.5, "val_patients": ["P012", "P017"]},
                ],
                "epoch_history": [
                    {"fold": 0, "best_dice": 0.3, "val_patients": ["P999"], "epochs": []},
                    {"fold": 1, "best_dice": 0.4, "val_patients": ["P998"], "epochs": []},
                    {"fold": 2, "best_dice": 0.5, "val_patients": ["P997"], "epochs": []},
                ],
            },
        )

        response = self.client.get("/api/experiments")

        self.assertEqual(response.status_code, 200)
        experiments = response.get_json()
        self.assertEqual(len(experiments), 1)
        exp = experiments[0]
        self.assertEqual(exp["timestamp"], "2026-05-04T21:12:38")
        self.assertEqual(exp["timestamp_source"], "filename")
        self.assertIn("updated_at", exp)
        self.assertTrue(exp["is_partial"])
        self.assertEqual(exp["expected_folds"], 4)
        self.assertEqual(exp["num_folds"], 3)
        self.assertEqual(exp["history_quality"], "mismatch")

    def test_experiment_list_marks_matching_history_as_partial_when_iou_is_missing(self):
        self.write_result(
            "results_lopo_20260423_235957.json",
            {
                "split_mode": "lopo",
                "mean_dice": 0.5,
                "fold_results": [
                    {"fold": 0, "best_dice": 0.3806438803594524, "val_patients": ["P001"]},
                ],
                "epoch_history": [
                    {
                        "fold": 0,
                        "best_dice": 0.3806,
                        "val_patients": ["P001"],
                        "epochs": [{"epoch": 1, "train_loss": 0.8, "val_dice": 0.1, "val_iou": 0.0}],
                    }
                ],
            },
        )

        response = self.client.get("/api/experiments")

        self.assertEqual(response.status_code, 200)
        exp = response.get_json()[0]
        self.assertEqual(exp["history_quality"], "partial")
        self.assertIn("IoU", exp["history_quality_notes"][0])

    def test_realtime_endpoint_suppresses_mismatched_history_chart_data(self):
        self.write_result(
            "results_stratified_20260504_211828.json",
            {
                "split_mode": "stratified_4fold",
                "mean_dice": 0.42,
                "fold_results": [
                    {"fold": 0, "best_dice": 0.3227, "val_patients": ["P014"]},
                ],
                "epoch_history": [
                    {
                        "fold": 0,
                        "best_dice": 0.3695,
                        "val_patients": ["P008"],
                        "epochs": [
                            {"epoch": 1, "train_loss": 1.0, "val_dice": 0.2, "val_iou": 0.1}
                        ],
                    }
                ],
            },
        )

        response = self.client.get("/api/experiment/results_stratified_20260504_211828/realtime")

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["history_quality"], "mismatch")
        self.assertEqual(data["chart_data"], [])
        self.assertTrue(data["history_quality_notes"])

    def test_completed_experiment_detail_includes_history_quality(self):
        self.write_result(
            "results_lopo_20260423_235957.json",
            {
                "split_mode": "lopo",
                "mean_dice": 0.5,
                "fold_results": [
                    {"fold": 0, "best_dice": 0.3806438803594524, "val_patients": ["P001"]},
                ],
                "epoch_history": [
                    {
                        "fold": 0,
                        "best_dice": 0.3806,
                        "val_patients": ["P001"],
                        "epochs": [{"epoch": 1, "train_loss": 0.8, "val_dice": 0.1, "val_iou": 0.0}],
                    }
                ],
            },
        )

        response = self.client.get("/api/experiment/results_lopo_20260423_235957")

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["history_quality"], "partial")
        self.assertTrue(data["history_quality_notes"])

    def test_legacy_history_endpoints_delegate_to_normalized_history(self):
        self.write_result(
            "results_lopo_20260423_222429.json",
            {
                "split_mode": "lopo",
                "mean_dice": 0.51,
                "fold_results": [{"fold": 0, "best_dice": 0.55}],
            },
        )

        status_response = self.client.get("/api/training_status")
        history_response = self.client.get("/api/training_history")

        self.assertEqual(status_response.status_code, 200)
        self.assertEqual(history_response.status_code, 200)
        status = status_response.get_json()
        history = history_response.get_json()
        self.assertEqual(status["status"], "idle")
        self.assertEqual(status["total_experiments"], 1)
        self.assertEqual(history[0]["id"], "results_lopo_20260423_222429")

    def test_running_progress_experiment_includes_progress_and_eta(self):
        progress = {
            "experiment_id": "exp_live",
            "status": "running",
            "start_time": "2026-05-05T12:00:00",
            "last_update": "2026-05-05T12:30:00",
            "current_fold": 1,
            "folds": [
                {
                    "fold": 0,
                    "status": "completed",
                    "total_epochs": 10,
                    "current_epoch": 10,
                    "epochs": [{"epoch": i, "timestamp": f"2026-05-05T12:{i:02d}:00"} for i in range(1, 11)],
                },
                {
                    "fold": 1,
                    "status": "running",
                    "total_epochs": 10,
                    "current_epoch": 5,
                    "epochs": [{"epoch": i, "timestamp": f"2026-05-05T12:{10+i:02d}:00"} for i in range(1, 6)],
                },
            ],
        }
        (self.logs_dir / "progress_exp_live.json").write_text(json.dumps(progress), encoding="utf-8")

        with patch.object(self.module, "now_iso", return_value="2026-05-05T12:30:00"):
            list_response = self.client.get("/api/experiments")
            realtime_response = self.client.get("/api/experiment/exp_live/realtime")

        self.assertEqual(list_response.status_code, 200)
        self.assertEqual(realtime_response.status_code, 200)
        exp = list_response.get_json()[0]
        realtime = realtime_response.get_json()
        self.assertEqual(exp["progress"]["completed_epochs"], 15)
        self.assertEqual(exp["progress"]["total_epochs"], 20)
        self.assertEqual(exp["progress"]["percent"], 75.0)
        self.assertEqual(exp["progress"]["eta_seconds"], 600)
        self.assertEqual(realtime["progress"]["percent"], 75.0)
        self.assertEqual(realtime["progress"]["eta_seconds"], 600)

    def test_completed_result_hides_matching_progress_file_from_list(self):
        self.write_result(
            "results_clean_weighted_20260505_132432.json",
            {
                "split_mode": "clean_weighted_4fold",
                "experiment_id": "clean_weighted_4fold_20260505_130442",
                "mean_dice": 0.425,
                "fold_results": [{"fold": 0, "best_dice": 0.425}],
                "epoch_history": [{"fold": 0, "epochs": []}],
            },
        )
        progress = {
            "experiment_id": "clean_weighted_4fold_20260505_130442",
            "status": "completed",
            "start_time": "2026-05-05T13:04:42",
            "last_update": "2026-05-05T13:24:32",
            "current_fold": 3,
            "folds": [{"fold": 0, "status": "completed", "total_epochs": 1, "current_epoch": 1, "epochs": []}],
        }
        (self.logs_dir / "progress_clean_weighted_4fold_20260505_130442.json").write_text(
            json.dumps(progress),
            encoding="utf-8",
        )

        response = self.client.get("/api/experiments")

        self.assertEqual(response.status_code, 200)
        experiments = response.get_json()
        self.assertEqual(len(experiments), 1)
        self.assertEqual(experiments[0]["id"], "results_clean_weighted_20260505_132432")
        self.assertEqual(experiments[0]["experiment_id"], "clean_weighted_4fold_20260505_130442")


if __name__ == "__main__":
    unittest.main()
