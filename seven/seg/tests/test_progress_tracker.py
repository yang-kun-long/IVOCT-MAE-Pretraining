import tempfile
import unittest
from pathlib import Path

from seven.seg.utils.progress_tracker import ProgressTracker


class ProgressTrackerTest(unittest.TestCase):
    def test_plan_folds_and_start_fold_updates_existing_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            tracker = ProgressTracker("planned", Path(tmp))
            tracker.plan_folds([
                {
                    "fold": 0,
                    "total_epochs": 180,
                    "train_patients": ["P001"],
                    "val_patients": ["P002"],
                },
                {
                    "fold": 1,
                    "total_epochs": 180,
                    "train_patients": ["P002"],
                    "val_patients": ["P001"],
                },
            ])

            tracker.start_fold(0, 180, ["P001"], ["P002"])
            data = tracker._read_json()

            self.assertEqual(len(data["folds"]), 2)
            self.assertEqual(data["folds"][0]["status"], "running")
            self.assertEqual(data["folds"][1]["status"], "pending")
            self.assertEqual(sum(f["total_epochs"] for f in data["folds"]), 360)


if __name__ == "__main__":
    unittest.main()
