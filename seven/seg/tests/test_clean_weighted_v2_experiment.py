import importlib.util
import unittest
from pathlib import Path

import torch


def load_module():
    module_path = Path(__file__).resolve().parents[1] / "train_clean_weighted_4fold_v2.py"
    spec = importlib.util.spec_from_file_location("clean_weighted_v2_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CleanWeightedV2ExperimentTest(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()

    def test_v2_folds_exclude_p011_and_balance_mask_counts(self):
        patients = [f"P{i:03d}" for i in range(1, 20)]
        stats = {
            "P001": {"num_masks": 10},
            "P002": {"num_masks": 18},
            "P003": {"num_masks": 8},
            "P004": {"num_masks": 11},
            "P005": {"num_masks": 20},
            "P006": {"num_masks": 11},
            "P007": {"num_masks": 10},
            "P008": {"num_masks": 7},
            "P009": {"num_masks": 9},
            "P010": {"num_masks": 19},
            "P012": {"num_masks": 26},
            "P013": {"num_masks": 7},
            "P014": {"num_masks": 25},
            "P015": {"num_masks": 3},
            "P016": {"num_masks": 9},
            "P017": {"num_masks": 33},
            "P018": {"num_masks": 9},
            "P019": {"num_masks": 14},
        }

        folds = self.mod.build_clean_folds(patients)

        all_val = [p for fold in folds for p in fold["val_patients"]]
        self.assertEqual(sorted(all_val), sorted([p for p in patients if p != "P011"]))
        self.assertNotIn("P011", all_val)
        mask_counts = [sum(stats[p]["num_masks"] for p in fold["val_patients"]) for fold in folds]
        self.assertLessEqual(max(mask_counts) - min(mask_counts), 10)

    def test_stage_for_epoch_switches_after_freeze_epochs(self):
        self.assertEqual(self.mod.stage_for_epoch(1), "freeze_all")
        self.assertEqual(self.mod.stage_for_epoch(self.mod.FREEZE_EPOCHS), "freeze_all")
        self.assertEqual(self.mod.stage_for_epoch(self.mod.FREEZE_EPOCHS + 1), "unfreeze_top_layers")

    def test_threshold_sweep_selects_best_threshold(self):
        probs = torch.tensor([[[[0.2, 0.4], [0.6, 0.8]]]], dtype=torch.float32)
        masks = torch.tensor([[[[0.0, 0.0], [1.0, 1.0]]]], dtype=torch.float32)

        sweep = self.mod.compute_threshold_sweep_from_probs(
            [(probs, masks)],
            thresholds=[0.3, 0.5, 0.7],
        )

        self.assertEqual(sweep["best_threshold"], 0.5)
        self.assertAlmostEqual(sweep["best_dice"], 1.0)
        self.assertEqual(len(sweep["thresholds"]), 3)


if __name__ == "__main__":
    unittest.main()
