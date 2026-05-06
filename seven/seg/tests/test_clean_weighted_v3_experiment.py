import importlib.util
import unittest
from pathlib import Path

import torch


def load_module():
    module_path = Path(__file__).resolve().parents[1] / "train_clean_weighted_4fold_v3.py"
    spec = importlib.util.spec_from_file_location("clean_weighted_v3_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CleanWeightedV3ExperimentTest(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()

    def test_v3_keeps_clean_fold_contract(self):
        patients = [f"P{i:03d}" for i in range(1, 20)]

        folds = self.mod.build_clean_folds(patients)

        all_val = [p for fold in folds for p in fold["val_patients"]]
        self.assertEqual(sorted(all_val), sorted([p for p in patients if p != "P011"]))
        self.assertNotIn("P011", all_val)

    def test_weighted_tversky_penalizes_false_positives_more_than_false_negatives(self):
        # Same target area and same number of wrong pixels, but alpha=0.7 should
        # make the FP-heavy prediction worse than the FN-heavy prediction.
        target = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])
        pred_fp_heavy = torch.tensor([[[[8.0, 8.0], [8.0, -8.0]]]])
        pred_fn_heavy = torch.tensor([[[[8.0, -8.0], [-8.0, -8.0]]]])
        weights = torch.ones(1)

        fp_loss = self.mod.weighted_tversky_loss(pred_fp_heavy, target, weights)
        fn_loss = self.mod.weighted_tversky_loss(pred_fn_heavy, target, weights)

        self.assertGreater(float(fp_loss), float(fn_loss))

    def test_v3_experiment_naming_is_distinct(self):
        self.assertEqual(self.mod.EXPERIMENT_PREFIX, "clean_weighted_4fold_v3")
        self.assertEqual(self.mod.RESULT_PREFIX, "results_clean_weighted_v3")


if __name__ == "__main__":
    unittest.main()
