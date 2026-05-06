import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


def load_module():
    module_path = Path(__file__).resolve().parents[1] / "train_clean_weighted_4fold.py"
    spec = importlib.util.spec_from_file_location("clean_weighted_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CleanWeightedExperimentTest(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write_sample(self, patient, name, foreground_pixels):
        patient_dir = self.root / patient
        data_dir = patient_dir / "Data"
        mask_dir = patient_dir / "mask"
        data_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (10, 10), color=(0, 0, 0)).save(data_dir / f"{name}.jpg")
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask.flat[:foreground_pixels] = 255
        Image.fromarray(mask).save(mask_dir / f"{name}_mask.png")

    def test_clean_folds_exclude_p011_and_cover_each_patient_once(self):
        patients = [f"P{i:03d}" for i in range(1, 20)]

        folds = self.mod.build_clean_folds(patients)

        self.assertEqual(len(folds), 4)
        all_val = [p for fold in folds for p in fold["val_patients"]]
        self.assertNotIn("P011", all_val)
        self.assertNotIn("P011", [p for fold in folds for p in fold["train_patients"]])
        self.assertEqual(sorted(all_val), sorted([p for p in patients if p != "P011"]))

    def test_discover_patients_uses_data_directories(self):
        (self.root / "P002").mkdir()
        (self.root / "P001").mkdir()
        (self.root / "notes").mkdir()

        self.assertEqual(self.mod.discover_patients(self.root), ["P001", "P002"])

    def test_validate_folds_rejects_duplicate_group_split(self):
        folds = [{
            "fold": 0,
            "train_patients": ["P010", "P001"],
            "val_patients": ["P011"],
        }]

        with self.assertRaisesRegex(ValueError, "Duplicate group"):
            self.mod.validate_folds(
                folds,
                patients=["P001", "P010", "P011"],
                duplicate_groups=[["P010", "P011"]],
            )

    def test_sample_weights_are_capped_and_normalized(self):
        self.write_sample("P001", "a", 1)
        self.write_sample("P001", "b", 1)
        self.write_sample("P002", "a", 50)

        stats = self.mod.collect_patient_stats(self.root, ["P001", "P002"])
        weights = self.mod.compute_sample_weights(stats, ["P001", "P002"])

        self.assertEqual(len(weights), 3)
        self.assertAlmostEqual(sum(weights.values()) / len(weights), 1.0)
        self.assertGreaterEqual(min(weights.values()), 0.5)
        self.assertLessEqual(max(weights.values()), 2.0)


if __name__ == "__main__":
    unittest.main()
