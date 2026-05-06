import importlib.util
import tempfile
import unittest
from pathlib import Path


def load_parser_module():
    parser_path = Path(__file__).resolve().parents[1] / "parse_epoch_history.py"
    spec = importlib.util.spec_from_file_location("parse_epoch_history_under_test", parser_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ParseEpochHistoryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)
        self.parser = load_parser_module()

    def tearDown(self):
        self.tmp.cleanup()

    def write_log(self, text):
        path = self.tmp_path / "train.log"
        path.write_text(text, encoding="utf-8")
        return path

    def test_lopo_parser_preserves_iou_and_zero_based_fold_index(self):
        log = self.write_log(
            """
======================================================================
Fold 1: Train=['P002', 'P003'], Val=['P001']
Epoch 1/180
Train Loss: 0.8544
Val Dice: 0.0214 +/- 0.0054
Val IoU:  0.0108 +/- 0.0028
Epoch 2/180
Train Loss: 0.8060
Val Dice: 0.0980 +/- 0.0206
Val IoU:  0.0516 +/- 0.0115
"""
        )

        folds = self.parser.parse_training_log(log)

        self.assertEqual(len(folds), 1)
        self.assertEqual(folds[0]["fold"], 0)
        self.assertEqual(folds[0]["val_patients"], ["P001"])
        self.assertEqual(folds[0]["epochs"][0]["val_iou"], 0.0108)
        self.assertEqual(folds[0]["epochs"][1]["val_iou"], 0.0516)
        self.assertEqual(folds[0]["best_epoch"], 2)
        self.assertEqual(folds[0]["best_dice"], 0.0980)

    def test_stratified_parser_uses_explicit_fold_number_when_present(self):
        log = self.write_log(
            """
======================================================================
Training Fold 1
Val patients: ['P008', 'P013']
Train patients (2): ['P014', 'P015']
Fold 2: Train=['P014', 'P015'], Val=['P008', 'P013']
Epoch 1/180
Train Loss: 1.0121
Val Dice: 0.2180 +/- 0.1092
Val IoU:  0.1267 +/- 0.0708
"""
        )

        folds = self.parser.parse_training_log(log)

        self.assertEqual(len(folds), 1)
        self.assertEqual(folds[0]["fold"], 1)
        self.assertEqual(folds[0]["val_patients"], ["P008", "P013"])
        self.assertEqual(folds[0]["epochs"][0]["val_iou"], 0.1267)


if __name__ == "__main__":
    unittest.main()
