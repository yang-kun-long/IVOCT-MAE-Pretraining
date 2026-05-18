import importlib.util
import sys
import unittest
from pathlib import Path

import torch


def load_module():
    module_path = Path(__file__).resolve().parents[1] / "train_clean_weighted_lopo_v13_two_stage.py"
    sys.path.insert(0, str(module_path.parent))
    spec = importlib.util.spec_from_file_location("v13_two_stage_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class V13TwoStageTest(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()

    def test_lopo_folds_cover_clean_patients(self):
        all_p = [f"P{i:03d}" for i in range(1, 20)]
        folds = self.mod.build_lopo_folds(all_p)
        self.assertEqual(len(folds), 18)
        val_patients = [f["val_patients"][0] for f in folds]
        self.assertNotIn("P011", val_patients)
        for fold in folds:
            self.assertEqual(len(fold["val_patients"]), 1)
            self.assertEqual(len(fold["train_patients"]), 17)
            self.assertNotIn(fold["val_patients"][0], fold["train_patients"])

    def test_joint_loss_has_cls_signal(self):
        torch.manual_seed(0)
        seg_logits = torch.full((4, 1, 8, 8), -3.0, requires_grad=True)
        cls_logits = torch.tensor([2.0, -2.0, 2.0, -2.0], requires_grad=True)
        masks = torch.zeros(4, 1, 8, 8)
        masks[0, :, :3, :3] = 1.0
        masks[2, :, :3, :3] = 1.0
        has_calc = torch.tensor([1.0, 0.0, 1.0, 0.0])
        weights = torch.ones(4)

        loss = self.mod.joint_loss(
            {"seg": seg_logits, "cls": cls_logits},
            masks, has_calc, weights,
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(loss), 0.0)
        self.assertIsNotNone(cls_logits.grad)
        self.assertGreater(float(cls_logits.grad.abs().sum()), 0.0)

    def test_joint_loss_negative_only_batch(self):
        seg_logits = torch.full((2, 1, 8, 8), -3.0, requires_grad=True)
        cls_logits = torch.tensor([-2.0, -2.0], requires_grad=True)
        masks = torch.zeros(2, 1, 8, 8)
        has_calc = torch.zeros(2)
        weights = torch.ones(2)

        loss = self.mod.joint_loss(
            {"seg": seg_logits, "cls": cls_logits},
            masks, has_calc, weights,
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

    def test_joint_loss_positive_only_batch(self):
        seg_logits = torch.full((2, 1, 8, 8), -3.0, requires_grad=True)
        cls_logits = torch.tensor([2.0, 2.0], requires_grad=True)
        masks = torch.zeros(2, 1, 8, 8)
        masks[:, :, :3, :3] = 1.0
        has_calc = torch.ones(2)
        weights = torch.ones(2)

        loss = self.mod.joint_loss(
            {"seg": seg_logits, "cls": cls_logits},
            masks, has_calc, weights,
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

    def test_gated_metrics_zeroed_when_cls_low(self):
        seg_probs = torch.full((2, 1, 4, 4), 0.9)
        cls_probs = torch.tensor([0.1, 0.9])  # frame 0 below gate, frame 1 above
        masks = torch.zeros(2, 1, 4, 4)
        masks[1, :, :2, :2] = 1.0  # only frame 1 is positive
        has_calc = torch.tensor([0.0, 1.0])

        _, det = self.mod._gated_metrics_for_threshold(
            seg_probs, cls_probs, masks, has_calc,
            seg_threshold=0.5, cls_threshold=0.5,
        )
        # frame 0: negative GT, gate=0 so pred=0 → TN
        # frame 1: positive GT, gate=1, seg above thr → TP
        self.assertEqual(det["tn"], 1)
        self.assertEqual(det["tp"], 1)
        self.assertEqual(det["fp"], 0)
        self.assertEqual(det["fn"], 0)


if __name__ == "__main__":
    unittest.main()
