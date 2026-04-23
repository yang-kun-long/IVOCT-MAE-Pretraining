import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


class SegStabilityPlanTests(unittest.TestCase):
    def test_config_exposes_stability_knobs(self):
        content = (ROOT / "seven" / "seg" / "config_seg.py").read_text(encoding="utf-8")
        self.assertIn("EVAL_THRESHOLD", content)
        self.assertIn("EARLY_STOPPING_PATIENCE", content)
        self.assertIn("MIN_EPOCHS", content)
        self.assertIn("BASE_LR = 1e-3", content)
        self.assertIn("EVAL_THRESHOLD = 0.3", content)

    def test_train_loop_uses_threshold_and_early_stopping(self):
        content = (ROOT / "seven" / "seg" / "train_seg.py").read_text(encoding="utf-8")
        self.assertIn("threshold=config.EVAL_THRESHOLD", content)
        self.assertIn("EARLY_STOPPING_PATIENCE", content)
        self.assertIn("MIN_EPOCHS", content)
        self.assertIn("patience_counter", content)
        self.assertIn("Early stopping", content)

    def test_decoder_keeps_batchnorm_for_checkpoint_compatibility(self):
        content = (ROOT / "seven" / "seg" / "models" / "seg_model.py").read_text(encoding="utf-8")
        self.assertIn("BatchNorm2d", content)
        self.assertNotIn("GroupNorm", content)

    def test_threshold_sweep_script_exists(self):
        script = ROOT / "scripts" / "sweep_seg_thresholds.py"
        self.assertTrue(script.exists(), f"Missing script: {script}")

    def test_threshold_sweep_script_has_checkpoint_compatibility_fallback(self):
        content = (ROOT / "scripts" / "sweep_seg_thresholds.py").read_text(encoding="utf-8")
        self.assertIn("strict=False", content)
        self.assertIn("running_mean", content)
        self.assertIn("num_batches_tracked", content)


if __name__ == "__main__":
    unittest.main()
