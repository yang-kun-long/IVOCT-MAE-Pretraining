import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


class SegLossModeTests(unittest.TestCase):
    def test_config_exposes_loss_mode_knobs(self):
        content = (ROOT / "seven" / "seg" / "config_seg.py").read_text(encoding="utf-8")
        self.assertIn("LOSS_MODE", content)
        self.assertIn("TVERSKY_ALPHA", content)
        self.assertIn("TVERSKY_BETA", content)
        self.assertIn("FOCAL_TVERSKY_GAMMA", content)

    def test_seg_losses_support_tversky_variants(self):
        content = (ROOT / "seven" / "seg" / "utils" / "seg_losses.py").read_text(encoding="utf-8")
        self.assertIn("def tversky_loss", content)
        self.assertIn("def focal_tversky_loss", content)
        self.assertIn("loss_mode", content)
        self.assertIn("focal_tversky", content)


if __name__ == "__main__":
    unittest.main()
