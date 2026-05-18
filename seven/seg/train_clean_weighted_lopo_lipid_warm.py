"""Base trainer: v7 skip decoder + LOPO + encoder loaded from lipid-supervised pretrain.

Single-variable change vs v7 baseline:
  * Encoder initialization comes from `train_mae_lipid_supervised_v1.py`'s
    encoder ckpt instead of `mae_v2_best.pth`.
  * Decoder, sampler, loss, schedule, dataset, threshold sweep all identical
    to v7 (via the v5 base + skip decoder).

Override the lipid-warm encoder path with the LIPID_WARM_CKPT env var if
you want to point at a specific epoch (defaults to `encoder_best.pth`).
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(1, str(Path(__file__).parent.parent))

import train_clean_weighted_4fold_v5 as base
from models import MAELipidWarmSkipSegmenter


LIPID_WARM_CKPT_PATH = os.environ.get(
    "LIPID_WARM_CKPT",
    str(Path("/root/CN_seg/seven/seg/checkpoints_lipid_supervised_v1/encoder_best.pth")),
)


class _LipidWarmModelFactory:
    """Adapter so v5's `MAESegmenter(mae_path, ...)` call gives us a model
    whose encoder is initialized from the lipid-supervised checkpoint."""

    def __call__(self, mae_checkpoint_path, **kwargs):
        return MAELipidWarmSkipSegmenter(
            mae_checkpoint_path=mae_checkpoint_path,
            lipid_warm_encoder_path=LIPID_WARM_CKPT_PATH,
            **kwargs,
        )


base.EXPERIMENT_PREFIX = "clean_weighted_lopo_lipid_warm"
base.RESULT_PREFIX = "results_clean_weighted_lopo_lipid_warm"
base.CHECKPOINT_PREFIX = "clean_weighted_lopo_lipid_warm"
base.VIS_PREFIX = "clean_weighted_lopo_lipid_warm"
base.MAESegmenter = _LipidWarmModelFactory()


def build_lopo_folds(all_patients=None):
    patients = base.clean_patients(all_patients)
    folds = []
    for idx, patient in enumerate(patients):
        folds.append({
            "fold": idx,
            "train_patients": [p for p in patients if p != patient],
            "val_patients": [patient],
        })
    base.validate_folds(folds, patients=patients, duplicate_groups=base.DUPLICATE_GROUPS)
    return folds


base.build_clean_folds = build_lopo_folds


if __name__ == "__main__":
    print(f"Using LIPID_WARM_CKPT: {LIPID_WARM_CKPT_PATH}")
    base.main()
