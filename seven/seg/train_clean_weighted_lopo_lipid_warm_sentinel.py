"""Core-4 sentinel wrapper for lipid-warm fine-tune.

Folds: 12 (P014), 7 (P008), 11 (P013), 5 (P006). Result split_mode ends in
`_sentinel` so the monitor's composite_config groups this as the Core-4 panel
(see `tools/monitor/app.py`).
"""

import train_clean_weighted_lopo_lipid_warm as warm  # warm.base is v5


SENTINEL_FOLD_INDICES = [12, 7, 11, 5]  # P014, P008, P013, P006


warm.base.EXPERIMENT_PREFIX = "clean_weighted_lopo_lipid_warm_sentinel"
warm.base.RESULT_PREFIX = "results_clean_weighted_lopo_lipid_warm_sentinel"
warm.base.CHECKPOINT_PREFIX = "clean_weighted_lopo_lipid_warm_sentinel"
warm.base.VIS_PREFIX = "clean_weighted_lopo_lipid_warm_sentinel"


def build_sentinel_folds(all_patients=None):
    all_folds = warm.build_lopo_folds(all_patients)
    return [f for f in all_folds if f["fold"] in SENTINEL_FOLD_INDICES]


warm.base.build_clean_folds = build_sentinel_folds


if __name__ == "__main__":
    warm.base.main()
