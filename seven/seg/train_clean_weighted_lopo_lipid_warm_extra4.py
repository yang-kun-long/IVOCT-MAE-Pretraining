"""Extra-4 wrapper for lipid-warm fine-tune.

Folds: 13 (P015), 9 (P010), 14 (P016), 8 (P009). `_extra4` suffix triggers
monitor's `composite_config` to reuse Core-4 sentinel results and surface the
combined Expanded-8 panel.
"""

import train_clean_weighted_lopo_lipid_warm as warm


EXTRA4_FOLD_INDICES = [13, 9, 14, 8]  # P015, P010, P016, P009


warm.base.EXPERIMENT_PREFIX = "clean_weighted_lopo_lipid_warm_extra4"
warm.base.RESULT_PREFIX = "results_clean_weighted_lopo_lipid_warm_extra4"
warm.base.CHECKPOINT_PREFIX = "clean_weighted_lopo_lipid_warm_extra4"
warm.base.VIS_PREFIX = "clean_weighted_lopo_lipid_warm_extra4"


def build_extra4_folds(all_patients=None):
    all_folds = warm.build_lopo_folds(all_patients)
    return [f for f in all_folds if f["fold"] in EXTRA4_FOLD_INDICES]


warm.base.build_clean_folds = build_extra4_folds


if __name__ == "__main__":
    warm.base.main()
