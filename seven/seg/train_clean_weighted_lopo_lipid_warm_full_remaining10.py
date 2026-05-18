"""Full-remaining-10 wrapper for lipid-warm fine-tune.

Folds: 0,1,2,3,4,6,10,15,16,17 (P001-P005, P007, P012, P017-P019).
The `_full_remaining10` suffix triggers monitor's `composite_config` to
combine with sentinel + extra4 into the Full LOPO18 panel.
"""

import train_clean_weighted_lopo_lipid_warm as warm


REMAINING10_FOLD_INDICES = [0, 1, 2, 3, 4, 6, 10, 15, 16, 17]


warm.base.EXPERIMENT_PREFIX = "clean_weighted_lopo_lipid_warm_full_remaining10"
warm.base.RESULT_PREFIX = "results_clean_weighted_lopo_lipid_warm_full_remaining10"
warm.base.CHECKPOINT_PREFIX = "clean_weighted_lopo_lipid_warm_full_remaining10"
warm.base.VIS_PREFIX = "clean_weighted_lopo_lipid_warm_full_remaining10"


def build_remaining10_folds(all_patients=None):
    all_folds = warm.build_lopo_folds(all_patients)
    return [f for f in all_folds if f["fold"] in REMAINING10_FOLD_INDICES]


warm.base.build_clean_folds = build_remaining10_folds


if __name__ == "__main__":
    warm.base.main()
