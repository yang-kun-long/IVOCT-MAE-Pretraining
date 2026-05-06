# Server Sync Notes 2026-05-06

Source server path: `/root/CN_seg`

## Git State

- Local branch before sync: `master` at `8110bbd`
- Server branch: `master` at `67f7f6b`
- The server was not ahead by committed Git history. Its progress was in uncommitted experiment files.
- Synced into local branch: `sync/server-progress-20260506`

## Synced

- Adapter tuning support in `seven/models/mae_hybrid_v2.py` and `seven/train_mae_v2.py`
- Segmentation adapter loading, configurable patch size, baseline `dice_bce`, and clean weighted 4-fold scripts
- Progress tracking, notifier helpers, diagnostics, experiment docs, and script entry points
- Latest small JSON result summaries copied into this directory

## Latest Server Results

| Experiment | Mean Dice | Std Dice | Notes |
| --- | ---: | ---: | --- |
| `clean_weighted_4fold` | 0.4254 | 0.0883 | First clean weighted run |
| `clean_weighted_4fold_v2` | 0.4714 | 0.0558 | Best of the three synced runs |
| `clean_weighted_4fold_v3` | 0.4235 | 0.0583 | FP-penalizing Tversky variant |

## Not Synced To Git

These should stay on the server or move to external artifact storage, not normal Git:

- `seven/checkpoints_v2/*.pth`: 86-287 MB each
- `seven/seg/checkpoints/*.pth`: 166-222 MB each
- `DATA_backup_20260504/`
- `seven/seg/vis/`, `seven/recon_vis_v2/`
- training `.log` and `.pid` files
