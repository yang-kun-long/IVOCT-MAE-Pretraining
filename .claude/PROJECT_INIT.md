# Project Initialization and Handoff

This document is the zero-context bootstrap for future Claude/Codex sessions in this repository. Read this first, then use `CLAUDE.md` for the higher-level project description.

## Repository Purpose

This repository is an IVOCT self-supervised pretraining project built around a Hybrid MAE encoder-decoder, with a downstream calcification segmentation branch under `seven/seg/`.

The current codebase is small, but several documents describe an intended or historical state rather than the exact code path that runs today. Treat code as the source of truth.

## Top-Level Map

- `CLAUDE.md`: main Claude-facing repository guidance
- `.claude/REMOTE_CONTEXT.md`: remote server path, SSH helper usage, sync state, and server environment notes
- `MONITORING_CONTRACT.md`: progress/results JSON contract for the training monitor
- `tools/monitor/`: Flask monitor source imported from the server deployment
- `DEPLOY.md`: segmentation training/evaluation quick start
- `IMPROVEMENTS.md`: version history and intended MAE improvements
- `SEGMENTATION_NOTES.md`: the most useful bug log for the segmentation branch
- `result/train_log_v2.json`: historical MAE training metrics snapshot
- `seven/`: actual Python code for pretraining, evaluation, inference, and segmentation
- `.claude/settings.local.json`: local Claude permissions/preferences
- `scripts/remote_ops.py`: local SSH/SFTP helper for running server commands and transferring files

## Actual Runnable Entry Points

### MAE Pretraining

- Entry: `seven/train_mae_v2.py`
- Config: `seven/config_v2.py`
- Dataset loader: `seven/datasets/ivoct_pretrain_dataset_v2.py`
- Model: `seven/models/mae_hybrid_v2.py`
- Engine: `seven/engine/pretrain_engine_v2.py`

Run from repo root:

```powershell
Set-Location seven
python train_mae_v2.py
```

### MAE Evaluation

- Masked reconstruction: `seven/evaluate_masked.py`
- Full reconstruction: `seven/evaluate_full.py`
- Legacy equivalent: `seven/evaluate_v2.py`

### MAE Inference / Visualization

- Script: `seven/infer_reconstruct_v2.py`

### Segmentation

- Training: `seven/seg/train_seg.py`
- Eval: `seven/seg/eval_seg.py`
- Config: `seven/seg/config_seg.py`
- Model: `seven/seg/models/seg_model.py`

Run from repo root:

```powershell
Set-Location seven/seg
python train_seg.py --split single_holdout --holdout P004
```

## Source-of-Truth Findings

These are the most important facts for future work:

1. `seven/config_v2.py` does not match the instantiated MAE architecture.
   - Config says `PATCH_SIZE = 16`, `DECODER_EMBED_DIM = 384`, `DECODER_DEPTH = 8`, `EPOCHS = 400`.
   - Actual training/eval code calls `hybrid_mae_vit_small_patch8()` from `seven/models/mae_hybrid_v2.py`.
   - That factory hardcodes `patch_size=8`, `decoder_embed_dim=256`, `decoder_depth=4`.

2. `SEGMENTATION_NOTES.md` is correct about the patch-size mismatch.
   - `ConvStemPatchEmbed` asserts `patch_size == 8`.
   - Any new checkpoint-compatible segmentation work must continue to assume patch size 8 unless the MAE model is redesigned and retrained.

3. The historical MAE log in `result/train_log_v2.json` contains 200 epochs, not 400.
   - This means the saved log artifact reflects an earlier or interrupted training run.
   - Do not assume the repository already contains a finished 400-epoch v4 result.

4. The segmentation branch already includes two critical bug fixes.
   - Checkpoint path fixed to `seven/checkpoints_v2/mae_v2_best.pth` in `seven/seg/config_seg.py`.
   - Spatial token order preserved by bypassing `random_masking` in `seven/seg/models/seg_model.py`.

5. There is no dataset bundled in the repository snapshot.
   - All training and evaluation scripts expect a `DATA/` directory at repo root.
   - Pretraining scans `DATA/P00X/Data/*`.
   - Segmentation scans `DATA/P00X/mask/*_mask.png` and pairs them to `DATA/P00X/Data/*.jpg`.

## Expected Data Layout

The code expects:

```text
DATA/
  P001/
    Data/
    mask/
    label/
  P002/
  P003/
  P004/
```

Important details:

- Pretraining only needs `Data/`.
- Segmentation uses `mask/` and derives the image name by stripping `_mask` and appending `.jpg`.
- Labels are mentioned in docs as LabelMe JSON, but the segmentation code consumes binary PNG masks, not JSON directly.

## Environment Assumptions

Dependencies declared in `requirements.txt`:

- `torch>=2.0.0`
- `torchvision>=0.15.0`
- `timm>=0.9.0`
- `Pillow>=9.0.0`
- `numpy>=1.24.0`
- `matplotlib>=3.7.0`
- `tqdm>=4.65.0`

Additional undeclared dependency:

- `scikit-image` is required by `seven/evaluate_v2.py`, `seven/evaluate_masked.py`, and `seven/evaluate_full.py` because they import `skimage.metrics`.

## Practical Workflow Guidance

For future sessions, use this order:

1. Confirm whether `DATA/` exists locally.
2. Confirm whether `seven/checkpoints_v2/mae_v2_best.pth` exists before touching segmentation.
3. If modifying the MAE architecture, update all of the following together:
   - `seven/config_v2.py`
   - `seven/models/mae_hybrid_v2.py`
   - `CLAUDE.md`
   - `IMPROVEMENTS.md`
   - any checkpoint-loading code in `seven/seg/`
4. If investigating segmentation quality, read `SEGMENTATION_NOTES.md` before changing loss or decoder code.

## Known Risks and Inconsistencies

- `CLAUDE.md` describes a stronger v4 configuration than the code currently instantiates.
- `DEPLOY.md` still refers to old output names like `seg_checkpoints/`, `seg_logs/`, and `seg_vis/`, while current code writes to `seven/seg/checkpoints/`, `seven/seg/logs/`, and `seven/seg/vis/`.
- Several helper scripts (`check_grad.py`, `check_masks.py`, `check_output.py`) contain hardcoded Linux paths from an earlier environment and are not portable as-is.
- File header comments still reference an old path like `F:\\CN_seg\\...`; treat them as historical residue only.

## Recommended First Actions In Any New Session

If the request is to continue development rather than just inspect:

1. Read `CLAUDE.md` and this file.
2. Inspect the exact files on the target path before assuming the docs are current.
3. Verify whether the user wants to preserve checkpoint compatibility with the existing MAE weights.
4. Prefer small, explicit fixes over broad refactors because the code-doc mismatch is already non-trivial.

## Local Claude Notes

`.claude/settings.local.json` currently allows broad local Python execution and basic git initialization/add operations in Claude. That file is environment-specific; do not rely on it as project logic.

For local remote access workflows, use `scripts/remote_ops.py`. Session credentials should be saved only to `.claude/remote_session.json`, which is git-ignored and intended to change across conversations or machines.

For server-specific context, read `.claude/REMOTE_CONTEXT.md` before issuing remote commands. It records the remote repo path, current sync state, safe update pattern, ignored large artifacts, and non-secret environment-variable conventions.
