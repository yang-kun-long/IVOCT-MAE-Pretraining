# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Session Bootstrap

Read `.claude/PROJECT_INIT.md` first for the zero-context handoff. It records the current runnable state of the repository, including code-document mismatches that are important for safe continuation.
If `CLAUDE.md`, other docs, and the code disagree, trust `.claude/PROJECT_INIT.md` and the current code.

## Project Overview

This is an IVOCT (Intravascular Optical Coherence Tomography) medical image analysis project using a Hybrid Masked Autoencoder (MAE) with Vision Transformer for self-supervised pretraining. The goal is to learn robust representations from unlabeled IVOCT images through reconstruction.

## Architecture

### Core Components

**Hybrid MAE Architecture** (`seven/models/mae_hybrid_v2.py`):
- ConvStem patch embedding (256×256 → 32×32 token grid via 3-stage convolution)
- ViT encoder with sinusoidal positional embeddings
- Lightweight decoder for reconstruction
- Foreground-aware masking strategy that biases masking toward tissue regions

**Training Pipeline**:
1. Images preprocessed with center ROI cropping (removes outer artifacts/text)
2. Foreground masks computed to identify tissue vs background
3. Patches masked with foreground bias (50% mask ratio, 60% bias toward tissue)
4. Encoder processes visible patches → Decoder reconstructs masked regions
5. Multi-component loss: MSE + SSIM + Gradient (weighted 1.0 + 0.7 + 0.2)

**Data Flow**:
```
DATA/P00X/Data/*.jpg → IVOCTPretrainDatasetV2 → DataLoader
                          ↓
                    center crop + resize
                          ↓
                    foreground mask
                          ↓
                    HybridMAEViT (encoder + decoder)
                          ↓
                    reconstruction loss
```

### Directory Structure

- `DATA/`: Patient datasets (P001-P004)
  - `Data/`: Raw IVOCT images (897 total)
  - `label/`: Polygon annotations in LabelMe JSON format (43 files, all single-class "CA" / calcification)
  - `mask/`: Binary segmentation masks
- `seven/`: Main codebase
  - `config_v2.py`: Central configuration (paths, hyperparameters)
  - `train_mae_v2.py`: Training entry point
  - `infer_reconstruct_v2.py`: Inference/visualization script
  - `models/`: MAE architecture
  - `datasets/`: Data loading with preprocessing
  - `engine/`: Training loop
  - `utils/`: Losses, learning rate scheduling, visualization
  - `checkpoints_v2/`: Saved model weights
  - `logs_v2/`: Training logs
  - `recon_vis_v2/`: Reconstruction visualizations

## Common Commands

### Training
```bash
cd seven
python train_mae_v2.py
```
Trains from scratch for 400 epochs (v4 改进版) with:
- Batch size 32, 4 workers
- AdamW optimizer (lr=1.5e-4, warmup=10 epochs)
- Mixed precision (AMP) enabled
- Checkpoints saved every 10 epochs to `checkpoints_v2/`
- Visualizations saved every 10 epochs to `recon_vis_v2/`
- Training logs with timestamps to `logs_v2/train_YYYYMMDD_HHMMSS.log`

### Evaluation
```bash
cd seven
python evaluate_v2.py
```
Evaluates reconstruction quality with standard metrics:
- SSIM (Structural Similarity Index): 0-1, higher is better
- PSNR (Peak Signal-to-Noise Ratio): dB, higher is better
- MSE (Mean Squared Error): lower is better
Results saved to `logs_v2/evaluation_results.json`

### Inference/Visualization
```bash
cd seven
python infer_reconstruct_v2.py
```
Loads best checkpoint (`mae_v2_best.pth`) and generates 4-panel reconstruction visualization showing: original, masked, reconstruction, and error map.

### Configuration
All hyperparameters centralized in `seven/config_v2.py`:
- Modify `ROOT_DIR` if project location changes (auto-detected by default)
- Adjust `MASK_RATIO`, `FG_MASK_BIAS` for masking strategy
- Tune loss weights: `LAMBDA_MSE`, `LAMBDA_SSIM`, `LAMBDA_GRAD`
- Change model size via `EMBED_DIM`, `DEPTH`, `NUM_HEADS`, `PATCH_SIZE`

**v4 Improvements** (cumulative vs. original; see IMPROVEMENTS.md for details):
- Patch size: 8 → 16 (reduces checkerboard artifacts)
- Decoder: 4 layers → 8 layers (better reconstruction)
- Mask ratio: 70% → 50% (v3 was 60%; further lowered for easier training)
- Loss weights: SSIM 0.2 → 0.7 (v3 was 0.5), Grad 0.1 → 0.2 (better structure/edges)
- Epochs: 200 → 400 (v3 was 300; more thorough training)

## Output Locations & Packaging

All training outputs are centralized under `seven/` for easy server-to-local transfer:

**Pretraining (MAE)** — configured in `seven/config_v2.py`:
- `seven/checkpoints_v2/` — `mae_v2_epoch_*.pth` (every 10 epochs), `mae_v2_best.pth`, `mae_v2_encoder_only.pth`
- `seven/logs_v2/` — `train_YYYYMMDD_HHMMSS.log` (text), `train_log_v2.json` (metrics)
- `seven/recon_vis_v2/` — `recon_v2_epoch_*.png` (training vis), `reconstruction_check_v2.png` (inference)

**Segmentation** — configured in `seven/seg/config_seg.py`:
- `seven/seg/checkpoints/` — `seg_fold{N}_best.pth` (one per LOPO fold)
- `seven/seg/logs/` — `results_{split_mode}_YYYYMMDD_HHMMSS.json`
- `seven/seg/vis/fold_{N}/` — `epoch_*.png` per fold

**Packaging for download**:
```bash
# All outputs (pretrain + segmentation)
cd seven && tar -czf all_outputs.tar.gz checkpoints_v2/ logs_v2/ recon_vis_v2/ seg/

# Pretrain only
cd seven && tar -czf pretrain_outputs.tar.gz checkpoints_v2/ logs_v2/ recon_vis_v2/

# Segmentation only
cd seven && tar -czf seg_outputs.tar.gz seg/checkpoints/ seg/logs/ seg/vis/

# Essentials only (best model + final logs)
cd seven && tar -czf essentials.tar.gz \
  checkpoints_v2/mae_v2_best.pth checkpoints_v2/mae_v2_encoder_only.pth \
  logs_v2/train_log_v2.json seg/checkpoints/ seg/logs/
```

## Key Design Decisions

**Foreground-Aware Masking**: Unlike standard MAE which masks randomly, this implementation biases masking toward tissue regions (controlled by `FG_MASK_BIAS=0.6`). This forces the model to focus on reconstructing clinically relevant areas rather than background.

**Hybrid Architecture**: Uses ConvStem instead of linear patch embedding to better capture local patterns in medical images. The 3-stage convolution (256→128→64→32) provides inductive bias for spatial hierarchies.

**Multi-Component Loss**: Combines pixel-level MSE with perceptual SSIM and gradient loss. The gradient term preserves edge sharpness critical for identifying tissue boundaries in IVOCT.

**ROI Preprocessing**: Center crops to 86% of original size to remove catheter artifacts and text overlays common in IVOCT acquisitions.

## Data Format

**Images**: IVOCT cross-sections (`.jpg`), 1024×1024, stored as 3-channel color JPEGs in Abbott OCT's orange false-color palette. Dataset loader converts to single-channel grayscale via `.convert("L")` before training. Each frame contains an Abbott logo and "1 mm" scale bar near the bottom-right corner (largely survives the 86% ROI crop), plus a central catheter artifact (concentric circles).

**Labels**: LabelMe JSON format with polygon annotations. Each shape has:
- `label`: Tissue type (e.g., "CA" for calcification)
- `points`: List of [x, y] coordinates defining polygon
- `shape_type`: "polygon"

**Masks**: Binary PNG masks (0/255) corresponding to labeled regions, co-registered to the 1024×1024 original. Filename convention: `IMG-0001-XXXXX_mask.png` (note the `_mask` suffix, differs from the source `.jpg` basename).

## Notes

- Model expects single-channel grayscale input (IN_CHANS=1)
- Patch size fixed at 16 (changing requires modifying ConvStem architecture)
- Training uses CUDA by default; falls back to CPU if unavailable
- Best model selected by lowest total loss (MSE + SSIM + Gradient)
