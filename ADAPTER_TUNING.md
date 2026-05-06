# Adapter Tuning for IVOCT MAE Pretraining

## Overview

This document describes the **Adapter Tuning** implementation for parameter-efficient fine-tuning of the Hybrid MAE model on the expanded IVOCT dataset (4,554 images from 19 patients).

## Motivation

With the dataset expansion from 897 images (4 patients) to 4,554 images (19 patients), we introduce **Adapter Tuning** to:

1. **Efficiency**: Train only ~5-10% of parameters while maintaining performance
2. **Speed**: 2-3x faster training with reduced memory footprint
3. **Flexibility**: Preserve pretrained knowledge while adapting to new data
4. **Scalability**: Easy to add new adapters for future datasets without retraining the entire model

## Architecture

### Adapter Module

```python
class Adapter(nn.Module):
    """
    Bottleneck adapter inserted after each Transformer block.

    Architecture:
        Input (dim=384) → Down (bottleneck=64) → GELU → Up (dim=384) → Residual

    Parameters per adapter: 384*64 + 64*384 = 49,152
    Total adapters: 12 (one per encoder block)
    Total adapter params: ~590K
    """
```

**Key Features:**
- **Bottleneck design**: Compresses features to reduce parameters
- **Residual connection**: Preserves original features with learnable scaling
- **Small initialization**: Starts with minimal impact, gradually adapts

### Integration

Adapters are inserted **after the MLP** in each Transformer block:

```
Transformer Block:
  ├─ Attention
  ├─ MLP
  └─ Adapter (NEW) ← Only this is trainable in adapter mode
```

## Training Modes

### Mode 1: Full Training (Baseline)
```python
USE_ADAPTER = False
FREEZE_MODE = "none"
```
- **Trainable params**: ~90M (100%)
- **Use case**: Training from scratch or full fine-tuning
- **Training time**: Baseline (1x)

### Mode 2: Adapter Tuning (Recommended)
```python
USE_ADAPTER = True
FREEZE_MODE = "adapter_only"
```
- **Trainable params**: ~5M (5-6%)
  - 12 adapters: ~590K
  - Decoder: ~4.5M
- **Use case**: Adapting pretrained model to new data
- **Training time**: ~0.4x (2.5x faster)
- **Memory**: ~60% of full training

### Mode 3: Frozen Encoder
```python
USE_ADAPTER = False
FREEZE_MODE = "full"
```
- **Trainable params**: ~4.5M (5%)
- **Use case**: Decoder-only fine-tuning
- **Training time**: ~0.35x (3x faster)

## Configuration Changes

### Dataset Expansion Adjustments

| Parameter | Old (897 imgs) | New (4,554 imgs) | Reason |
|-----------|----------------|------------------|--------|
| `EPOCHS` | 400 | 200 | 5x data, reduce redundancy |
| `BATCH_SIZE` | 32 | 64 | Leverage 5090 32GB VRAM |
| `BASE_LR` | 1.5e-4 | 3e-4 | Scale with batch size |
| `WARMUP_EPOCHS` | 10 | 20 | Smoother warmup |
| `MASK_RATIO` | 0.50 | 0.65 | Harder task with more data |

### Adapter-Specific Parameters

```python
# Adapter configuration
USE_ADAPTER = True           # Enable adapter tuning
ADAPTER_BOTTLENECK = 64      # Bottleneck dimension (32/64/128)
FREEZE_MODE = "adapter_only" # Freeze encoder, train adapter + decoder
```

**Bottleneck dimension trade-off:**
- `32`: Fewer params (~300K), faster, may underfit
- `64`: Balanced (~590K), recommended
- `128`: More capacity (~1.2M), slower, may overfit

## Training Statistics

### Expected Performance (200 epochs)

| Mode | Params | Training Time | Memory | Expected SSIM |
|------|--------|---------------|--------|---------------|
| Full Training | 90M | ~40 hours | 28GB | 0.85-0.87 |
| **Adapter Tuning** | **5M** | **~16 hours** | **18GB** | **0.83-0.86** |
| Frozen Encoder | 4.5M | ~14 hours | 16GB | 0.80-0.83 |

*Estimates based on RTX 5090 32GB, batch size 64*

### Training Steps Comparison

| Dataset | Images | Batch | Steps/Epoch | Total Steps (200 ep) |
|---------|--------|-------|-------------|----------------------|
| Old | 897 | 32 | 28 | 5,600 |
| **New** | **4,554** | **64** | **71** | **14,200** |

## Usage

### 1. Test Adapter Implementation

```bash
cd seven
python test_adapter.py
```

This will verify:
- Parameter counts for each mode
- Forward pass functionality
- Adapter integration

### 2. Train with Adapter Tuning

```bash
cd seven
python train_mae_v2.py
```

The script will automatically:
- Load configuration from `config_v2.py`
- Create model with adapters
- Freeze encoder (keeping adapters trainable)
- Train with progress bars (tqdm)
- Save checkpoints every 10 epochs

### 3. Switch Training Modes

Edit `seven/config_v2.py`:

```python
# For full training
USE_ADAPTER = False
FREEZE_MODE = "none"

# For adapter tuning (recommended)
USE_ADAPTER = True
FREEZE_MODE = "adapter_only"

# For frozen encoder
USE_ADAPTER = False
FREEZE_MODE = "full"
```

## Experimental Design

### Baseline Comparisons

To validate adapter tuning effectiveness, run three experiments:

1. **Baseline**: Full training on 4,554 images
   - `USE_ADAPTER = False`, `FREEZE_MODE = "none"`
   - Expected: Best performance, longest training

2. **Ours**: Adapter tuning on 4,554 images
   - `USE_ADAPTER = True`, `FREEZE_MODE = "adapter_only"`
   - Expected: 95-98% of baseline performance, 2.5x faster

3. **Ablation**: Frozen encoder (no adapter)
   - `USE_ADAPTER = False`, `FREEZE_MODE = "full"`
   - Expected: Lower performance, fastest training

### Evaluation Metrics

- **Reconstruction quality**: SSIM, PSNR, MSE
- **Downstream task**: Segmentation Dice score (LOPO cross-validation)
- **Efficiency**: Training time, memory usage, parameter count

## Implementation Details

### Files Modified

1. **`seven/models/mae_hybrid_v2.py`**
   - Added `Adapter` class
   - Added `BlockWithAdapter` wrapper
   - Added `freeze_encoder()` method
   - Added `get_trainable_params()` statistics

2. **`seven/config_v2.py`**
   - Updated hyperparameters for 4,554 images
   - Added adapter configuration section
   - Optimized for RTX 5090 32GB

3. **`seven/train_mae_v2.py`**
   - Added tqdm import
   - Added adapter mode logging
   - Added parameter statistics display
   - Filter trainable parameters for optimizer

4. **`seven/engine/pretrain_engine_v2.py`**
   - Already has tqdm progress bars ✓

### Checkpoint Compatibility

Adapter checkpoints are **compatible** with full model checkpoints:

- **Loading pretrained encoder**: Adapter weights are optional, will be ignored if not present
- **Saving adapter model**: Includes both frozen encoder and trainable adapters
- **Encoder-only export**: Works the same, adapters are part of encoder blocks

## Future Extensions

### 1. Multiple Adapters

Train different adapters for different datasets:

```python
# Adapter for P001-P019 (calcification)
adapter_ca = load_adapter("adapter_ca.pth")

# Adapter for future P020-P030 (lipid)
adapter_lipid = load_adapter("adapter_lipid.pth")

# Switch adapters without retraining encoder
model.load_adapter(adapter_ca)
```

### 2. Adapter Fusion

Combine multiple adapters for multi-task learning:

```python
# Weighted combination of adapters
output = α * adapter_ca(x) + β * adapter_lipid(x)
```

### 3. Progressive Unfreezing

Gradually unfreeze encoder layers during training:

```python
# Epoch 0-50: Freeze all encoder
# Epoch 51-100: Unfreeze last 4 blocks
# Epoch 101-150: Unfreeze last 8 blocks
# Epoch 151-200: Unfreeze all
```

## References

- **Adapter Tuning**: Houlsby et al., "Parameter-Efficient Transfer Learning for NLP", ICML 2019
- **MAE**: He et al., "Masked Autoencoders Are Scalable Vision Learners", CVPR 2022
- **Medical Imaging Adapters**: Chen et al., "Towards Efficient Adaptation of Medical Foundation Models", arXiv 2023

## Troubleshooting

### Issue: Adapter not training

**Check:**
```python
# Verify adapter parameters are trainable
for name, param in model.named_parameters():
    if 'adapter' in name:
        print(f"{name}: requires_grad={param.requires_grad}")
```

### Issue: Out of memory

**Solutions:**
1. Reduce `BATCH_SIZE` from 64 to 48 or 32
2. Reduce `ADAPTER_BOTTLENECK` from 64 to 32
3. Enable gradient checkpointing (future work)

### Issue: Poor reconstruction quality

**Try:**
1. Increase `ADAPTER_BOTTLENECK` from 64 to 128
2. Increase `EPOCHS` from 200 to 300
3. Reduce `MASK_RATIO` from 0.65 to 0.60
4. Switch to full training mode

## Contact

For questions or issues, please check:
- Training logs in `seven/logs_v2/`
- Checkpoint statistics with `test_adapter.py`
- Configuration in `seven/config_v2.py`
