#!/usr/bin/env python
"""
Diagnostic script to check segmentation training issues.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(1, str(Path(__file__).parent.parent))

import torch
import numpy as np
from torch.utils.data import DataLoader

import config_seg as config
from datasets import IVOCTSegDataset
from models import MAESegmenter

print("="*80)
print("Segmentation Training Diagnostics")
print("="*80)

# 1. Check dataset
print("\n[1] Dataset Check")
print("-"*80)
train_dataset = IVOCTSegDataset(
    config.DATA_DIR, ['P002', 'P003'],
    img_size=config.IMG_SIZE,
    crop_ratio=config.ROI_CROP_RATIO,
    is_train=False
)
print(f"Train samples: {len(train_dataset)}")

# Check foreground ratio
fg_ratios = []
for i in range(min(20, len(train_dataset))):
    sample = train_dataset[i]
    mask = sample['mask'].numpy()
    fg_ratio = mask.mean()
    fg_ratios.append(fg_ratio)
    if i < 5:
        print(f"  Sample {i}: fg_ratio={fg_ratio:.4f}, shape={mask.shape}")

print(f"\nForeground ratio statistics:")
print(f"  Mean: {np.mean(fg_ratios):.4f}")
print(f"  Std:  {np.std(fg_ratios):.4f}")
print(f"  Min:  {np.min(fg_ratios):.4f}")
print(f"  Max:  {np.max(fg_ratios):.4f}")

# 2. Check model
print("\n[2] Model Check")
print("-"*80)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

try:
    model = MAESegmenter(
        config.MAE_CHECKPOINT,
        patch_size=config.PATCH_SIZE,
        freeze_encoder=config.FREEZE_ENCODER,
        use_adapter=config.USE_ADAPTER,
        adapter_bottleneck=config.ADAPTER_BOTTLENECK,
    ).to(device)
    print("✓ Model loaded successfully")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")

except Exception as e:
    print(f"✗ Model loading failed: {e}")
    sys.exit(1)

# 3. Test forward pass
print("\n[3] Forward Pass Test")
print("-"*80)
model.eval()
loader = DataLoader(train_dataset, batch_size=2, shuffle=False)
batch = next(iter(loader))

images = batch['image'].to(device)
masks = batch['mask'].to(device)

print(f"Input shape: {images.shape}")
print(f"Mask shape: {masks.shape}")
print(f"Mask range: [{masks.min():.3f}, {masks.max():.3f}]")
print(f"Mask mean: {masks.mean():.4f}")

with torch.no_grad():
    logits = model(images)
    probs = torch.sigmoid(logits)
    preds = (probs > 0.3).float()

print(f"\nOutput shape: {logits.shape}")
print(f"Logits range: [{logits.min():.3f}, {logits.max():.3f}]")
print(f"Probs range: [{probs.min():.3f}, {probs.max():.3f}]")
print(f"Probs mean: {probs.mean():.4f}")
print(f"Preds mean: {preds.mean():.4f}")

# 4. Check loss
print("\n[4] Loss Check")
print("-"*80)
from utils import seg_loss

loss = seg_loss(
    logits, masks,
    lambda_dice=config.LAMBDA_DICE,
    lambda_bce=config.LAMBDA_BCE,
    loss_mode=config.LOSS_MODE,
    tversky_alpha=config.TVERSKY_ALPHA,
    tversky_beta=config.TVERSKY_BETA,
    focal_tversky_gamma=config.FOCAL_TVERSKY_GAMMA,
)
print(f"Loss: {loss.item():.4f}")

# 5. Summary
print("\n" + "="*80)
print("Summary")
print("="*80)
print(f"✓ Dataset: {len(train_dataset)} samples, fg_ratio={np.mean(fg_ratios):.4f}")
print(f"✓ Model: {trainable_params:,} trainable params")
print(f"✓ Forward pass: probs_mean={probs.mean():.4f}")
print(f"✓ Loss: {loss.item():.4f}")

if probs.mean() < 0.01:
    print("\n⚠️  WARNING: Model predicts very low probabilities!")
    print("   Possible issues:")
    print("   - Model initialization problem")
    print("   - Encoder frozen too aggressively")
    print("   - Loss function not working properly")
elif probs.mean() > 0.5:
    print("\n⚠️  WARNING: Model predicts very high probabilities!")
    print("   Possible issues:")
    print("   - Model overfitting")
    print("   - Loss function too weak")
else:
    print("\n✓ Model predictions look reasonable")

print("="*80)
