"""
Test script to verify adapter implementation and parameter counts.
"""
import torch
import sys
sys.path.insert(0, str(__file__).rsplit('\\', 1)[0])

from models.mae_hybrid_v2 import hybrid_mae_vit_small_patch8
import config_v2 as config


def test_adapter_model():
    print("=" * 80)
    print("Testing Adapter Implementation")
    print("=" * 80)

    # Test 1: Full training (no adapter)
    print("\n[Test 1] Full Training (no adapter)")
    model_full = hybrid_mae_vit_small_patch8(
        img_size=config.IMG_SIZE,
        in_chans=config.IN_CHANS,
        use_adapter=False,
    )
    stats_full = model_full.get_trainable_params()
    print(f"  Total params: {stats_full['total']:,}")
    print(f"  Trainable params: {stats_full['trainable']:,} ({stats_full['trainable_ratio']:.2f}%)")

    # Test 2: Adapter tuning (adapter_only)
    print("\n[Test 2] Adapter Tuning (freeze encoder, train adapter + decoder)")
    model_adapter = hybrid_mae_vit_small_patch8(
        img_size=config.IMG_SIZE,
        in_chans=config.IN_CHANS,
        use_adapter=True,
        adapter_bottleneck=64,
    )
    model_adapter.freeze_encoder(freeze_mode="adapter_only")
    stats_adapter = model_adapter.get_trainable_params()
    print(f"  Total params: {stats_adapter['total']:,}")
    print(f"  Trainable params: {stats_adapter['trainable']:,} ({stats_adapter['trainable_ratio']:.2f}%)")
    print(f"  Adapter params: {stats_adapter['adapter']:,}")
    print(f"  Decoder params: {stats_adapter['decoder']:,}")

    # Test 3: Frozen encoder (no adapter)
    print("\n[Test 3] Frozen Encoder (train decoder only)")
    model_frozen = hybrid_mae_vit_small_patch8(
        img_size=config.IMG_SIZE,
        in_chans=config.IN_CHANS,
        use_adapter=False,
    )
    model_frozen.freeze_encoder(freeze_mode="full")
    stats_frozen = model_frozen.get_trainable_params()
    print(f"  Total params: {stats_frozen['total']:,}")
    print(f"  Trainable params: {stats_frozen['trainable']:,} ({stats_frozen['trainable_ratio']:.2f}%)")

    # Test 4: Forward pass
    print("\n[Test 4] Forward Pass Test")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_adapter = model_adapter.to(device)

    dummy_img = torch.randn(2, 1, 256, 256).to(device)
    dummy_mask = torch.ones(2, 1, 256, 256).to(device)

    with torch.no_grad():
        out = model_adapter(dummy_img, dummy_mask, mask_ratio=0.65)

    print(f"  Input shape: {dummy_img.shape}")
    print(f"  Output pred shape: {out['pred'].shape}")
    print(f"  Output mask shape: {out['mask'].shape}")
    print(f"  MSE loss: {out['loss_mse'].item():.6f}")
    print("  ✓ Forward pass successful!")

    # Summary
    print("\n" + "=" * 80)
    print("Summary: Parameter Efficiency")
    print("=" * 80)
    print(f"Full training:     {stats_full['trainable']:,} params (100.00%)")
    print(f"Adapter tuning:    {stats_adapter['trainable']:,} params ({stats_adapter['trainable_ratio']:.2f}%)")
    print(f"Frozen encoder:    {stats_frozen['trainable']:,} params ({stats_frozen['trainable_ratio']:.2f}%)")
    print(f"\nAdapter efficiency: {stats_adapter['trainable'] / stats_full['trainable'] * 100:.2f}% of full training")
    print(f"Speed-up estimate: ~{stats_full['trainable'] / stats_adapter['trainable']:.1f}x faster")
    print("=" * 80)


if __name__ == "__main__":
    test_adapter_model()
