"""
Progressive unfreezing utilities for segmentation training.

Usage:
    from progressive_unfreezing import setup_progressive_unfreezing

    # Stage 1: Freeze encoder
    setup_progressive_unfreezing(model, stage="freeze_all")

    # Stage 2: Unfreeze adapters
    setup_progressive_unfreezing(model, stage="unfreeze_adapters")

    # Stage 3: Unfreeze top layers
    setup_progressive_unfreezing(model, stage="unfreeze_top_layers", num_layers=3)

    # Stage 4: Unfreeze all
    setup_progressive_unfreezing(model, stage="unfreeze_all")
"""

import torch.nn as nn


def count_parameters(model):
    """Count trainable and total parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total': total,
        'trainable': trainable,
        'trainable_ratio': 100.0 * trainable / total if total > 0 else 0
    }


def setup_progressive_unfreezing(model, stage="freeze_all", num_layers=3):
    """
    Setup progressive unfreezing for segmentation model.

    Args:
        model: MAESegmenter model
        stage: One of ["freeze_all", "unfreeze_adapters", "unfreeze_top_layers", "unfreeze_all"]
        num_layers: Number of top layers to unfreeze (for "unfreeze_top_layers")

    Returns:
        dict: Parameter statistics
    """

    if stage == "freeze_all":
        # Stage 1: Freeze entire encoder, train decoder only
        print("=" * 80)
        print("Stage 1: Freeze Encoder (train decoder only)")
        print("=" * 80)

        for param in model.encoder.parameters():
            param.requires_grad = False

        for param in model.decoder.parameters():
            param.requires_grad = True

    elif stage == "unfreeze_adapters":
        # Stage 2: Unfreeze adapters, keep encoder frozen
        print("=" * 80)
        print("Stage 2: Unfreeze Adapters (fine-tune adapters + decoder)")
        print("=" * 80)

        for name, param in model.encoder.named_parameters():
            if 'adapter' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        for param in model.decoder.parameters():
            param.requires_grad = True

    elif stage == "unfreeze_top_layers":
        # Stage 3: Unfreeze top N transformer layers
        print("=" * 80)
        print(f"Stage 3: Unfreeze Top {num_layers} Layers (fine-tune top layers + adapters + decoder)")
        print("=" * 80)

        total_blocks = len(model.encoder.blocks)
        unfreeze_from = total_blocks - num_layers

        # Freeze bottom layers
        for name, param in model.encoder.named_parameters():
            param.requires_grad = False

        # Unfreeze top layers and adapters
        for i, block in enumerate(model.encoder.blocks):
            if i >= unfreeze_from:
                for param in block.parameters():
                    param.requires_grad = True

        # Unfreeze norm layer
        for param in model.encoder.norm.parameters():
            param.requires_grad = True

        # Unfreeze decoder
        for param in model.decoder.parameters():
            param.requires_grad = True

    elif stage == "unfreeze_all":
        # Stage 4: Unfreeze everything
        print("=" * 80)
        print("Stage 4: Unfreeze All (full fine-tuning)")
        print("=" * 80)

        for param in model.parameters():
            param.requires_grad = True

    else:
        raise ValueError(f"Unknown stage: {stage}")

    # Count parameters
    stats = count_parameters(model)

    # Count encoder/decoder separately
    encoder_total = sum(p.numel() for p in model.encoder.parameters())
    encoder_trainable = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    decoder_total = sum(p.numel() for p in model.decoder.parameters())
    decoder_trainable = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)

    print(f"\nParameter Statistics:")
    print(f"  Total: {stats['total']:,}")
    print(f"  Trainable: {stats['trainable']:,} ({stats['trainable_ratio']:.2f}%)")
    print(f"\n  Encoder: {encoder_trainable:,} / {encoder_total:,} trainable")
    print(f"  Decoder: {decoder_trainable:,} / {decoder_total:,} trainable")
    print("=" * 80)

    return stats


def get_recommended_lr(stage):
    """Get recommended learning rate for each stage."""
    lr_map = {
        "freeze_all": 1e-3,          # Decoder only, can use larger LR
        "unfreeze_adapters": 5e-5,   # Adapters are sensitive
        "unfreeze_top_layers": 3e-5, # Top layers need careful tuning
        "unfreeze_all": 1e-5,        # Full model, very small LR
    }
    return lr_map.get(stage, 1e-4)


def get_recommended_epochs(stage):
    """Get recommended epochs for each stage."""
    epochs_map = {
        "freeze_all": 100,
        "unfreeze_adapters": 50,
        "unfreeze_top_layers": 50,
        "unfreeze_all": 30,
    }
    return epochs_map.get(stage, 50)


if __name__ == "__main__":
    # Example usage
    print("Progressive Unfreezing Stages:")
    print("\nStage 1: freeze_all")
    print(f"  - LR: {get_recommended_lr('freeze_all')}")
    print(f"  - Epochs: {get_recommended_epochs('freeze_all')}")

    print("\nStage 2: unfreeze_adapters")
    print(f"  - LR: {get_recommended_lr('unfreeze_adapters')}")
    print(f"  - Epochs: {get_recommended_epochs('unfreeze_adapters')}")

    print("\nStage 3: unfreeze_top_layers")
    print(f"  - LR: {get_recommended_lr('unfreeze_top_layers')}")
    print(f"  - Epochs: {get_recommended_epochs('unfreeze_top_layers')}")

    print("\nStage 4: unfreeze_all")
    print(f"  - LR: {get_recommended_lr('unfreeze_all')}")
    print(f"  - Epochs: {get_recommended_epochs('unfreeze_all')}")
