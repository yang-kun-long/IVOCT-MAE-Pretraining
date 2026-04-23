import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add parent directory to path to import from seven/
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.mae_hybrid_v2 import HybridMAEViT


class UpBlock(nn.Module):
    """Upsample block: bilinear 2x + Conv + BN + GELU"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x


class MAESegmenter(nn.Module):
    """
    Segmentation model using pretrained MAE encoder + UNet-style decoder.

    Architecture:
        - Encoder: HybridMAEViT (patch=8, produces 32x32 token grid at 384 dim)
        - Decoder: 4-stage upsampling (32→64→128→256) with channel reduction
    """
    def __init__(self, mae_checkpoint_path, freeze_encoder=False):
        super().__init__()

        # Build MAE encoder (must use patch=8 to match checkpoint)
        self.encoder = HybridMAEViT(
            img_size=256,
            patch_size=8,
            in_chans=1,
            embed_dim=384,
            depth=12,
            num_heads=6,
            decoder_embed_dim=256,
            decoder_depth=4,
            decoder_num_heads=8,
            mlp_ratio=4.0,
            norm_pix_loss=True,
            fg_mask_bias=0.6,
            mask_mode="foreground_aware"
        )

        # Load pretrained weights
        print(f"Loading MAE checkpoint from {mae_checkpoint_path}")
        ckpt = torch.load(mae_checkpoint_path, map_location='cpu', weights_only=False)
        self.encoder.load_state_dict(ckpt["model"], strict=True)
        print("MAE encoder loaded successfully")

        # Freeze encoder if requested
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("Encoder frozen")

        # Segmentation decoder
        # Input: [B, 384, 32, 32] from encoder tokens
        # Output: [B, 1, 256, 256] segmentation logits
        self.decoder = nn.Sequential(
            UpBlock(384, 192),   # 32 → 64
            UpBlock(192, 96),    # 64 → 128
            UpBlock(96, 48),     # 128 → 256
            nn.Conv2d(48, 1, kernel_size=1)  # Final 1x1 conv to single channel
        )

    def forward(self, x):
        """
        Args:
            x: [B, 1, 256, 256] input images

        Returns:
            [B, 1, 256, 256] segmentation logits (before sigmoid)
        """
        B = x.shape[0]

        # Create dummy foreground mask (all ones, so all tokens are kept)
        fg_mask = torch.ones(B, 1, 256, 256, device=x.device)

        # Run encoder with mask_ratio=0 (no masking, keep all tokens)
        with torch.set_grad_enabled(not self.encoder.training or True):
            latent, _, _ = self.encoder.forward_encoder(x, fg_mask, mask_ratio=0.0)

        # latent: [B, 1025, 384] (1024 patches + 1 CLS token)
        # Remove CLS token and reshape to spatial grid
        tokens = latent[:, 1:, :]  # [B, 1024, 384]

        # Reshape to 2D grid: 1024 = 32 × 32 (from patch_size=8 on 256px image)
        h = w = 32
        tokens = tokens.transpose(1, 2).reshape(B, 384, h, w)  # [B, 384, 32, 32]

        # Decode to segmentation map
        logits = self.decoder(tokens)  # [B, 1, 256, 256]

        return logits
