import torch
import torch.nn as nn
import sys
import importlib.util
from pathlib import Path

# Load mae_hybrid_v2 directly by file path to avoid models/ name collision
_mae_path = Path(__file__).parent.parent.parent / "models" / "mae_hybrid_v2.py"
_spec = importlib.util.spec_from_file_location("mae_hybrid_v2", _mae_path)
_mae_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mae_module)
HybridMAEViT = _mae_module.HybridMAEViT


class UpBlock(nn.Module):
    """Upsample block: bilinear 2x + double Conv + GN + GELU"""
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
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

        # Segmentation decoder: 32→64→128→256, with deeper double-conv blocks
        self.decoder = nn.Sequential(
            UpBlock(384, 256),   # 32 → 64
            UpBlock(256, 128),   # 64 → 128
            UpBlock(128, 64),    # 128 → 256
            nn.Conv2d(64, 1, kernel_size=1)
        )

    def forward(self, x):
        """
        Args:
            x: [B, 1, 256, 256] input images

        Returns:
            [B, 1, 256, 256] segmentation logits (before sigmoid)
        """
        B = x.shape[0]
        model_training = self.encoder.training

        # Bypass random_masking to preserve spatial token order
        with torch.no_grad() if not model_training else torch.enable_grad():
            x = self.encoder.patch_embed(x)
            x = x + self.encoder.pos_embed[:, 1:, :]
            cls_token = self.encoder.cls_token + self.encoder.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            for blk in self.encoder.blocks:
                x = blk(x)
            latent = self.encoder.norm(x)

        # latent: [B, 1025, 384] (1024 patches + 1 CLS token)
        # Remove CLS token and reshape to spatial grid
        tokens = latent[:, 1:, :]  # [B, 1024, 384]

        # Reshape to 2D grid: 1024 = 32 × 32 (from patch_size=8 on 256px image)
        h = w = 32
        tokens = tokens.transpose(1, 2).reshape(B, 384, h, w)  # [B, 384, 32, 32]

        # Decode to segmentation map
        logits = self.decoder(tokens)  # [B, 1, 256, 256]

        return logits
