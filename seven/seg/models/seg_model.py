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
    """Upsample block: bilinear 2x + double Conv + BN + GELU"""
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
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
        - Encoder: HybridMAEViT (patch=16, produces 16x16 token grid at 384 dim)
        - Decoder: 4-stage upsampling (16→32→64→128→256) with channel reduction
    """
    def __init__(self, mae_checkpoint_path, patch_size=16, freeze_encoder=False,
                 use_adapter=False, adapter_bottleneck=64):
        super().__init__()

        # Build MAE encoder (must match checkpoint configuration)
        self.encoder = HybridMAEViT(
            img_size=256,
            patch_size=patch_size,
            in_chans=1,
            embed_dim=384,
            depth=12,
            num_heads=6,
            decoder_embed_dim=384,
            decoder_depth=8,
            decoder_num_heads=8,
            mlp_ratio=4.0,
            norm_pix_loss=True,
            fg_mask_bias=0.6,
            mask_mode="foreground_aware",
            use_adapter=use_adapter,
            adapter_bottleneck=adapter_bottleneck,
        )

        # Load pretrained weights
        print(f"Loading MAE checkpoint from {mae_checkpoint_path}")
        ckpt = torch.load(mae_checkpoint_path, map_location='cpu', weights_only=False)

        # Load encoder weights (including adapters if present)
        if "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt

        # Filter out decoder weights (we only need encoder)
        encoder_state_dict = {k: v for k, v in state_dict.items()
                             if not k.startswith('decoder') and not k.startswith('mask_token')}

        self.encoder.load_state_dict(encoder_state_dict, strict=False)
        print(f"MAE encoder loaded successfully (patch_size={patch_size}, use_adapter={use_adapter})")

        # Freeze encoder if requested
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("Encoder frozen")

        # Calculate spatial dimensions after patch embedding
        # patch_size=16: 256/16 = 16x16 grid
        # patch_size=8:  256/8  = 32x32 grid
        self.patch_size = patch_size
        self.grid_size = 256 // patch_size

        # Segmentation decoder: adapt to grid size
        if patch_size == 16:
            # 16→32→64→128→256 (4 upsampling stages)
            self.decoder = nn.Sequential(
                UpBlock(384, 256),   # 16 → 32
                UpBlock(256, 128),   # 32 → 64
                UpBlock(128, 64),    # 64 → 128
                UpBlock(64, 32),     # 128 → 256
                nn.Conv2d(32, 1, kernel_size=1)
            )
        elif patch_size == 8:
            # 32→64→128→256 (3 upsampling stages)
            self.decoder = nn.Sequential(
                UpBlock(384, 256),   # 32 → 64
                UpBlock(256, 128),   # 64 → 128
                UpBlock(128, 64),    # 128 → 256
                nn.Conv2d(64, 1, kernel_size=1)
            )
        else:
            raise ValueError(f"Unsupported patch_size: {patch_size}")

        # Initialize decoder properly
        self._init_decoder()

    def _init_decoder(self):
        """Initialize decoder weights properly for segmentation."""
        for m in self.decoder.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    # For final layer, initialize bias to predict low probability
                    # (since foreground is rare ~2-5%)
                    if m.out_channels == 1:  # Final layer
                        # sigmoid(-4) ≈ 0.018, close to actual fg_ratio of 0.025
                        nn.init.constant_(m.bias, -4.0)
                    else:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x: [B, 1, 256, 256] input images

        Returns:
            [B, 1, 256, 256] segmentation logits (before sigmoid)
        """
        B = x.shape[0]

        # Bypass random_masking to preserve spatial token order
        # Note: gradient flow is controlled by param.requires_grad, not torch.no_grad()
        x = self.encoder.patch_embed(x)
        x = x + self.encoder.pos_embed[:, 1:, :]
        cls_token = self.encoder.cls_token + self.encoder.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        for blk in self.encoder.blocks:
            x = blk(x)
        latent = self.encoder.norm(x)

        # latent: [B, N+1, 384] where N = grid_size^2
        # Remove CLS token and reshape to spatial grid
        tokens = latent[:, 1:, :]  # [B, N, 384]

        # Reshape to 2D grid
        h = w = self.grid_size
        tokens = tokens.transpose(1, 2).reshape(B, 384, h, w)  # [B, 384, h, w]

        # Decode to segmentation map
        logits = self.decoder(tokens)  # [B, 1, 256, 256]

        return logits
