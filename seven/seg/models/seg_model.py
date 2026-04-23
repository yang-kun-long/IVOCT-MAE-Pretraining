import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import importlib.util
from pathlib import Path

# Load mae_hybrid_v2 directly by file path to avoid models/ name collision
_mae_path = Path(__file__).parent.parent.parent / "models" / "mae_hybrid_v2.py"
_spec = importlib.util.spec_from_file_location("mae_hybrid_v2", _mae_path)
_mae_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mae_module)
HybridMAEViT = _mae_module.HybridMAEViT


class PrototypicalSegHead(nn.Module):
    """
    Prototypical segmentation head with learnable projection.

    Projects encoder features into a discriminative space,
    then segments via prototype matching.
    """
    def __init__(self, feat_dim=384, proj_dim=128, upsample_factor=8):
        super().__init__()
        self.upsample_factor = upsample_factor

        # Learnable projection: maps encoder features to discriminative space
        self.proj = nn.Sequential(
            nn.Conv2d(feat_dim, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, proj_dim, kernel_size=1),
        )

        self.temperature = nn.Parameter(torch.tensor(10.0))

    def extract_prototypes(self, support_feats, support_masks):
        """
        Args:
            support_feats: [K, proj_dim, H, W] projected support features
            support_masks: [K, 1, H_mask, W_mask] support masks
        Returns:
            fg_proto: [proj_dim]
            bg_proto: [proj_dim]
        """
        K, C, H, W = support_feats.shape

        masks_down = F.interpolate(support_masks.float(), size=(H, W), mode='nearest')
        feats_flat = support_feats.view(K, C, -1)   # [K, C, H*W]
        masks_flat = masks_down.view(K, 1, -1)       # [K, 1, H*W]

        fg_mask = masks_flat > 0.5
        fg_proto = (feats_flat * fg_mask.float()).sum(dim=(0, 2)) / fg_mask.float().sum(dim=(0, 2)).clamp(min=1)

        bg_mask = ~fg_mask
        bg_proto = (feats_flat * bg_mask.float()).sum(dim=(0, 2)) / bg_mask.float().sum(dim=(0, 2)).clamp(min=1)

        return fg_proto, bg_proto

    def forward(self, query_feats, support_feats, support_masks):
        """
        Args:
            query_feats:   [B, feat_dim, H, W]
            support_feats: [K, feat_dim, H, W]
            support_masks: [K, 1, H_mask, W_mask]
        Returns:
            logits: [B, 1, H*upsample, W*upsample]
        """
        # Project to discriminative space
        query_proj = self.proj(query_feats)      # [B, proj_dim, H, W]
        support_proj = self.proj(support_feats)  # [K, proj_dim, H, W]

        B, C, H, W = query_proj.shape

        # Extract prototypes from projected support features
        fg_proto, bg_proto = self.extract_prototypes(support_proj, support_masks)

        # Cosine similarity
        query_norm = F.normalize(query_proj, dim=1)           # [B, C, H, W]
        fg_norm = F.normalize(fg_proto, dim=0).view(1, C, 1, 1)
        bg_norm = F.normalize(bg_proto, dim=0).view(1, C, 1, 1)

        fg_sim = (query_norm * fg_norm).sum(dim=1, keepdim=True)  # [B, 1, H, W]
        bg_sim = (query_norm * bg_norm).sum(dim=1, keepdim=True)

        logits = (fg_sim - bg_sim) * self.temperature

        # Upsample to original image size
        logits = F.interpolate(logits, scale_factor=self.upsample_factor, mode='bilinear', align_corners=False)

        return logits


class MAESegmenter(nn.Module):
    """
    Prototypical segmentation model using pretrained MAE encoder.

    Architecture:
        - Encoder: HybridMAEViT (frozen, produces 32x32 token grid at 384 dim)
        - Head: PrototypicalSegHead (computes prototypes from support set)
    """
    def __init__(self, mae_checkpoint_path, freeze_encoder=True):
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

        # Freeze encoder (recommended for few-shot learning)
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("Encoder frozen")

        # Prototypical segmentation head with learnable projection
        self.seg_head = PrototypicalSegHead(feat_dim=384, proj_dim=128, upsample_factor=8)

    def extract_features(self, x):
        """
        Extract features from encoder.

        Args:
            x: [B, 1, 256, 256] input images

        Returns:
            [B, 384, 32, 32] spatial features
        """
        B = x.shape[0]

        # Bypass random_masking to preserve spatial token order
        with torch.no_grad():
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
        features = tokens.transpose(1, 2).reshape(B, 384, h, w)  # [B, 384, 32, 32]

        return features

    def forward(self, query_imgs, support_imgs, support_masks):
        """
        Forward pass for prototypical segmentation.

        Args:
            query_imgs: [B, 1, 256, 256] query images to segment
            support_imgs: [K, 1, 256, 256] support images
            support_masks: [K, 1, 256, 256] support masks (0/1)

        Returns:
            [B, 1, 256, 256] segmentation logits (before sigmoid)
        """
        # Extract features
        query_feats = self.extract_features(query_imgs)      # [B, 384, 32, 32]
        support_feats = self.extract_features(support_imgs)  # [K, 384, 32, 32]

        # Compute prototypes and segment
        logits = self.seg_head(query_feats, support_feats, support_masks)

        return logits
