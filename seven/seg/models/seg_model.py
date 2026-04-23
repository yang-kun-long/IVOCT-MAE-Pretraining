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
    Prototypical segmentation head for few-shot learning.

    Computes foreground/background prototypes from support set,
    then segments query images via cosine similarity.
    """
    def __init__(self, feat_dim=384, upsample_factor=8):
        super().__init__()
        self.feat_dim = feat_dim
        self.upsample_factor = upsample_factor

        # Optional: learnable temperature for similarity scaling
        self.temperature = nn.Parameter(torch.tensor(10.0))

    def extract_prototypes(self, support_feats, support_masks):
        """
        Extract foreground and background prototypes from support set.

        Args:
            support_feats: [K, C, H, W] support features
            support_masks: [K, 1, H_mask, W_mask] support masks (0/1)

        Returns:
            fg_proto: [C] foreground prototype
            bg_proto: [C] background prototype
        """
        K, C, H, W = support_feats.shape

        # Downsample masks to match feature resolution
        masks_down = F.interpolate(
            support_masks.float(),
            size=(H, W),
            mode='nearest'
        )  # [K, 1, H, W]

        # Flatten spatial dimensions
        feats_flat = support_feats.view(K, C, -1)  # [K, C, H*W]
        masks_flat = masks_down.view(K, 1, -1)     # [K, 1, H*W]

        # Compute foreground prototype (masked average pooling)
        fg_mask = masks_flat > 0.5  # [K, 1, H*W]
        fg_count = fg_mask.sum(dim=(0, 2), keepdim=True).clamp(min=1)  # [1, 1, 1]
        fg_sum = (feats_flat * fg_mask.float()).sum(dim=(0, 2))  # [C]
        fg_proto = fg_sum / fg_count.squeeze()  # [C]

        # Compute background prototype
        bg_mask = ~fg_mask
        bg_count = bg_mask.sum(dim=(0, 2), keepdim=True).clamp(min=1)
        bg_sum = (feats_flat * bg_mask.float()).sum(dim=(0, 2))
        bg_proto = bg_sum / bg_count.squeeze()

        return fg_proto, bg_proto

    def forward(self, query_feats, support_feats, support_masks):
        """
        Segment query images using prototypes from support set.

        Args:
            query_feats: [B, C, H, W] query features
            support_feats: [K, C, H, W] support features
            support_masks: [K, 1, H_mask, W_mask] support masks

        Returns:
            logits: [B, 1, H_out, W_out] segmentation logits
        """
        B, C, H, W = query_feats.shape

        # Extract prototypes
        fg_proto, bg_proto = self.extract_prototypes(support_feats, support_masks)

        # Normalize features and prototypes for cosine similarity
        query_norm = F.normalize(query_feats, dim=1)  # [B, C, H, W]
        fg_proto_norm = F.normalize(fg_proto.unsqueeze(0), dim=1)  # [1, C]
        bg_proto_norm = F.normalize(bg_proto.unsqueeze(0), dim=1)

        # Compute cosine similarity
        # query_norm: [B, C, H, W] -> [B, C, H*W]
        # proto_norm: [1, C] -> [C, 1]
        query_flat = query_norm.view(B, C, -1)  # [B, C, H*W]

        fg_sim = torch.matmul(fg_proto_norm, query_flat)  # [1, H*W]
        bg_sim = torch.matmul(bg_proto_norm, query_flat)

        fg_sim = fg_sim.view(B, 1, H, W)  # [B, 1, H, W]
        bg_sim = bg_sim.view(B, 1, H, W)

        # Compute logits: fg_sim - bg_sim, scaled by temperature
        logits = (fg_sim - bg_sim) * self.temperature

        # Upsample to original image size
        H_out = H * self.upsample_factor
        W_out = W * self.upsample_factor
        logits = F.interpolate(logits, size=(H_out, W_out), mode='bilinear', align_corners=False)

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

        # Prototypical segmentation head
        self.seg_head = PrototypicalSegHead(feat_dim=384, upsample_factor=8)

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
