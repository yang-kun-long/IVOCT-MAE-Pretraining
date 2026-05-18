import importlib.util
from pathlib import Path

import torch
import torch.nn as nn

from .seg_model import make_norm


_mae_path = Path(__file__).parent.parent.parent / "models" / "mae_hybrid_v2.py"
_spec = importlib.util.spec_from_file_location("mae_hybrid_v2", _mae_path)
_mae_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mae_module)
HybridMAEViT = _mae_module.HybridMAEViT


class SkipUpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, dropout=0.1, norm_type="batch"):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            make_norm(norm_type, out_channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            make_norm(norm_type, out_channels),
            nn.GELU(),
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class FinalUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1, norm_type="batch"):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            make_norm(norm_type, out_channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            make_norm(norm_type, out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.conv(self.up(x))


class SkipDecoder(nn.Module):
    def __init__(self, norm_type="batch"):
        super().__init__()
        self.up64 = SkipUpBlock(384, 64, 192, norm_type=norm_type)
        self.up128 = SkipUpBlock(192, 32, 96, norm_type=norm_type)
        self.up256 = FinalUpBlock(96, 48, norm_type=norm_type)
        self.head = nn.Conv2d(48, 1, kernel_size=1)

    def forward(self, tokens, skip64, skip128):
        x = self.up64(tokens, skip64)
        x = self.up128(x, skip128)
        x = self.up256(x)
        return self.head(x)


class MAESkipSegmenter(nn.Module):
    """MAE segmenter with conv-stem skip connections for small target detail."""

    def __init__(
        self,
        mae_checkpoint_path,
        patch_size=8,
        freeze_encoder=False,
        use_adapter=False,
        adapter_bottleneck=64,
        decoder_norm="batch",
    ):
        super().__init__()
        if patch_size != 8:
            raise ValueError("MAESkipSegmenter currently supports patch_size=8 only")

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

        print(f"Loading MAE checkpoint from {mae_checkpoint_path}")
        ckpt = torch.load(mae_checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["model"] if "model" in ckpt else ckpt
        encoder_state_dict = {
            k: v for k, v in state_dict.items()
            if not k.startswith("decoder") and not k.startswith("mask_token")
        }
        self.encoder.load_state_dict(encoder_state_dict, strict=False)
        print("MAE encoder loaded successfully for skip segmenter")

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("Encoder frozen")

        self.patch_size = patch_size
        self.grid_size = 256 // patch_size
        self.decoder = SkipDecoder(norm_type=decoder_norm)
        self._init_decoder()

    def _init_decoder(self):
        for module in self.decoder.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    if module.out_channels == 1:
                        nn.init.constant_(module.bias, -4.0)
                    else:
                        nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _patch_embed_with_skips(self, x):
        stem = self.encoder.patch_embed.stem
        x = stem[0](x)
        x = stem[1](x)
        skip128 = stem[2](x)
        x = stem[3](skip128)
        x = stem[4](x)
        skip64 = stem[5](x)
        x = stem[6](skip64)
        x = stem[7](x)
        patch32 = stem[8](x)
        tokens = patch32.flatten(2).transpose(1, 2)
        return tokens, patch32, skip64, skip128

    def forward(self, x):
        batch_size = x.shape[0]
        x, _, skip64, skip128 = self._patch_embed_with_skips(x)
        x = x + self.encoder.pos_embed[:, 1:, :]
        cls_token = self.encoder.cls_token + self.encoder.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        for block in self.encoder.blocks:
            x = block(x)
        latent = self.encoder.norm(x)

        tokens = latent[:, 1:, :]
        tokens = tokens.transpose(1, 2).reshape(batch_size, 384, self.grid_size, self.grid_size)
        return self.decoder(tokens, skip64, skip128)
