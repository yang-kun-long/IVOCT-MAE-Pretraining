"""V13 segmentation model: v7 skip decoder + a frame-level CA classification head.

The classification head reads the encoder CLS token after self-attention,
so the entire encoder is shared between detection and segmentation.
"""

import torch
import torch.nn as nn

from .seg_model_skip import MAESkipSegmenter


class MAESkipSegmenterV13(MAESkipSegmenter):
    """MAE skip segmenter + binary calcification classifier head.

    forward(x) returns a dict:
        {"seg": [B, 1, 256, 256] logits, "cls": [B] logits}
    """

    def __init__(
        self,
        mae_checkpoint_path,
        patch_size=8,
        freeze_encoder=False,
        use_adapter=False,
        adapter_bottleneck=64,
        decoder_norm="batch",
        cls_hidden=256,
        cls_dropout=0.1,
    ):
        super().__init__(
            mae_checkpoint_path=mae_checkpoint_path,
            patch_size=patch_size,
            freeze_encoder=freeze_encoder,
            use_adapter=use_adapter,
            adapter_bottleneck=adapter_bottleneck,
            decoder_norm=decoder_norm,
        )

        embed_dim = self.encoder.cls_token.shape[-1]
        self.cls_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, cls_hidden),
            nn.GELU(),
            nn.Dropout(cls_dropout),
            nn.Linear(cls_hidden, 1),
        )

        for module in self.cls_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        with torch.no_grad():
            self.cls_head[-1].bias.fill_(-1.0)

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

        cls_feat = latent[:, 0, :]
        cls_logits = self.cls_head(cls_feat).squeeze(-1)

        tokens = latent[:, 1:, :]
        tokens = tokens.transpose(1, 2).reshape(
            batch_size, 384, self.grid_size, self.grid_size
        )
        seg_logits = self.decoder(tokens, skip64, skip128)

        return {"seg": seg_logits, "cls": cls_logits}

    @torch.no_grad()
    def predict_gated(self, x, cls_threshold=0.5, seg_threshold=0.5):
        """Inference convenience: returns hard mask with detection gating."""
        out = self.forward(x)
        cls_prob = torch.sigmoid(out["cls"])
        seg_prob = torch.sigmoid(out["seg"])
        gate = (cls_prob >= cls_threshold).float().view(-1, 1, 1, 1)
        pred = (seg_prob >= seg_threshold).float() * gate
        return {
            "cls_prob": cls_prob,
            "seg_prob": seg_prob,
            "pred_mask": pred,
        }
