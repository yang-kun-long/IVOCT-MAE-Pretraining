"""MAE skip segmenter initialized from a lipid-supervised pretrained encoder.

Wraps MAESkipSegmenter: builds the architecture from the standard MAE ckpt
(for ConvStem / pos embeds / cls token shapes), then overrides encoder weights
with those from a lipid-supervised pretrain artifact saved by
`train_mae_lipid_supervised_v1.py` (keys prefixed with "encoder.").

This is the cleanest single-variable change versus the v7 baseline: same
decoder, same architecture, same data — only the encoder initialization
differs.
"""

import torch

from .seg_model_skip import MAESkipSegmenter


class MAELipidWarmSkipSegmenter(MAESkipSegmenter):
    def __init__(
        self,
        mae_checkpoint_path,
        lipid_warm_encoder_path,
        patch_size=8,
        freeze_encoder=False,
        use_adapter=False,
        adapter_bottleneck=64,
        decoder_norm="batch",
    ):
        super().__init__(
            mae_checkpoint_path=mae_checkpoint_path,
            patch_size=patch_size,
            freeze_encoder=freeze_encoder,
            use_adapter=use_adapter,
            adapter_bottleneck=adapter_bottleneck,
            decoder_norm=decoder_norm,
        )
        self._load_lipid_warm_encoder(lipid_warm_encoder_path)

    def _load_lipid_warm_encoder(self, path):
        print(f"Overriding encoder weights from lipid-warm checkpoint: {path}")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if "encoder" not in ckpt:
            raise KeyError(
                f"Lipid-warm checkpoint at {path} must contain an 'encoder' "
                "state dict (saved by train_mae_lipid_supervised_v1.py)."
            )
        raw = ckpt["encoder"]
        stripped = {
            (k[len("encoder."):] if k.startswith("encoder.") else k): v
            for k, v in raw.items()
        }
        missing, unexpected = self.encoder.load_state_dict(stripped, strict=False)
        print(
            f"  lipid-warm encoder loaded: {len(stripped)} tensors, "
            f"{len(missing)} missing, {len(unexpected)} unexpected, "
            f"source_epoch={ckpt.get('epoch')} val_dice={ckpt.get('val_metrics', {}).get('dice_mean')}"
        )
