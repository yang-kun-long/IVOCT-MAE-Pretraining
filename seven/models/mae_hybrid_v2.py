# F:\CN_seg\seven\models\mae_hybrid_v2.py
import math
import torch
import torch.nn as nn
import numpy as np
from timm.models.vision_transformer import Block


class Adapter(nn.Module):
    """
    Adapter module for parameter-efficient fine-tuning.
    Inserted after each transformer block's MLP.

    Args:
        dim: Input/output dimension
        bottleneck: Bottleneck dimension (compression ratio = dim/bottleneck)
        dropout: Dropout rate
        init_scale: Initial scale for residual connection
    """
    def __init__(self, dim=384, bottleneck=64, dropout=0.1, init_scale=1e-3):
        super().__init__()
        self.down_proj = nn.Linear(dim, bottleneck)
        self.activation = nn.GELU()
        self.up_proj = nn.Linear(bottleneck, dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = nn.Parameter(torch.ones(1) * init_scale)

        # Initialize with small weights for stable training
        nn.init.xavier_uniform_(self.down_proj.weight, gain=init_scale)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.xavier_uniform_(self.up_proj.weight, gain=init_scale)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x):
        residual = x
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.up_proj(x)
        x = self.dropout(x)
        return residual + self.scale * x


class BlockWithAdapter(nn.Module):
    """
    Transformer Block with optional Adapter module.
    Wraps timm's Block and adds adapter after MLP.
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True,
                 norm_layer=nn.LayerNorm, use_adapter=False, adapter_bottleneck=64):
        super().__init__()
        self.block = Block(dim, num_heads, mlp_ratio, qkv_bias=qkv_bias, norm_layer=norm_layer)
        self.use_adapter = use_adapter
        if use_adapter:
            self.adapter = Adapter(dim=dim, bottleneck=adapter_bottleneck)

    def forward(self, x):
        x = self.block(x)
        if self.use_adapter:
            x = self.adapter(x)
        return x


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / (10000 ** omega)

    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


class ConvStemPatchEmbed(nn.Module):
    """
    Conv Stem with flexible patch size support.

    For patch_size=8:  256 -> 128 -> 64 -> 32 (3 stages, stride=2 each)
    For patch_size=16: 256 -> 128 -> 64 -> 32 -> 16 (4 stages, stride=2 each)
    """
    def __init__(self, img_size=256, patch_size=8, in_chans=1, embed_dim=384):
        super().__init__()

        if patch_size == 8:
            # 3-stage: 256 -> 128 -> 64 -> 32
            self.stem = nn.Sequential(
                nn.Conv2d(in_chans, 32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.GELU(),

                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.GELU(),

                nn.Conv2d(64, embed_dim, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(embed_dim),
                nn.GELU(),
            )
        elif patch_size == 16:
            # 4-stage: 256 -> 128 -> 64 -> 32 -> 16
            self.stem = nn.Sequential(
                nn.Conv2d(in_chans, 32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.GELU(),

                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.GELU(),

                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.GELU(),

                nn.Conv2d(128, embed_dim, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(embed_dim),
                nn.GELU(),
            )
        else:
            raise ValueError(f"Unsupported patch_size: {patch_size}. Only 8 and 16 are supported.")

        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.embed_dim = embed_dim

    def forward(self, x):
        x = self.stem(x)  # [B, embed_dim, grid_size, grid_size]
        x = x.flatten(2).transpose(1, 2)  # [B, N, C]
        return x


class HybridMAEViT(nn.Module):
    def __init__(
        self,
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
        norm_layer=nn.LayerNorm,
        norm_pix_loss=True,
        fg_mask_bias=0.6,
        mask_mode="foreground_aware",
        use_adapter=False,
        adapter_bottleneck=64,
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.norm_pix_loss = norm_pix_loss
        self.fg_mask_bias = fg_mask_bias
        self.mask_mode = mask_mode
        self.use_adapter = use_adapter

        # encoder
        self.patch_embed = ConvStemPatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches
        grid_size = self.patch_embed.grid_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        )

        # Use BlockWithAdapter if adapter is enabled
        if use_adapter:
            self.blocks = nn.ModuleList([
                BlockWithAdapter(embed_dim, num_heads, mlp_ratio, qkv_bias=True,
                               norm_layer=norm_layer, use_adapter=True,
                               adapter_bottleneck=adapter_bottleneck)
                for _ in range(depth)
            ])
        else:
            self.blocks = nn.ModuleList([
                Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                for _ in range(depth)
            ])
        self.norm = norm_layer(embed_dim)

        # decoder
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False
        )
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size * patch_size * in_chans, bias=True)

        self.initialize_weights(grid_size)

    def freeze_encoder(self, freeze_mode="full"):
        """
        Freeze encoder parameters for adapter tuning.

        Args:
            freeze_mode:
                - "full": Freeze all encoder parameters (patch_embed + blocks + norm)
                - "adapter_only": Freeze encoder but keep adapters trainable
                - "none": Don't freeze anything
        """
        if freeze_mode == "none":
            return

        # Freeze patch embedding
        for param in self.patch_embed.parameters():
            param.requires_grad = False

        # Freeze positional embeddings and cls token
        self.pos_embed.requires_grad = False
        self.cls_token.requires_grad = False

        # Freeze encoder blocks
        for block in self.blocks:
            if self.use_adapter and freeze_mode == "adapter_only":
                # Freeze transformer block but keep adapter trainable
                if hasattr(block, 'block'):
                    for param in block.block.parameters():
                        param.requires_grad = False
                    # Adapter remains trainable
                else:
                    for param in block.parameters():
                        param.requires_grad = False
            else:
                # Freeze everything including adapters
                for param in block.parameters():
                    param.requires_grad = False

        # Freeze encoder norm
        for param in self.norm.parameters():
            param.requires_grad = False

        print(f"Encoder frozen with mode: {freeze_mode}")

    def get_trainable_params(self):
        """Return statistics about trainable parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        encoder_params = sum(p.numel() for p in self.patch_embed.parameters()) + \
                        sum(p.numel() for block in self.blocks for p in block.parameters()) + \
                        sum(p.numel() for p in self.norm.parameters())

        decoder_params = sum(p.numel() for p in self.decoder_embed.parameters()) + \
                        sum(p.numel() for block in self.decoder_blocks for p in block.parameters()) + \
                        sum(p.numel() for p in self.decoder_norm.parameters()) + \
                        sum(p.numel() for p in self.decoder_pred.parameters())

        adapter_params = 0
        if self.use_adapter:
            for block in self.blocks:
                if hasattr(block, 'adapter'):
                    adapter_params += sum(p.numel() for p in block.adapter.parameters())

        return {
            "total": total_params,
            "trainable": trainable_params,
            "encoder": encoder_params,
            "decoder": decoder_params,
            "adapter": adapter_params,
            "trainable_ratio": trainable_params / total_params * 100
        }

    def initialize_weights(self, grid_size):
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], grid_size, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], grid_size, cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def patchify(self, imgs):
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], self.in_chans, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * self.in_chans))
        return x

    def unpatchify(self, x):
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.in_chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self.in_chans, h * p, h * p))
        return imgs

    def patchify_mask(self, fg_mask):
        """
        fg_mask: [B,1,H,W]
        返回每个 patch 的前景占比 [B, L]
        """
        p = self.patch_size
        h = w = fg_mask.shape[2] // p
        x = fg_mask.reshape(fg_mask.shape[0], 1, h, p, w, p)
        x = x.mean(dim=(3, 5))    # [B,1,h,w]
        x = x.flatten(2).squeeze(1)
        return x

    def random_masking(self, x, mask_ratio, patch_fg=None):
        """
        foreground-aware masking:
        有效前景区域的 noise 增大，更容易被 mask，
        避免把太多监督浪费在纯黑背景 patch 上。
        """
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)

        if self.mask_mode == "foreground_aware" and patch_fg is not None:
            patch_fg = patch_fg / (patch_fg.max(dim=1, keepdim=True)[0] + 1e-8)
            noise = noise + self.fg_mask_bias * patch_fg

        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D)
        )

        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, imgs, fg_mask, mask_ratio):
        x = self.patch_embed(imgs)
        x = x + self.pos_embed[:, 1:, :]

        patch_fg = self.patchify_mask(fg_mask)

        x, mask, ids_restore = self.random_masking(
            x, mask_ratio=mask_ratio, patch_fg=patch_fg
        )

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)

        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
        )
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(
            x_, dim=1,
            index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )

        x = torch.cat([x[:, :1, :], x_], dim=1)
        x = x + self.decoder_pos_embed

        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        x = self.decoder_pred(x)
        x = x[:, 1:, :]
        return x

    def forward_loss_mse(self, imgs, pred, mask):
        target = self.patchify(imgs)

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6).sqrt()

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / (mask.sum() + 1e-8)
        return loss

    def forward(self, imgs, fg_mask, mask_ratio=0.70):
        latent, mask, ids_restore = self.forward_encoder(imgs, fg_mask, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss_mse = self.forward_loss_mse(imgs, pred, mask)

        return {
            "loss_mse": loss_mse,
            "pred": pred,
            "mask": mask
        }


def hybrid_mae_vit_small_patch8(
    img_size=256,
    in_chans=1,
    norm_pix_loss=True,
    fg_mask_bias=0.6,
    mask_mode="foreground_aware",
    use_adapter=False,
    adapter_bottleneck=64,
):
    model = HybridMAEViT(
        img_size=img_size,
        patch_size=8,
        in_chans=in_chans,
        embed_dim=384,
        depth=12,
        num_heads=6,
        decoder_embed_dim=256,
        decoder_depth=4,
        decoder_num_heads=8,
        mlp_ratio=4.0,
        norm_pix_loss=norm_pix_loss,
        fg_mask_bias=fg_mask_bias,
        mask_mode=mask_mode,
        use_adapter=use_adapter,
        adapter_bottleneck=adapter_bottleneck,
    )
    return model