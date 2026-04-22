# F:\CN_seg\seven\utils\losses_v2.py
import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian_window(window_size=11, sigma=1.5, channels=1, device="cpu"):
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_1d = g.unsqueeze(1)
    window_2d = window_1d @ window_1d.t()
    window_2d = window_2d.unsqueeze(0).unsqueeze(0)
    window = window_2d.repeat(channels, 1, 1, 1)
    return window


def ssim_loss(x, y, fg_mask=None, window_size=11):
    """
    x, y: [B,1,H,W]
    fg_mask: [B,1,H,W] or None
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    channels = x.size(1)
    window = gaussian_window(window_size=window_size, channels=channels, device=x.device)

    mu_x = F.conv2d(x, window, padding=window_size // 2, groups=channels)
    mu_y = F.conv2d(y, window, padding=window_size // 2, groups=channels)

    mu_x2 = mu_x ** 2
    mu_y2 = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, window, padding=window_size // 2, groups=channels) - mu_x2
    sigma_y2 = F.conv2d(y * y, window, padding=window_size // 2, groups=channels) - mu_y2
    sigma_xy = F.conv2d(x * y, window, padding=window_size // 2, groups=channels) - mu_xy

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / (
        (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2) + 1e-8
    )

    if fg_mask is not None:
        weight = F.interpolate(fg_mask, size=ssim_map.shape[-2:], mode="nearest")
        loss = 1 - (ssim_map * weight).sum() / (weight.sum() + 1e-8)
    else:
        loss = 1 - ssim_map.mean()

    return loss


def gradient_map(img):
    gx = img[:, :, :, 1:] - img[:, :, :, :-1]
    gy = img[:, :, 1:, :] - img[:, :, :-1, :]

    gx = F.pad(gx, (0, 1, 0, 0))
    gy = F.pad(gy, (0, 0, 0, 1))

    return gx, gy


def gradient_loss(x, y, fg_mask=None):
    gx_x, gy_x = gradient_map(x)
    gx_y, gy_y = gradient_map(y)

    loss_map = (gx_x - gx_y).abs() + (gy_x - gy_y).abs()

    if fg_mask is not None:
        weight = fg_mask.expand_as(loss_map)
        loss = (loss_map * weight).sum() / (weight.sum() + 1e-8)
    else:
        loss = loss_map.mean()

    return loss