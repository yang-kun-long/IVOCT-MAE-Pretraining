import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(pred_logit, target, eps=1e-6):
    """
    Dice loss for binary segmentation.

    Args:
        pred_logit: [B, 1, H, W] logits (before sigmoid)
        target: [B, 1, H, W] binary mask (0 or 1)
        eps: smoothing term

    Returns:
        scalar loss
    """
    pred = torch.sigmoid(pred_logit)

    # Flatten spatial dimensions
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)

    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)

    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


def seg_loss(pred_logit, target, lambda_dice=1.0, lambda_bce=1.0):
    """
    Combined Dice + BCE loss.

    Args:
        pred_logit: [B, 1, H, W] logits
        target: [B, 1, H, W] binary mask
        lambda_dice: weight for Dice loss
        lambda_bce: weight for BCE loss

    Returns:
        scalar loss
    """
    bce = F.binary_cross_entropy_with_logits(pred_logit, target)
    dice = dice_loss(pred_logit, target)

    return lambda_dice * dice + lambda_bce * bce
