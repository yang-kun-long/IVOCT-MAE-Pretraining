import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(pred_logit, target, eps=1e-6):
    pred = torch.sigmoid(pred_logit)
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


def focal_loss(pred_logit, target, alpha=0.75, gamma=2.0):
    """Focal loss: down-weights easy negatives, focuses on hard positives."""
    bce = F.binary_cross_entropy_with_logits(pred_logit, target, reduction='none')
    prob = torch.sigmoid(pred_logit)
    p_t = prob * target + (1 - prob) * (1 - target)
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    focal_weight = alpha_t * (1 - p_t) ** gamma
    return (focal_weight * bce).mean()


def tversky_loss(pred_logit, target, alpha=0.3, beta=0.7, eps=1e-6):
    pred = torch.sigmoid(pred_logit)
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)

    tp = (pred_flat * target_flat).sum(dim=1)
    fp = (pred_flat * (1 - target_flat)).sum(dim=1)
    fn = ((1 - pred_flat) * target_flat).sum(dim=1)

    tversky = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return 1.0 - tversky.mean()


def focal_tversky_loss(pred_logit, target, alpha=0.3, beta=0.7, gamma=1.33, eps=1e-6):
    pred = torch.sigmoid(pred_logit)
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)

    tp = (pred_flat * target_flat).sum(dim=1)
    fp = (pred_flat * (1 - target_flat)).sum(dim=1)
    fn = ((1 - pred_flat) * target_flat).sum(dim=1)

    tversky = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return ((1.0 - tversky) ** gamma).mean()


def seg_loss(
    pred_logit,
    target,
    lambda_dice=1.0,
    lambda_bce=1.0,
    loss_mode="dice_focal",
    tversky_alpha=0.3,
    tversky_beta=0.7,
    focal_tversky_gamma=1.33,
):
    """Configurable segmentation loss."""
    if loss_mode == "dice_focal":
        return lambda_dice * dice_loss(pred_logit, target) + lambda_bce * focal_loss(pred_logit, target)
    if loss_mode == "tversky":
        return tversky_loss(pred_logit, target, alpha=tversky_alpha, beta=tversky_beta)
    if loss_mode == "focal_tversky":
        return focal_tversky_loss(
            pred_logit,
            target,
            alpha=tversky_alpha,
            beta=tversky_beta,
            gamma=focal_tversky_gamma,
        )
    raise ValueError(f"Unknown loss_mode: {loss_mode}")
