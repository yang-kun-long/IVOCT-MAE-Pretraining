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


def seg_loss(pred_logit, target, lambda_dice=1.0, lambda_bce=1.0):
    """Dice + Focal loss."""
    return lambda_dice * dice_loss(pred_logit, target) + lambda_bce * focal_loss(pred_logit, target)
