import torch
import numpy as np


def compute_metrics(pred_logit, target, threshold=0.3):
    """
    Compute per-image segmentation metrics.

    Args:
        pred_logit: [B, 1, H, W] logits
        target: [B, 1, H, W] binary mask
        threshold: threshold for binarizing predictions

    Returns:
        dict with keys: dice, iou, sensitivity, specificity (all as lists of per-image values)
    """
    pred = torch.sigmoid(pred_logit)
    pred_binary = (pred > threshold).float()

    # Move to CPU for numpy operations
    pred_binary = pred_binary.cpu().numpy()
    target = target.cpu().numpy()

    batch_size = pred_binary.shape[0]

    dice_scores = []
    iou_scores = []
    sensitivity_scores = []
    specificity_scores = []

    for i in range(batch_size):
        pred_i = pred_binary[i, 0].flatten()
        target_i = target[i, 0].flatten()

        # Compute confusion matrix elements
        tp = np.sum((pred_i == 1) & (target_i == 1))
        fp = np.sum((pred_i == 1) & (target_i == 0))
        fn = np.sum((pred_i == 0) & (target_i == 1))
        tn = np.sum((pred_i == 0) & (target_i == 0))

        # Dice
        dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
        dice_scores.append(dice)

        # IoU
        iou = tp / (tp + fp + fn + 1e-8)
        iou_scores.append(iou)

        # Sensitivity (Recall)
        sensitivity = tp / (tp + fn + 1e-8)
        sensitivity_scores.append(sensitivity)

        # Specificity
        specificity = tn / (tn + fp + 1e-8)
        specificity_scores.append(specificity)

    return {
        "dice": dice_scores,
        "iou": iou_scores,
        "sensitivity": sensitivity_scores,
        "specificity": specificity_scores
    }


def aggregate_metrics(metrics_list):
    """
    Aggregate metrics from multiple batches.

    Args:
        metrics_list: list of dicts from compute_metrics

    Returns:
        dict with mean ± std for each metric
    """
    all_dice = []
    all_iou = []
    all_sens = []
    all_spec = []

    for metrics in metrics_list:
        all_dice.extend(metrics["dice"])
        all_iou.extend(metrics["iou"])
        all_sens.extend(metrics["sensitivity"])
        all_spec.extend(metrics["specificity"])

    return {
        "dice_mean": np.mean(all_dice),
        "dice_std": np.std(all_dice),
        "iou_mean": np.mean(all_iou),
        "iou_std": np.std(all_iou),
        "sensitivity_mean": np.mean(all_sens),
        "sensitivity_std": np.std(all_sens),
        "specificity_mean": np.mean(all_spec),
        "specificity_std": np.std(all_spec),
    }
