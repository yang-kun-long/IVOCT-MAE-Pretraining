import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def save_seg_visualization(images, preds, targets, save_path, num_samples=4):
    """
    Save 3-column visualization: image | prediction | ground truth.

    Args:
        images: [B, 1, H, W] input images
        preds: [B, 1, H, W] prediction logits
        targets: [B, 1, H, W] ground truth masks
        save_path: path to save figure
        num_samples: number of samples to visualize
    """
    images = images.cpu().numpy()
    preds = torch.sigmoid(preds).cpu().numpy()
    targets = targets.cpu().numpy()

    batch_size = min(images.shape[0], num_samples)

    fig, axes = plt.subplots(batch_size, 3, figsize=(12, 4 * batch_size))
    if batch_size == 1:
        axes = axes[np.newaxis, :]

    for i in range(batch_size):
        img = images[i, 0]
        pred = preds[i, 0]
        target = targets[i, 0]

        # Image
        axes[i, 0].imshow(img, cmap='gray')
        axes[i, 0].set_title('Input Image')
        axes[i, 0].axis('off')

        # Prediction
        axes[i, 1].imshow(pred, cmap='jet', vmin=0, vmax=1)
        axes[i, 1].set_title('Prediction')
        axes[i, 1].axis('off')

        # Ground Truth
        axes[i, 2].imshow(target, cmap='gray', vmin=0, vmax=1)
        axes[i, 2].set_title('Ground Truth')
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
