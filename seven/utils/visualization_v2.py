# F:\CN_seg\seven\utils\visualization_v2.py
import torch
import matplotlib.pyplot as plt


def patch_mask_to_image(mask, img_size, patch_size):
    """
    mask: [B, L]  (0=visible, 1=masked)
    return: [B,1,H,W]
    """
    B, L = mask.shape
    h = w = int(L ** 0.5)
    mask = mask.reshape(B, 1, h, w)
    mask = mask.repeat_interleave(patch_size, dim=2).repeat_interleave(patch_size, dim=3)
    return mask


def save_reconstruction_four_panel(model, imgs, pred, mask, save_path):
    """
    保存 4 联图：
    original / masked input / reconstruction / pasted reconstruction
    """
    model.eval()
    with torch.no_grad():
        recon = model.unpatchify(pred).detach().cpu()

    imgs_cpu = imgs.detach().cpu()
    mask_img = patch_mask_to_image(mask.detach().cpu(), imgs.shape[-1], model.patch_size)

    original = imgs_cpu[0, 0]
    masked_input = imgs_cpu[0, 0] * (1 - mask_img[0, 0])
    reconstruction = recon[0, 0]
    pasted = imgs_cpu[0, 0] * (1 - mask_img[0, 0]) + recon[0, 0] * mask_img[0, 0]

    plt.figure(figsize=(14, 4))

    plt.subplot(1, 4, 1)
    plt.imshow(original, cmap="gray")
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(masked_input, cmap="gray")
    plt.title("Masked Input")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(reconstruction, cmap="gray")
    plt.title("Reconstruction")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(pasted, cmap="gray")
    plt.title("Pasted Recon")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()