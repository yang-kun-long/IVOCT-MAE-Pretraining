"""
评估脚本：遮挡重建评估
评估模型在遮挡部分区域后的重建能力（MAE 的真实任务）
"""
import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from torch.utils.data import DataLoader

import config_v2 as config
from datasets.ivoct_pretrain_dataset_v2 import IVOCTPretrainDatasetV2
from models.mae_hybrid_v2 import hybrid_mae_vit_small_patch8


def evaluate_masked():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"评估模式: 遮挡重建（Masked Reconstruction）")
    print(f"Mask ratio: {config.MASK_RATIO}")

    dataset = IVOCTPretrainDatasetV2(
        root_dir=config.DATA_DIR,
        img_size=config.IMG_SIZE,
        crop_ratio=config.ROI_CROP_RATIO,
        fg_threshold=config.FOREGROUND_THRESHOLD,
        image_exts=config.IMAGE_EXTS
    )

    print(f"Total images: {len(dataset)}")

    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=2)

    model = hybrid_mae_vit_small_patch8(
        img_size=config.IMG_SIZE,
        in_chans=config.IN_CHANS,
        norm_pix_loss=config.NORM_PIX_LOSS,
        fg_mask_bias=config.FG_MASK_BIAS,
        mask_mode=config.MASK_MODE,
    ).to(device)

    ckpt = torch.load(config.CHECKPOINT_DIR / "mae_v2_best.pth", map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print("Evaluating masked reconstruction quality...")

    ssim_scores = []
    psnr_scores = []
    mse_scores = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            imgs = batch["image"].to(device)
            fg_mask = batch["fg_mask"].to(device)

            # 使用配置的 mask ratio 进行遮挡
            out = model(imgs, fg_mask, mask_ratio=config.MASK_RATIO)
            recon = model.unpatchify(out["pred"])

            imgs_np = imgs.cpu().numpy()
            recon_np = recon.cpu().numpy()

            for i in range(imgs.shape[0]):
                img = imgs_np[i, 0]
                rec = recon_np[i, 0]

                ssim_val = ssim(img, rec, data_range=1.0)
                ssim_scores.append(ssim_val)

                psnr_val = psnr(img, rec, data_range=1.0)
                psnr_scores.append(psnr_val)

                mse_val = np.mean((img - rec) ** 2)
                mse_scores.append(mse_val)

            if (batch_idx + 1) % 20 == 0:
                print(f"Processed {(batch_idx + 1) * 8} images...")

    print("\n" + "=" * 70)
    print("遮挡重建质量评估 (Masked Reconstruction Quality)")
    print("=" * 70)
    print(f"SSIM: {np.mean(ssim_scores):.4f} ± {np.std(ssim_scores):.4f}")
    print(f"PSNR: {np.mean(psnr_scores):.2f} ± {np.std(psnr_scores):.2f} dB")
    print(f"MSE:  {np.mean(mse_scores):.6f} ± {np.std(mse_scores):.6f}")
    print("=" * 70)

    avg_ssim = np.mean(ssim_scores)
    avg_psnr = np.mean(psnr_scores)

    print("\n质量评级:")
    if avg_ssim > 0.85:
        ssim_grade = "优秀 (Excellent)"
    elif avg_ssim > 0.70:
        ssim_grade = "良好 (Good)"
    elif avg_ssim > 0.50:
        ssim_grade = "一般 (Fair)"
    else:
        ssim_grade = "较差 (Poor)"

    if avg_psnr > 30:
        psnr_grade = "优秀 (Excellent)"
    elif avg_psnr > 25:
        psnr_grade = "良好 (Good)"
    elif avg_psnr > 20:
        psnr_grade = "一般 (Fair)"
    else:
        psnr_grade = "较差 (Poor)"

    print(f"  SSIM: {ssim_grade}")
    print(f"  PSNR: {psnr_grade}")

    print("\n说明:")
    print(f"  此评估测试模型在遮挡 {int(config.MASK_RATIO*100)}% 区域后的重建能力")
    print("  这是 MAE 预训练的真实任务场景")
    print("=" * 70)

    results = {
        "mode": "masked_reconstruction",
        "mask_ratio": config.MASK_RATIO,
        "ssim_mean": float(np.mean(ssim_scores)),
        "ssim_std": float(np.std(ssim_scores)),
        "psnr_mean": float(np.mean(psnr_scores)),
        "psnr_std": float(np.std(psnr_scores)),
        "mse_mean": float(np.mean(mse_scores)),
        "mse_std": float(np.std(mse_scores)),
        "ssim_grade": ssim_grade,
        "psnr_grade": psnr_grade,
    }

    import json
    with open(config.LOG_DIR / "evaluation_masked.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n结果已保存到: {config.LOG_DIR / 'evaluation_masked.json'}")


if __name__ == "__main__":
    evaluate_masked()
