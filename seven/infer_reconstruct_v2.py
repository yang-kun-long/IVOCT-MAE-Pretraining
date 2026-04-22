# F:\CN_seg\seven\infer_reconstruct_v2.py
import torch
from torch.utils.data import DataLoader

import config_v2 as config
from datasets.ivoct_pretrain_dataset_v2 import IVOCTPretrainDatasetV2
from models.mae_hybrid_v2 import hybrid_mae_vit_small_patch8
from utils.visualization_v2 import save_reconstruction_four_panel


def main():
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

    dataset = IVOCTPretrainDatasetV2(
        root_dir=config.DATA_DIR,
        img_size=config.IMG_SIZE,
        crop_ratio=config.ROI_CROP_RATIO,
        fg_threshold=config.FOREGROUND_THRESHOLD,
        image_exts=config.IMAGE_EXTS
    )

    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)

    model = hybrid_mae_vit_small_patch8(
        img_size=config.IMG_SIZE,
        in_chans=config.IN_CHANS,
        norm_pix_loss=config.NORM_PIX_LOSS,
        fg_mask_bias=config.FG_MASK_BIAS,
        mask_mode=config.MASK_MODE,
    ).to(device)

    ckpt = torch.load(config.CHECKPOINT_DIR / "mae_v2_best.pth", map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    batch = next(iter(loader))
    imgs = batch["image"].to(device)
    fg_mask = batch["fg_mask"].to(device)

    with torch.no_grad():
        out = model(imgs, fg_mask, mask_ratio=config.MASK_RATIO)

    save_path = config.RECON_DIR / "reconstruction_check_v2.png"
    save_reconstruction_four_panel(model, imgs, out["pred"], out["mask"], save_path)
    print(f"Saved to: {save_path}")


if __name__ == "__main__":
    main()