# F:\CN_seg\seven\datasets\ivoct_pretrain_dataset_v2.py
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from utils.preprocess_v2 import crop_center_square, build_foreground_mask


class IVOCTPretrainDatasetV2(Dataset):
    def __init__(
        self,
        root_dir,
        img_size=256,
        crop_ratio=0.86,
        fg_threshold=0.05,
        image_exts=(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    ):
        self.root_dir = Path(root_dir)
        self.img_size = img_size
        self.crop_ratio = crop_ratio
        self.fg_threshold = fg_threshold
        self.image_exts = image_exts

        self.image_paths = []
        for patient_dir in sorted(self.root_dir.iterdir()):
            if not patient_dir.is_dir():
                continue
            data_dir = patient_dir / "Data"
            if not data_dir.exists():
                continue
            for ext in image_exts:
                self.image_paths.extend(sorted(data_dir.glob(f"*{ext}")))

        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

        print(f"[DatasetV2] Total IVOCT images for pretraining: {len(self.image_paths)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("L")

        # 先做中心 ROI 裁剪
        img = crop_center_square(img, crop_ratio=self.crop_ratio)

        # 再 resize + tensor
        img = self.transform(img)  # [1,H,W], [0,1]

        # 前景 mask
        fg_mask = build_foreground_mask(img, threshold=self.fg_threshold)

        return {
            "image": img,
            "fg_mask": fg_mask,
            "path": str(img_path)
        }