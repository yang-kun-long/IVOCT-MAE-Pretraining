from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
import random


def crop_center_square(img: Image.Image, crop_ratio=0.86) -> Image.Image:
    """与预训练一致的中心裁剪"""
    w, h = img.size
    side = min(w, h)
    crop_side = int(side * crop_ratio)

    cx, cy = w // 2, h // 2
    half = crop_side // 2

    left = max(0, cx - half)
    upper = max(0, cy - half)
    right = min(w, cx + half)
    lower = min(h, cy + half)

    return img.crop((left, upper, right, lower))


class IVOCTSegDataset(Dataset):
    def __init__(
        self,
        data_dir,
        patient_ids,
        img_size=256,
        crop_ratio=0.86,
        is_train=False
    ):
        """
        Args:
            data_dir: 根目录 (DATA/)
            patient_ids: 患者列表，如 ["P001", "P002"]
            img_size: resize 后尺寸
            crop_ratio: 中心裁剪比例
            is_train: 是否训练集（决定是否做数据增强）
        """
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.crop_ratio = crop_ratio
        self.is_train = is_train

        self.samples = []
        for patient_id in patient_ids:
            mask_dir = self.data_dir / patient_id / "mask"
            if not mask_dir.exists():
                continue

            for mask_path in sorted(mask_dir.glob("*_mask.png")):
                # 派生对应的图像路径：去掉 _mask 后缀
                img_name = mask_path.stem.replace("_mask", "") + ".jpg"
                img_path = self.data_dir / patient_id / "Data" / img_name

                if img_path.exists():
                    self.samples.append({
                        "image": img_path,
                        "mask": mask_path,
                        "patient": patient_id
                    })

        print(f"[IVOCTSegDataset] Loaded {len(self.samples)} samples from {patient_ids}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 加载图像和 mask
        img = Image.open(sample["image"]).convert("L")
        mask = Image.open(sample["mask"]).convert("L")

        # 中心裁剪
        img = crop_center_square(img, self.crop_ratio)
        mask = crop_center_square(mask, self.crop_ratio)

        # Resize
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)

        # 转 tensor
        img = TF.to_tensor(img)    # [1, H, W], [0, 1]
        mask = TF.to_tensor(mask)  # [1, H, W], [0, 1]

        # mask 二值化
        mask = (mask > 0.5).float()

        # 数据增强（训练集）
        if self.is_train:
            # 随机水平翻转
            if random.random() > 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)

            # 随机垂直翻转
            if random.random() > 0.5:
                img = TF.vflip(img)
                mask = TF.vflip(mask)

            # 随机旋转 ±180°（IVOCT 是圆形，任意角度都合理）
            angle = random.uniform(-180, 180)
            img = TF.rotate(img, angle, interpolation=TF.InterpolationMode.BILINEAR)
            mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)

            # 亮度 jitter
            brightness_factor = random.uniform(0.9, 1.1)
            img = TF.adjust_brightness(img, brightness_factor)
            img = torch.clamp(img, 0, 1)

            # 对比度 jitter
            contrast_factor = random.uniform(0.85, 1.15)
            img = TF.adjust_contrast(img, contrast_factor)
            img = torch.clamp(img, 0, 1)

            # 高斯噪声（模拟 IVOCT 散斑噪声）
            noise = torch.randn_like(img) * 0.005
            img = img + noise
            img = torch.clamp(img, 0, 1)

            # 随机平移 + 缩放（模拟导管偏移、血管尺寸变化）
            dx = random.uniform(-0.03, 0.03)
            dy = random.uniform(-0.03, 0.03)
            scale = random.uniform(0.97, 1.03)
            img = TF.affine(img, angle=0, translate=[dx, dy], scale=scale, shear=0,
                            interpolation=TF.InterpolationMode.BILINEAR)
            mask = TF.affine(mask, angle=0, translate=[dx, dy], scale=scale, shear=0,
                             interpolation=TF.InterpolationMode.NEAREST)

        return {
            "image": img,
            "mask": mask,
            "path": str(sample["image"])
        }

    def get_patient_id(self, idx):
        """获取指定索引样本的患者ID"""
        return self.samples[idx]["patient"]
