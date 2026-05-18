"""V13 dataset: includes negative (no-mask) frames for joint detection + segmentation.

For positive frames (have CA mask): mask=GT mask, has_calc=1.
For negative frames (no mask): mask=zeros, has_calc=0.

Training mode subsamples negatives to keep a manageable neg:pos ratio.
Evaluation mode includes ALL frames so the detection gate is tested honestly.
"""

import random
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

from .ivoct_seg_dataset import crop_center_square


class IVOCTSegDatasetV13(Dataset):
    def __init__(
        self,
        data_dir,
        patient_ids,
        img_size=256,
        crop_ratio=0.86,
        is_train=False,
        include_negatives=True,
        neg_pos_ratio=2.0,
        neg_subsample_seed=42,
    ):
        """
        Args:
            include_negatives: if False, behave like the original IVOCTSegDataset
                (positives only). Useful for legacy comparisons.
            neg_pos_ratio: training-time ratio of negative:positive samples.
                Ignored when is_train=False (val/test keeps all negatives).
            neg_subsample_seed: deterministic seed for negative subsampling so
                folds are reproducible across runs.
        """
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.crop_ratio = crop_ratio
        self.is_train = is_train
        self.include_negatives = include_negatives
        self.neg_pos_ratio = neg_pos_ratio

        positives = []
        negatives = []

        for patient_id in patient_ids:
            data_dir_p = self.data_dir / patient_id / "Data"
            mask_dir_p = self.data_dir / patient_id / "mask"
            if not data_dir_p.exists():
                continue

            mask_basenames = {
                p.stem.replace("_mask", "")
                for p in mask_dir_p.glob("*_mask.png")
            } if mask_dir_p.exists() else set()

            for img_path in sorted(data_dir_p.glob("*.jpg")):
                stem = img_path.stem
                if stem in mask_basenames:
                    positives.append({
                        "image": img_path,
                        "mask": mask_dir_p / f"{stem}_mask.png",
                        "patient": patient_id,
                        "has_calc": 1,
                    })
                else:
                    negatives.append({
                        "image": img_path,
                        "mask": None,
                        "patient": patient_id,
                        "has_calc": 0,
                    })

        if include_negatives and is_train and len(positives) > 0:
            target_neg = int(round(len(positives) * neg_pos_ratio))
            if len(negatives) > target_neg:
                rng = random.Random(neg_subsample_seed)
                negatives = rng.sample(negatives, target_neg)

        if not include_negatives:
            negatives = []

        self.samples = positives + negatives
        self.num_positive = len(positives)
        self.num_negative = len(negatives)

        if is_train:
            random.Random(neg_subsample_seed + 1).shuffle(self.samples)

        print(
            f"[IVOCTSegDatasetV13] is_train={is_train} include_neg={include_negatives} "
            f"patients={patient_ids} pos={self.num_positive} neg={self.num_negative} "
            f"total={len(self.samples)}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        img = Image.open(sample["image"]).convert("L")
        img = crop_center_square(img, self.crop_ratio)
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        img = TF.to_tensor(img)

        if sample["mask"] is not None:
            mask = Image.open(sample["mask"]).convert("L")
            mask = crop_center_square(mask, self.crop_ratio)
            mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)
            mask = TF.to_tensor(mask)
            mask = (mask > 0.5).float()
        else:
            mask = torch.zeros(1, self.img_size, self.img_size, dtype=torch.float32)

        if self.is_train:
            if random.random() > 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)
            if random.random() > 0.5:
                img = TF.vflip(img)
                mask = TF.vflip(mask)

            angle = random.uniform(-180, 180)
            img = TF.rotate(img, angle, interpolation=TF.InterpolationMode.BILINEAR)
            mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)

            brightness_factor = random.uniform(0.9, 1.1)
            img = torch.clamp(TF.adjust_brightness(img, brightness_factor), 0, 1)
            contrast_factor = random.uniform(0.85, 1.15)
            img = torch.clamp(TF.adjust_contrast(img, contrast_factor), 0, 1)

            noise = torch.randn_like(img) * 0.005
            img = torch.clamp(img + noise, 0, 1)

            dx = random.uniform(-0.03, 0.03)
            dy = random.uniform(-0.03, 0.03)
            scale = random.uniform(0.97, 1.03)
            img = TF.affine(
                img, angle=0, translate=[dx, dy], scale=scale, shear=0,
                interpolation=TF.InterpolationMode.BILINEAR,
            )
            mask = TF.affine(
                mask, angle=0, translate=[dx, dy], scale=scale, shear=0,
                interpolation=TF.InterpolationMode.NEAREST,
            )

        return {
            "image": img,
            "mask": mask,
            "has_calc": torch.tensor(sample["has_calc"], dtype=torch.float32),
            "path": str(sample["image"]),
            "patient": sample["patient"],
        }

    def get_patient_id(self, idx):
        return self.samples[idx]["patient"]
