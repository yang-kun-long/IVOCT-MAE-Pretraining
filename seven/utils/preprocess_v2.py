# F:\CN_seg\seven\utils\preprocess_v2.py
import numpy as np
from PIL import Image


def crop_center_square(img: Image.Image, crop_ratio=0.86) -> Image.Image:
    """
    从图像中心裁成正方形，并按 crop_ratio 缩小范围，
    以去掉大面积无效黑背景和边缘文字/比例尺干扰。
    """
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


def build_foreground_mask(img_tensor, threshold=0.05):
    """
    img_tensor: [1, H, W], 值域 [0,1]
    返回前景 mask: [1, H, W]
    目的：区分有效成像区域和大面积黑背景
    """
    fg = (img_tensor > threshold).float()
    return fg