import sys, torch
import numpy as np
sys.path.insert(0, '/root/CN_seg/seven/seg')
sys.path.insert(1, '/root/CN_seg/seven')
from models import MAESegmenter
from datasets import IVOCTSegDataset
import config_seg as config
from torch.utils.data import DataLoader

model = MAESegmenter(config.MAE_CHECKPOINT, freeze_encoder=True).cuda()

# Load best checkpoint if exists
import glob
ckpts = sorted(glob.glob(str(config.SEG_CHECKPOINT_DIR / 'seg_fold0_best.pth')))
if ckpts:
    ckpt = torch.load(ckpts[-1], map_location='cpu')
    model.load_state_dict(ckpt['model'])
    print(f"Loaded checkpoint, epoch={ckpt['epoch']}, dice={ckpt['metrics']['dice_mean']:.4f}")

model.eval()
ds = IVOCTSegDataset(config.DATA_DIR, ['P001'], img_size=256, crop_ratio=0.86)
loader = DataLoader(ds, batch_size=4)

with torch.no_grad():
    batch = next(iter(loader))
    imgs = batch['image'].cuda()
    masks = batch['mask'].cuda()
    logits = model(imgs)
    probs = torch.sigmoid(logits)

print(f"Logits: min={logits.min():.3f}, max={logits.max():.3f}, mean={logits.mean():.3f}")
print(f"Probs:  min={probs.min():.3f}, max={probs.max():.3f}, mean={probs.mean():.3f}")
print(f"Mask fg ratio: {masks.mean():.4f}")
print(f"Pred>0.5 ratio: {(probs>0.5).float().mean():.4f}")
print(f"Pred>0.1 ratio: {(probs>0.1).float().mean():.4f}")
