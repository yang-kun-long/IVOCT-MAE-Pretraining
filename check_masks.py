import numpy as np
from pathlib import Path
from PIL import Image

mask_files = list(Path('/root/CN_seg/DATA').glob('*/mask/*_mask.png'))
print(f'Total masks: {len(mask_files)}')
ratios = []
for f in sorted(mask_files):
    m = np.array(Image.open(f).convert('L'))
    ratio = (m > 127).sum() / m.size
    ratios.append(ratio)
    print(f'{f.name}: {ratio:.4f}')
print(f'Mean fg ratio: {np.mean(ratios):.4f}')
print(f'Min: {np.min(ratios):.4f}, Max: {np.max(ratios):.4f}')
