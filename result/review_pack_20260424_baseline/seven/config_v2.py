# F:\CN_seg\seven\config_v2.py
from pathlib import Path

# =========================
# 路径
# =========================
ROOT_DIR = Path(__file__).parent.parent  # 自动获取项目根目录
DATA_DIR = ROOT_DIR / "DATA"
SEVEN_DIR = ROOT_DIR / "seven"

CHECKPOINT_DIR = SEVEN_DIR / "checkpoints_v2"
LOG_DIR = SEVEN_DIR / "logs_v2"
RECON_DIR = SEVEN_DIR / "recon_vis_v2"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
RECON_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# 数据
# =========================
IMG_SIZE = 256
IN_CHANS = 1
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# 预处理参数
ROI_CROP_RATIO = 0.86       # 中心裁剪比例，去掉外层无效黑背景与文字区域
FOREGROUND_THRESHOLD = 0.05 # 前景阈值（归一化后）

# =========================
# 模型
# =========================
PATCH_SIZE = 16          # 改进：8 → 16，减少棋盘格伪影
EMBED_DIM = 384
DEPTH = 12
NUM_HEADS = 6
MLP_RATIO = 4.0

DECODER_EMBED_DIM = 384  # 改进：256 → 384，增加重建容量
DECODER_DEPTH = 8        # 改进：4 → 8，增强重建能力
DECODER_NUM_HEADS = 8

MASK_RATIO = 0.50        # v4 改进：0.60 → 0.50，进一步降低重建难度
NORM_PIX_LOSS = True

# foreground-aware masking
FG_MASK_BIAS = 0.6
MASK_MODE = "foreground_aware"   # "random" or "foreground_aware"

# =========================
# 损失
# =========================
LAMBDA_MSE = 1.0
LAMBDA_SSIM = 0.7        # v4 改进：0.5 → 0.7，更重视结构相似性
LAMBDA_GRAD = 0.2        # 改进：0.10 → 0.2，更重视边缘保持

# =========================
# 训练
# =========================
BATCH_SIZE = 32
NUM_WORKERS = 4
EPOCHS = 400             # v4 改进：300 → 400，更充分训练
WARMUP_EPOCHS = 10

BASE_LR = 1.5e-4
WEIGHT_DECAY = 0.05
BETAS = (0.9, 0.95)

DEVICE = "cuda"
SEED = 42
SAVE_FREQ = 10
VIS_FREQ = 10
USE_AMP = True