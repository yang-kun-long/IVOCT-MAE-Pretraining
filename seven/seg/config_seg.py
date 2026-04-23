from pathlib import Path

# =========================
# 路径
# =========================
ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "DATA"
SEVEN_DIR = ROOT_DIR / "seven"

MAE_CHECKPOINT = ROOT_DIR / "results" / "pretrain" / "v4" / "mae_v2_best.pth"
SEG_CHECKPOINT_DIR = SEVEN_DIR / "seg" / "checkpoints"
SEG_LOG_DIR = SEVEN_DIR / "seg" / "logs"
SEG_VIS_DIR = SEVEN_DIR / "seg" / "vis"

SEG_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
SEG_LOG_DIR.mkdir(parents=True, exist_ok=True)
SEG_VIS_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# 数据
# =========================
PATIENTS = ["P001", "P002", "P003", "P004"]
SPLIT_MODE = "lopo"          # "lopo" | "single_holdout"
HOLDOUT_PATIENT = "P004"     # 仅 single_holdout 时使用

IMG_SIZE = 256
ROI_CROP_RATIO = 0.86        # 与预训练一致
IN_CHANS = 1

# =========================
# 模型（必须与 checkpoint 匹配）
# =========================
# 注意：checkpoint 真实架构是 patch=8，不是 config_v2.py 里的 16
PATCH_SIZE = 8
EMBED_DIM = 384
DEPTH = 12
NUM_HEADS = 6
FREEZE_ENCODER = False       # 全量 fine-tune

# =========================
# 训练
# =========================
EPOCHS = 100
BATCH_SIZE = 8
BASE_LR = 1e-4               # decoder lr
ENCODER_LR_SCALE = 0.1       # encoder lr = BASE_LR * 0.1
WEIGHT_DECAY = 0.05
WARMUP_EPOCHS = 5
USE_AMP = True

# =========================
# 损失
# =========================
LAMBDA_DICE = 1.0
LAMBDA_BCE = 1.0

# =========================
# 其他
# =========================
DEVICE = "cuda"
SEED = 42
NUM_WORKERS = 4
