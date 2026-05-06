from pathlib import Path

# =========================
# 路径
# =========================
ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "DATA"
SEVEN_DIR = ROOT_DIR / "seven"

MAE_CHECKPOINT = SEVEN_DIR / "checkpoints_v2" / "mae_v2_best.pth"
SEG_CHECKPOINT_DIR = SEVEN_DIR / "seg" / "checkpoints"
SEG_LOG_DIR = SEVEN_DIR / "seg" / "logs"
SEG_VIS_DIR = SEVEN_DIR / "seg" / "vis"

SEG_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
SEG_LOG_DIR.mkdir(parents=True, exist_ok=True)
SEG_VIS_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# 数据
# =========================
PATIENTS = ["P001", "P002", "P003", "P004"]  # 恢复旧的 4 个患者，验证是否能复现 0.55
SPLIT_MODE = "lopo"          # "lopo" | "single_holdout"
HOLDOUT_PATIENT = "P004"     # 仅 single_holdout 时使用

IMG_SIZE = 256
ROI_CROP_RATIO = 0.86        # 与预训练一致
IN_CHANS = 1

# =========================
# 模型（必须与 checkpoint 匹配）
# =========================
PATCH_SIZE = 8               # 匹配预训练模型（实际是 8，不是 16）
EMBED_DIM = 384
DEPTH = 12
NUM_HEADS = 6
USE_ADAPTER = True           # 新增：加载适配器权重
ADAPTER_BOTTLENECK = 64      # 新增：适配器瓶颈维度
FREEZE_ENCODER = True        # 先冻结encoder，只训练decoder

# =========================
# 训练
# =========================
EPOCHS = 180
BATCH_SIZE = 4
BASE_LR = 1e-3               # 回到 baseline 学习率，只保留 early stopping
ENCODER_LR_SCALE = 0.0       # 冻结时无效
WEIGHT_DECAY = 0.05
WARMUP_EPOCHS = 5
USE_AMP = True
MIN_EPOCHS = 80
EARLY_STOPPING_PATIENCE = 50
EVAL_THRESHOLD = 0.3         # 与 baseline 一致；新 checkpoint 再做阈值 sweep

# =========================
# 损失
# =========================
LOSS_MODE = "dice_bce"       # 恢复 baseline：简单的 Dice + BCE
LAMBDA_DICE = 1.0
LAMBDA_BCE = 1.0
TVERSKY_ALPHA = 0.3          # focal_tversky 模式才用
TVERSKY_BETA = 0.7
FOCAL_TVERSKY_GAMMA = 1.33

# =========================
# 其他
# =========================
DEVICE = "cuda"
SEED = 42
NUM_WORKERS = 4
