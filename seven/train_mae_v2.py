# F:\CN_seg\seven\train_mae_v2.py
from pathlib import Path
import json
import sys
import logging
import time
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import GradScaler
from tqdm import tqdm

import config_v2 as config
from datasets.ivoct_pretrain_dataset_v2 import IVOCTPretrainDatasetV2
from models.mae_hybrid_v2 import hybrid_mae_vit_small_patch8
from engine.pretrain_engine_v2 import train_one_epoch_v2
from utils.misc import set_seed, save_checkpoint
from utils.lr_sched import adjust_learning_rate
from utils.visualization_v2 import save_reconstruction_four_panel


def setup_logger():
    """设置日志记录器，同时输出到控制台和文件"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = config.LOG_DIR / f"train_{timestamp}.log"

    # 创建 logger
    logger = logging.getLogger("MAE_Training")
    logger.setLevel(logging.INFO)

    # 文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # 格式化
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger, log_file


def main():
    # 设置日志
    logger, log_file = setup_logger()
    logger.info("=" * 80)
    logger.info("Starting MAE Training")
    logger.info(f"Log file: {log_file}")
    logger.info("=" * 80)

    set_seed(config.SEED)
    logger.info(f"Random seed: {config.SEED}")

    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # 记录配置
    logger.info("-" * 80)
    logger.info("Configuration:")
    logger.info(f"  Dataset: {config.DATA_DIR}")
    logger.info(f"  Image size: {config.IMG_SIZE}")
    logger.info(f"  Batch size: {config.BATCH_SIZE}")
    logger.info(f"  Epochs: {config.EPOCHS}")
    logger.info(f"  Learning rate: {config.BASE_LR}")
    logger.info(f"  Warmup epochs: {config.WARMUP_EPOCHS}")
    logger.info(f"  Mask ratio: {config.MASK_RATIO}")
    logger.info(f"  Loss weights: MSE={config.LAMBDA_MSE}, SSIM={config.LAMBDA_SSIM}, Grad={config.LAMBDA_GRAD}")
    logger.info(f"  Adapter tuning: {config.USE_ADAPTER}")
    if config.USE_ADAPTER:
        logger.info(f"    Bottleneck dim: {config.ADAPTER_BOTTLENECK}")
        logger.info(f"    Freeze mode: {config.FREEZE_MODE}")
    logger.info("-" * 80)

    try:
        dataset = IVOCTPretrainDatasetV2(
            root_dir=config.DATA_DIR,
            img_size=config.IMG_SIZE,
            crop_ratio=config.ROI_CROP_RATIO,
            fg_threshold=config.FOREGROUND_THRESHOLD,
            image_exts=config.IMAGE_EXTS
        )
        logger.info(f"Dataset loaded: {len(dataset)} images")

        loader = DataLoader(
            dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True
        )
        logger.info(f"DataLoader created: {len(loader)} batches per epoch")

        model = hybrid_mae_vit_small_patch8(
            img_size=config.IMG_SIZE,
            in_chans=config.IN_CHANS,
            norm_pix_loss=config.NORM_PIX_LOSS,
            fg_mask_bias=config.FG_MASK_BIAS,
            mask_mode=config.MASK_MODE,
            use_adapter=config.USE_ADAPTER,
            adapter_bottleneck=config.ADAPTER_BOTTLENECK,
        ).to(device)

        # Apply freezing if using adapter tuning
        if config.USE_ADAPTER and config.FREEZE_MODE != "none":
            model.freeze_encoder(freeze_mode=config.FREEZE_MODE)

        # Get parameter statistics
        param_stats = model.get_trainable_params()
        logger.info(f"Model created:")
        logger.info(f"  Total params: {param_stats['total']:,}")
        logger.info(f"  Trainable params: {param_stats['trainable']:,} ({param_stats['trainable_ratio']:.2f}%)")
        logger.info(f"  Encoder params: {param_stats['encoder']:,}")
        logger.info(f"  Decoder params: {param_stats['decoder']:,}")
        if config.USE_ADAPTER:
            logger.info(f"  Adapter params: {param_stats['adapter']:,}")

        # Only optimize trainable parameters
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = AdamW(
            trainable_params,
            lr=config.BASE_LR,
            betas=config.BETAS,
            weight_decay=config.WEIGHT_DECAY
        )
        logger.info(f"Optimizer: AdamW (lr={config.BASE_LR}, weight_decay={config.WEIGHT_DECAY})")

        scaler = GradScaler(enabled=config.USE_AMP)
        logger.info(f"Mixed precision training: {config.USE_AMP}")

    except Exception as e:
        logger.error(f"Initialization failed: {e}", exc_info=True)
        raise

    log_history = []
    best_loss = 1e9
    start_time = time.time()

    logger.info("=" * 80)
    logger.info("Starting training loop")
    logger.info(f"Training mode: {'Adapter Tuning' if config.USE_ADAPTER else 'Full Training'}")
    if config.USE_ADAPTER:
        logger.info(f"  Freeze mode: {config.FREEZE_MODE}")
        logger.info(f"  Training {param_stats['trainable']:,} / {param_stats['total']:,} params ({param_stats['trainable_ratio']:.2f}%)")
    logger.info("=" * 80)

    for epoch in range(1, config.EPOCHS + 1):
        try:
            lr = adjust_learning_rate(
                optimizer,
                epoch - 1,
                config.EPOCHS,
                config.WARMUP_EPOCHS,
                config.BASE_LR
            )

            logger.info(f"Epoch {epoch}/{config.EPOCHS} - Learning rate: {lr:.8f}")

            metrics = train_one_epoch_v2(
                model=model,
                loader=loader,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                epoch=epoch,
                mask_ratio=config.MASK_RATIO,
                use_amp=config.USE_AMP,
                lambda_mse=config.LAMBDA_MSE,
                lambda_ssim=config.LAMBDA_SSIM,
                lambda_grad=config.LAMBDA_GRAD,
            )

            logger.info(
                f"Epoch {epoch} completed - "
                f"total_loss={metrics['total_loss']:.6f}, "
                f"mse={metrics['mse_loss']:.6f}, "
                f"ssim={metrics['ssim_loss']:.6f}, "
                f"grad={metrics['grad_loss']:.6f}"
            )

            log_item = {
                "epoch": epoch,
                "lr": lr,
                "total_loss": metrics["total_loss"],
                "mse_loss": metrics["mse_loss"],
                "ssim_loss": metrics["ssim_loss"],
                "grad_loss": metrics["grad_loss"],
            }
            log_history.append(log_item)

            if epoch % config.SAVE_FREQ == 0 or epoch == config.EPOCHS:
                ckpt_path = config.CHECKPOINT_DIR / f"mae_v2_epoch_{epoch}.pth"
                save_checkpoint({
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "metrics": metrics
                }, ckpt_path)
                logger.info(f"Checkpoint saved: {ckpt_path}")

            if metrics["total_loss"] < best_loss:
                best_loss = metrics["total_loss"]
                best_path = config.CHECKPOINT_DIR / "mae_v2_best.pth"
                save_checkpoint({
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "metrics": metrics
                }, best_path)
                logger.info(f"New best model saved: {best_path} (loss={best_loss:.6f})")

            if epoch % config.VIS_FREQ == 0 or epoch == 1:
                model.eval()
                batch = next(iter(loader))
                imgs = batch["image"].to(device)
                fg_mask = batch["fg_mask"].to(device)
                with torch.no_grad():
                    out = model(imgs, fg_mask, mask_ratio=config.MASK_RATIO)
                vis_path = config.RECON_DIR / f"recon_v2_epoch_{epoch}.png"
                save_reconstruction_four_panel(model, imgs, out["pred"], out["mask"], vis_path)
                logger.info(f"Visualization saved: {vis_path}")

            with open(config.LOG_DIR / "train_log_v2.json", "w", encoding="utf-8") as f:
                json.dump(log_history, f, indent=2)

        except Exception as e:
            logger.error(f"Error in epoch {epoch}: {e}", exc_info=True)
            raise

    # Training completed
    elapsed_time = time.time() - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    time_str = f"{hours}h {minutes}m"

    logger.info("=" * 80)
    logger.info("Training completed successfully")
    logger.info(f"Total time: {time_str}")
    logger.info("=" * 80)

    # 导出 encoder-only 权重
    try:
        best_ckpt = torch.load(config.CHECKPOINT_DIR / "mae_v2_best.pth", map_location="cpu")
        model_state = best_ckpt["model"]

        encoder_state = {}
        for k, v in model_state.items():
            if any([
                k.startswith("patch_embed"),
                k.startswith("cls_token"),
                k.startswith("pos_embed"),
                k.startswith("blocks"),
                k.startswith("norm"),
            ]):
                encoder_state[k] = v

        encoder_path = config.CHECKPOINT_DIR / "mae_v2_encoder_only.pth"
        torch.save(encoder_state, encoder_path)
        logger.info(f"Encoder-only weights saved: {encoder_path}")
        logger.info(f"Final best loss: {best_loss:.6f}")
    except Exception as e:
        logger.error(f"Error saving encoder weights: {e}", exc_info=True)


if __name__ == "__main__":
    main()
