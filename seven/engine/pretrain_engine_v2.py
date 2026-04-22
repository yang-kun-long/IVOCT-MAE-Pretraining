# F:\CN_seg\seven\engine\pretrain_engine_v2.py
from tqdm import tqdm
import torch

from utils.misc import AverageMeter
from utils.losses_v2 import ssim_loss, gradient_loss


def train_one_epoch_v2(
    model,
    loader,
    optimizer,
    scaler,
    device,
    epoch,
    mask_ratio,
    use_amp,
    lambda_mse,
    lambda_ssim,
    lambda_grad,
):
    model.train()

    total_meter = AverageMeter()
    mse_meter = AverageMeter()
    ssim_meter = AverageMeter()
    grad_meter = AverageMeter()

    pbar = tqdm(loader, total=len(loader), desc=f"Epoch {epoch}")
    for batch in pbar:
        imgs = batch["image"].to(device, non_blocking=True)
        fg_mask = batch["fg_mask"].to(device, non_blocking=True)

        optimizer.zero_grad()

        if use_amp:
            with torch.cuda.amp.autocast():
                out = model(imgs, fg_mask, mask_ratio=mask_ratio)
                loss_mse = out["loss_mse"]

                recon = model.unpatchify(out["pred"])
                loss_ssim = ssim_loss(recon, imgs, fg_mask=fg_mask)
                loss_grad = gradient_loss(recon, imgs, fg_mask=fg_mask)

                total_loss = (
                    lambda_mse * loss_mse
                    + lambda_ssim * loss_ssim
                    + lambda_grad * loss_grad
                )

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(imgs, fg_mask, mask_ratio=mask_ratio)
            loss_mse = out["loss_mse"]

            recon = model.unpatchify(out["pred"])
            loss_ssim = ssim_loss(recon, imgs, fg_mask=fg_mask)
            loss_grad = gradient_loss(recon, imgs, fg_mask=fg_mask)

            total_loss = (
                lambda_mse * loss_mse
                + lambda_ssim * loss_ssim
                + lambda_grad * loss_grad
            )

            total_loss.backward()
            optimizer.step()

        total_meter.update(total_loss.item(), imgs.size(0))
        mse_meter.update(loss_mse.item(), imgs.size(0))
        ssim_meter.update(loss_ssim.item(), imgs.size(0))
        grad_meter.update(loss_grad.item(), imgs.size(0))

        pbar.set_postfix(
            total=f"{total_meter.avg:.6f}",
            mse=f"{mse_meter.avg:.6f}",
            ssim=f"{ssim_meter.avg:.6f}",
            grad=f"{grad_meter.avg:.6f}"
        )

    return {
        "total_loss": total_meter.avg,
        "mse_loss": mse_meter.avg,
        "ssim_loss": ssim_meter.avg,
        "grad_loss": grad_meter.avg,
    }