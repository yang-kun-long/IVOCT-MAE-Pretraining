# F:\CN_seg\seven\utils\lr_sched.py
import math


def adjust_learning_rate(optimizer, epoch, epochs, warmup_epochs, base_lr):
    if epoch < warmup_epochs:
        # 避免 epoch1 直接 lr=0
        lr = base_lr * (epoch + 1) / warmup_epochs
    else:
        lr = base_lr * 0.5 * (
            1. + math.cos(math.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs))
        )

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr