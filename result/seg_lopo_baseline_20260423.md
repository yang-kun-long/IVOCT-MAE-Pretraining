# 2026-04-23 LOPO 分割训练记录

本记录对应服务器上的一次完整 LOPO-CV 分割训练结果。

## 训练说明

- 任务：基于 MAE 预训练 encoder 的钙化分割
- 运行脚本：`seven/seg/train_seg.py`
- 运行方式：`python train_seg.py --split lopo`
- 训练位置：服务器环境

## 最终结果

- LOPO 平均 Dice：`0.5145 ± 0.1426`

分折结果：

- Fold 1，验证集 `P001`：`Dice = 0.5505`
- Fold 2，验证集 `P002`：`Dice = 0.2722`
- Fold 3，验证集 `P003`：`Dice = 0.6129`
- Fold 4，验证集 `P004`：`Dice = 0.6224`

补充：

- Fold 4 best：`Dice = 0.6224`，`epoch 80`
- 当前最弱折为 `P002`

## 结果文件位置

以下路径均为仓库相对路径，且当前文件存在于服务器上：

- 汇总结果：`seven/seg/logs/results_lopo_20260423_222429.json`
- 基线备份：`seven/seg/logs/results_lopo_baseline.json`
- Fold 0 最优权重：`seven/seg/checkpoints/seg_fold0_best.pth`
- Fold 1 最优权重：`seven/seg/checkpoints/seg_fold1_best.pth`
- Fold 2 最优权重：`seven/seg/checkpoints/seg_fold2_best.pth`
- Fold 3 最优权重：`seven/seg/checkpoints/seg_fold3_best.pth`

## 相关前置文件

以下前置文件也位于服务器上，可用于复现实验：

- MAE 最优权重：`seven/checkpoints_v2/mae_v2_best.pth`
- MAE encoder-only 权重：`seven/checkpoints_v2/mae_v2_encoder_only.pth`
- MAE 训练日志：`seven/logs_v2/train_log_v2.json`

## 当前结论

- 当前方案有效，LOPO 平均 Dice 已超过 `0.5`
- 患者间泛化波动较大，尤其是 `P002` 折明显偏低
- 该结果可作为当前分割 baseline
