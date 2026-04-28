# 2026-04-24 分割实验决策记录

本文件用于固定当前阶段的最终结论，避免继续围绕现有 4 个病人过度调参。

## 当前主线结论

当前应继续保留并使用 **reproduced baseline** 作为主线结果，不再继续围绕现有 4 个病人死磕调参。

原因：

- 当前全局最优结果来自 baseline 重跑，而不是后续调参。
- 阈值 sweep 基本没有带来实质提升。
- 针对 `P002` 的定向改动虽然能抬升单折结果，但会明显拉低全局 LOPO 表现。
- 在剩余 14 个病人到位前，继续围绕这 4 个病人优化，收益有限，且容易过拟合当前小样本分布。

## 当前最优实验策略

- 模型：`MAE encoder + Conv decoder`
- 训练方式：`LOPO-CV`
- 训练脚本：`seven/seg/train_seg.py`
- 运行目录：`/root/CN_seg_baseline/seven/seg`
- 运行命令：
  - `python train_seg.py --split lopo`
- 阈值策略：
  - 统一阈值继续使用 `0.3`
  - 该轮重跑后的 threshold sweep 结论仍然支持 `0.3` 作为全局统一阈值

## 当前最优结果

### 1. 全局最优主线结果

- 结果文件：`/root/CN_seg_baseline/seven/seg/logs/results_lopo_20260424_004401.json`
- Mean Dice：`0.5212 ± 0.1424`

分折结果：

- Fold 1，验证集 `P001`：`0.5288`
- Fold 2，验证集 `P002`：`0.2861`
- Fold 3，验证集 `P003`：`0.6323`
- Fold 4，验证集 `P004`：`0.6374`

配套文件：

- checkpoints：
  - `/root/CN_seg_baseline/seven/seg/checkpoints/seg_fold0_best.pth`
  - `/root/CN_seg_baseline/seven/seg/checkpoints/seg_fold1_best.pth`
  - `/root/CN_seg_baseline/seven/seg/checkpoints/seg_fold2_best.pth`
  - `/root/CN_seg_baseline/seven/seg/checkpoints/seg_fold3_best.pth`
- threshold sweep：
  - `/root/CN_seg_baseline/seven/seg/logs/threshold_sweep_summary.json`
- review pack：
  - `/root/CN_seg_baseline/result/review_pack_20260424_baseline.tar.gz`

本地同步文件：

- [threshold_sweep_summary_20260424_baseline.json](D:/ykl/hjl/CN_seg/result/threshold_sweep_summary_20260424_baseline.json)
- [review_pack_20260424_baseline.tar.gz](D:/ykl/hjl/CN_seg/result/review_pack_20260424_baseline.tar.gz)
- [review_pack_20260424_baseline_checkpoint_summary.json](D:/ykl/hjl/CN_seg/result/review_pack_20260424_baseline_checkpoint_summary.json)

### 2. 不作为主线、但值得记录的结果

#### `P002` 定向最优单折

目标：只提升当前最弱折 `P002`

- 配置：`focal_tversky + BN + early stopping`
- 运行目录：`/root/CN_seg_ftv_p002/seven/seg`
- 结果文件：`/root/CN_seg_ftv_p002/seven/seg/logs/results_single_holdout_20260424_131527.json`
- 单折 Dice：`0.3048`
- 单折 threshold sweep 最优约为 `0.6`

说明：

- 该配置对 `P002` 单折优于 baseline 的 `0.2861`
- 但它并不适合作为全局主线

#### 不推荐作为全局主线的 LOPO 方案

- 配置：`focal_tversky + BN + early stopping`
- 运行目录：`/root/CN_seg_ftv_lopo/seven/seg`
- 结果文件：`/root/CN_seg_ftv_lopo/seven/seg/logs/results_lopo_20260424_140731.json`
- Mean Dice：`0.4561 ± 0.1289`

分折结果：

- `P001`: `0.3376`
- `P002`: `0.3223`
- `P003`: `0.5450`
- `P004`: `0.6193`

结论：

- 虽然 `P002` 从 `0.2861` 提升到 `0.3223`
- 但 `P001`、`P003` 被明显拉低
- 全局结果显著低于当前主线 baseline，不采用

## 已验证但不继续推进的方向

### 1. 全局后处理规则

已验证“删掉上方假阳性连通域”的两版后处理规则。

结论：

- `P002` 能被明显拉起来
- 但 `P003`、`P004` 会被严重打坏
- 不适合做全局统一后处理

### 2. 调 focal `alpha`

实验目录：`/root/CN_seg_alpha050_p002/seven/seg`

- 配置：`alpha=0.5`
- 结果文件：`/root/CN_seg_alpha050_p002/seven/seg/logs/results_single_holdout_20260424_125055.json`
- `P002` 单折 Dice：`0.2762`

结论：

- 低于 baseline 的 `0.2861`
- 不继续

### 3. `BN + early stopping` 单独使用

实验目录：`/root/CN_seg_bn_es_p002/seven/seg`

- 结果文件：`/root/CN_seg_bn_es_p002/seven/seg/logs/results_single_holdout_20260424_125628.json`
- `P002` 单折 Dice：`0.2409`

结论：

- 明显差于 baseline
- 不继续

## 当前建议

在剩余 14 个病人数据到位前，当前阶段应执行以下策略：

1. 将 `/root/CN_seg_baseline` 这轮重跑结果视为当前正式 baseline。
2. 不再继续围绕当前 4 个病人做全局 loss、阈值、几何后处理的死磕优化。
3. 等待剩余 14 个病人加入后，再重新做：
   - LOPO/分层评估
   - patient-level 泛化分析
   - 是否需要重训或改 loss 的判断

## 一句话结论

当前最优主线是：

- `reproduced baseline`
- `results_lopo_20260424_004401.json`
- Mean Dice `0.5212 ± 0.1424`

在新增 14 个病人到来之前，先停在这里，不再围绕现有 4 个病人继续深挖。
