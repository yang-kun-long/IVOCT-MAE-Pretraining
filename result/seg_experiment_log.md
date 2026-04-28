# 分割实验记录汇总

本文件汇总所有 LOPO 分割实验结果，按时间顺序排列，便于横向对比。

---

## 基线参考

| 折 | 验证集 | Dice |
|----|--------|------|
| Fold 1 | P001 | 0.5288 |
| Fold 2 | P002 | 0.2861 |
| Fold 3 | P003 | 0.6323 |
| Fold 4 | P004 | 0.6374 |
| **均值** | | **0.5212 ± 0.1424** |

> 来源：`/root/CN_seg_baseline/seven/seg/logs/results_lopo_20260424_004401.json`
> 配置：dice_focal loss，冻结 encoder，early stopping，threshold=0.3

---

## 实验列表

### EXP-01｜Baseline v1（首次跑通）
- 日期：2026-04-23
- 配置：dice_focal loss，无 BN，无 early stopping
- **Mean Dice：0.5145 ± 0.1426**
- 分折：P001=0.5505, P002=0.2722, P003=0.6129, P004=0.6224
- 日志：`seven/seg/logs/results_lopo_20260423_222429.json`
- 结论：首次跑通，确认方案可行

---

### EXP-02｜BN + early stopping
- 日期：2026-04-23
- 配置：在 EXP-01 基础上加 BatchNorm + early stopping
- **Mean Dice：0.4786 ± 0.1291**
- 日志：`seven/seg/logs/results_lopo_20260423_233510.json`
- 结论：低于 baseline，不采用

---

### EXP-03｜focal_tversky + BN + early stopping
- 日期：2026-04-23
- 配置：focal_tversky loss + BN + early stopping
- **Mean Dice：0.4994 ± 0.1648**
- 日志：`seven/seg/logs/results_lopo_20260423_235957.json`
- 结论：低于 baseline，不采用

---

### EXP-04｜Baseline 重跑（当前官方基线）⭐
- 日期：2026-04-24
- 配置：与 EXP-01 相同，在隔离 worktree `/root/CN_seg_baseline` 中重跑
- **Mean Dice：0.5212 ± 0.1424**
- 分折：P001=0.5288, P002=0.2861, P003=0.6323, P004=0.6374
- 日志：`/root/CN_seg_baseline/seven/seg/logs/results_lopo_20260424_004401.json`
- 结论：当前最优主线，作为后续所有实验的对比基准

---

### EXP-05｜focal_tversky LOPO（全局）
- 日期：2026-04-24
- 配置：focal_tversky + BN + early stopping，LOPO 全局
- **Mean Dice：0.4561 ± 0.1289**
- 分折：P001=0.3376, P002=0.3223, P003=0.5450, P004=0.6193
- 日志：`/root/CN_seg_ftv_lopo/seven/seg/logs/results_lopo_20260424_140731.json`
- 结论：P002 有提升但 P001/P003 被严重拉低，全局低于基线，不采用

---

### EXP-06｜P002 定向优化（single holdout）
- 日期：2026-04-24
- 配置：focal_tversky + BN + early stopping，仅验证 P002
- **P002 Dice：0.3048**（基线 0.2861，+0.019）
- 日志：`/root/CN_seg_ftv_p002/seven/seg/logs/results_single_holdout_20260424_131527.json`
- 结论：单折有提升，但不适合作为全局方案

---

### EXP-07｜alpha=0.5 调参（P002 single holdout）
- 日期：2026-04-24
- 配置：focal_tversky alpha=0.5，仅验证 P002
- **P002 Dice：0.2762**（低于基线）
- 日志：`/root/CN_seg_alpha050_p002/seven/seg/logs/results_single_holdout_20260424_125055.json`
- 结论：不继续

---

### EXP-08｜BN + early stopping（P002 single holdout）
- 日期：2026-04-24
- 配置：BN + early stopping，仅验证 P002
- **P002 Dice：0.2409**（明显低于基线）
- 日志：`/root/CN_seg_bn_es_p002/seven/seg/logs/results_single_holdout_20260424_125628.json`
- 结论：不继续

---

### EXP-09｜数据增强 v1（强增强）
- 日期：2026-04-28
- 配置：基线 + 对比度 jitter(0.85~1.15) + 高斯噪声(std=0.02) + 仿射变换(平移±5%, 缩放0.95~1.05)
- **Mean Dice：0.5011 ± 0.1074**
- 分折：P001=0.4879, P002=0.3317, P003=0.5743, P004=0.6103
- 日志：`/root/CN_seg_baseline/seven/seg/logs/aug_v1_train.log`
- 结论：P002 +0.046 有提升，标准差从 0.1424 降至 0.1074（各折更稳定），但 P001/P003/P004 均有下降，全局低于基线 -0.020。增强强度过大，需降低噪声和仿射幅度

---

### EXP-10｜数据增强 v2（弱增强）⭐ 当前最优
- 日期：2026-04-28
- 配置：基线 + 对比度 jitter(0.85~1.15) + 高斯噪声(std=0.005) + 仿射变换(平移±3%, 缩放0.97~1.03)
- **Mean Dice：0.5387 ± 0.1710**
- 分折：P001=0.5908, P002=0.2466, P003=0.6671, P004=0.6502
- 日志：`/root/CN_seg_baseline/seven/seg/logs/aug_v2_train.log`
- 结论：P001/P003/P004 均超过基线，全局均值超基线 +0.0175。P002 仍是弱折（低于基线），标准差偏大。当前最优方案

---

### EXP-11｜去除 P002，3 折 LOPO + 弱增强
- 日期：2026-04-28
- 配置：EXP-10 弱增强配置，PATIENTS=["P001","P003","P004"]，180 epochs
- 注：P002 因标注质量存疑（面积与 P001 相近但 Dice 持续异常低）被排除
- **Mean Dice：0.5324 ± 0.1515**
- 分折：P001=0.3296, P003=0.6935, P004=0.5743
- 日志：`/root/CN_seg_baseline/seven/seg/logs/no_p002_train.log`
- 结论：P003 超过 EXP-10，但 P001 大幅下滑（0.59→0.33）。训练集从 33 降到 19 个样本，P001 折只剩 P003+P004 训练，数据量不足是主因。整体均值略低于 EXP-10（0.5324 vs 0.5387），去除 P002 并未带来全局提升

| 实验 | 配置摘要 | Mean Dice | vs 基线 |
|------|----------|-----------|---------|
| EXP-10 ⭐ | 弱增强 | 0.5387 ± 0.1710 | +0.018 |
| EXP-04 ⭐ | 官方基线 | 0.5212 ± 0.1424 | — |
| EXP-01 | 首次跑通 | 0.5145 ± 0.1426 | -0.007 |
| EXP-09 | 强增强 | 0.5011 ± 0.1074 | -0.020 |
| EXP-03 | focal_tversky+BN+ES | 0.4994 ± 0.1648 | -0.022 |
| EXP-02 | BN+ES | 0.4786 ± 0.1291 | -0.043 |
| EXP-05 | focal_tversky LOPO | 0.4561 ± 0.1289 | -0.065 |

---

## 当前结论

- 所有调参方向目前均未超过 EXP-04 基线
- EXP-09 的增强使 P002 提升 +0.046，标准差下降，但整体均值略低
- 等待 EXP-10（弱增强）结果，判断是否能在保住 P002 提升的同时减少对强折的损伤
- 在新增 14 个病人数据到位前，不做大幅架构改动
