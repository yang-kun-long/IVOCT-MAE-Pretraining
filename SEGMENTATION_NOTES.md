# IVOCT 分割开发踩坑记录

## 背景

- 数据集：4个患者，897张图，仅43张有CA（钙化）标注
- 预训练：MAE v4，patch_size=8，embed_dim=384，已完成400 epoch
- 目标：基于预训练encoder做钙化区域分割

---

## 坑1：MAE checkpoint路径错误

**现象**：`FileNotFoundError`，找不到checkpoint

**原因**：`config_seg.py`里写的路径是 `results/pretrain/v4/mae_v2_best.pth`，但服务器上实际路径是 `seven/checkpoints_v2/mae_v2_best.pth`

**修复**：
```python
# 错误
MAE_CHECKPOINT = ROOT_DIR / "results" / "pretrain" / "v4" / "mae_v2_best.pth"
# 正确
MAE_CHECKPOINT = SEVEN_DIR / "checkpoints_v2" / "mae_v2_best.pth"
```

---

## 坑2：Python模块名冲突

**现象**：`ImportError: cannot import name 'IVOCTSegDataset' from 'datasets'`

**原因**：`seven/` 和 `seven/seg/` 都有 `datasets/` 和 `models/` 目录，`sys.path` 顺序不对导致找到了错误的模块

**修复**：在 `train_seg.py` 里把 `seven/seg/` 插到 `seven/` 前面
```python
sys.path.insert(0, str(Path(__file__).parent))        # seven/seg/ 优先
sys.path.insert(1, str(Path(__file__).parent.parent))  # seven/ 其次
```

`seg_model.py` 里用 `importlib.util` 按文件路径直接加载 `mae_hybrid_v2.py`，绕过名称冲突：
```python
_mae_path = Path(__file__).parent.parent.parent / "models" / "mae_hybrid_v2.py"
_spec = importlib.util.spec_from_file_location("mae_hybrid_v2", _mae_path)
```

---

## 坑3：PATCH_SIZE配置与实际不符

**现象**：`config_v2.py` 写的是 `PATCH_SIZE=16`，但模型实际用的是 `PATCH_SIZE=8`

**原因**：`ConvStemPatchEmbed` 是3层 stride=2 的卷积（256→128→64→32），固定输出 32×32 token grid，等效 patch_size=8。`config_v2.py` 里的 `PATCH_SIZE=16` 是文档错误，实际代码有 `assert patch_size == 8`

**结论**：`config_seg.py` 里 `PATCH_SIZE=8` 是正确的，不要改

---

## 坑4：类别不平衡导致模型不收敛

**现象**：训练100 epoch，Val Dice 始终在 0.00~0.02，Loss 卡在 1.2

**原因**：钙化区域平均只占图像面积的 **1.4%**（最小0.2%，最大5%），BCE loss 被背景像素淹没，模型学会了"全预测背景"

**修复**：
1. 先尝试 `pos_weight` 自动加权（约70:1），但导致模型过度预测前景（36%像素被预测为前景）
2. 改用 **Focal loss**（alpha=0.75, gamma=2.0），压制简单背景像素的梯度

```python
def focal_loss(pred_logit, target, alpha=0.75, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(pred_logit, target, reduction='none')
    prob = torch.sigmoid(pred_logit)
    p_t = prob * target + (1 - prob) * (1 - target)
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    focal_weight = alpha_t * (1 - p_t) ** gamma
    return (focal_weight * bce).mean()
```

---

## 坑5：random_masking 打乱 token 空间顺序（最关键的bug）

**现象**：修复loss后，Dice仍然在 0.00~0.03 之间随机跳动，完全无规律

**原因**：`forward_encoder` 内部调用 `random_masking`，即使 `mask_ratio=0`（不遮挡任何token），`torch.argsort(noise)` 仍然会**打乱token顺序**。分割decoder把这些乱序的token reshape成 32×32 空间grid，导致空间位置完全错乱，模型看到的是随机打乱的特征图

**修复**：分割时绕过 `forward_encoder`，直接手动执行encoder的前向传播，跳过 `random_masking`：

```python
# 不要用这个（会打乱顺序）
latent, _, _ = self.encoder.forward_encoder(x, fg_mask, mask_ratio=0.0)

# 改成手动执行，保持空间顺序
with torch.no_grad():
    x = self.encoder.patch_embed(x)
    x = x + self.encoder.pos_embed[:, 1:, :]
    cls_token = self.encoder.cls_token + self.encoder.pos_embed[:, :1, :]
    cls_tokens = cls_token.expand(B, -1, -1)
    x = torch.cat((cls_tokens, x), dim=1)
    for blk in self.encoder.blocks:
        x = blk(x)
    latent = self.encoder.norm(x)
```

**效果**：修复后 Dice 从 epoch 1 开始稳定上升，epoch 10 达到 0.21，epoch 20 达到 0.24

---

## 坑6：原型分割 temperature 参数只有1个，无法学习

**现象**：`Learnable params: 1`，20个epoch Dice完全不变（0.072）

**原因**：原型分割本身是推理时方法，不需要训练。只有一个 temperature 标量，梯度信号极弱，学不到任何东西

**修复**：加入可学习的 **projection head**（384→128→64），让特征空间更适合分割任务

---

## 当前状态与结论

### Conv Decoder 方案（修复坑5后）
- 最佳 Dice：**0.274**（epoch 23）
- 问题：验证集波动大（0.03~0.27），过拟合严重
- 原因：33张训练样本，decoder参数量相对较大

### 原型分割方案
- 最佳 Dice：**0.197**（epoch 9）
- 问题：仍然不稳定，projection层容易过拟合
- 优势：理论上更适合小样本，不依赖大量参数

### 下一步方向
1. **Conv Decoder + 强数据增强**：弹性形变、更大旋转、MixUp
2. **原型分割 + 稳定训练**：固定support set、降低学习率、去掉BN改用LN
3. **半监督**：用897张无标注图生成伪标签扩充训练集

---

## 关键数据

| 指标 | 数值 |
|------|------|
| 总图像数 | 897 |
| 有标注图像数 | 43 |
| 钙化区域平均占比 | 1.4% |
| 最小占比 | 0.2% |
| 最大占比 | 5.0% |
| Token grid 尺寸 | 32×32（patch_size=8） |
| Encoder 输出维度 | 384 |

---

## 2026-04-23 服务器训练记录

本次在服务器上完成了一次完整的 LOPO-CV 分割训练。

### 训练命令

```bash
cd seven/seg
python train_seg.py --split lopo
```

### 最终结果

- Mean Dice：**0.5145 ± 0.1426**
- Fold 1（val=`P001`）：**0.5505**
- Fold 2（val=`P002`）：**0.2722**
- Fold 3（val=`P003`）：**0.6129**
- Fold 4（val=`P004`）：**0.6224**

### 结果位置

以下均为仓库相对路径，且文件当前存在于服务器上：

- `seven/seg/logs/results_lopo_20260423_222429.json`
- `seven/seg/logs/results_lopo_baseline.json`
- `seven/seg/checkpoints/seg_fold0_best.pth`
- `seven/seg/checkpoints/seg_fold1_best.pth`
- `seven/seg/checkpoints/seg_fold2_best.pth`
- `seven/seg/checkpoints/seg_fold3_best.pth`
- `seven/checkpoints_v2/mae_v2_best.pth`
- `seven/checkpoints_v2/mae_v2_encoder_only.pth`

### 当前判断

- 这套 `MAE encoder + Conv decoder` 方案已经证明有效
- 当前 baseline 可以作为后续调参对照
- 最大问题仍然是患者间泛化波动，`P002` 折最弱，后续优先分析这一折

---

## 2026-04-24 会话中断记录

### 已完成的额外实验

服务器上又完成了两轮基于 baseline 的调参实验：

- `seven/seg/logs/results_lopo_20260423_233510.json`
  - `BN + early stopping`
  - Mean Dice：**0.4786 ± 0.1291**
- `seven/seg/logs/results_lopo_20260423_235957.json`
  - `focal_tversky + BN + early stopping`
  - Mean Dice：**0.4994 ± 0.1648**

这两轮都没有超过最早的 baseline：

- `seven/seg/logs/results_lopo_20260423_222429.json`
  - Mean Dice：**0.5145 ± 0.1426**

### 中断时的工作状态

由于最早 baseline 的 `seg_fold*_best.pth` 已被后续实验覆盖，本次会话最后改为：

1. 在服务器上创建独立 worktree：`/root/CN_seg_baseline`
2. 将其固定到提交 `733df27`
3. 通过软链接接入：
   - `/root/CN_seg/DATA`
   - `/root/CN_seg/seven/checkpoints_v2`
4. 在该隔离目录中重跑最早 baseline，准备之后做真正可比的 threshold sweep

### 进行中的服务器任务

中断时，以下任务仍在运行：

- baseline 重跑目录：`/root/CN_seg_baseline/seven/seg`
- 日志：`/root/CN_seg_baseline/seven/seg/baseline_repro.log`

查看命令：

```bash
cd /root/CN_seg_baseline/seven/seg
tail -n 100 -f baseline_repro.log
```

### 下次续接

- 先确认 `baseline_repro.log` 是否已经跑完
- 如果已完成，立刻对这轮新生成的 baseline checkpoints 做 threshold sweep
- 再决定是否转向阈值优化或轻量后处理，而不是继续盲目改训练配置
