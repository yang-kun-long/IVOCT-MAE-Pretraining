# 分割训练指南

## ✅ 更新内容

### 1. 配置更新（`seven/seg/config_seg.py`）

**数据集扩展：**
- ✅ 患者数量：3 → 19（P001-P019）
- ✅ 图像数量：~100 → 4,554（45× 增长）
- ✅ 标注数量：~40 → 239（6× 增长）

**模型配置：**
- ✅ `PATCH_SIZE`: 8 → 16（匹配新训练的 MAE）
- ✅ `USE_ADAPTER`: True（加载适配器权重）
- ✅ `ADAPTER_BOTTLENECK`: 64（适配器瓶颈维度）

### 2. 模型更新（`seven/seg/models/seg_model.py`）

**新增功能：**
- ✅ 支持 patch_size=16 和 patch_size=8
- ✅ 支持加载适配器权重
- ✅ 自动适配解码器层数（16x16 grid → 4 层上采样）
- ✅ 更灵活的权重加载（strict=False）

**架构变化：**
```
patch_size=16: 16×16 grid → 4 层上采样 → 256×256
patch_size=8:  32×32 grid → 3 层上采样 → 256×256
```

### 3. 训练脚本更新（`seven/seg/train_seg.py`）

**新增参数：**
- `patch_size`: 传递给模型
- `use_adapter`: 是否加载适配器
- `adapter_bottleneck`: 适配器维度

---

## 🚀 开始分割训练

### 方法 1：使用启动脚本（推荐）

```bash
ssh -p 48198 root@connect.bjb2.seetacloud.com
cd /root/CN_seg
bash start_segmentation.sh
```

### 方法 2：直接运行

```bash
ssh -p 48198 root@connect.bjb2.seetacloud.com
source /root/miniconda3/bin/activate
cd /root/CN_seg/seven/seg
python train_seg.py
```

---

## 📊 训练配置

| 参数 | 值 | 说明 |
|------|-----|------|
| **数据集** | 19 患者 | P001-P019 |
| **Split 模式** | LOPO | Leave-One-Patient-Out |
| **Epochs** | 180 | 每个 fold |
| **Batch size** | 4 | 小 batch 适合分割 |
| **Learning rate** | 1e-3 | 只训练 decoder |
| **Freeze encoder** | True | 冻结预训练编码器 |
| **Loss** | Focal Tversky | 处理类别不平衡 |
| **Early stopping** | 50 epochs | 防止过拟合 |

---

## ⏱️ 预计训练时间

**LOPO 模式（19 folds）：**
- 每个 fold：~30-60 分钟（取决于 early stopping）
- 总时间：~10-20 小时

**建议：**
- 使用 `nohup` 或 `tmux` 后台运行
- 定期检查日志

---

## 📁 输出位置

训练完成后，结果保存在：

```
seven/seg/
├── checkpoints/
│   ├── seg_fold0_best.pth    # Fold 0 最佳模型
│   ├── seg_fold1_best.pth    # Fold 1 最佳模型
│   └── ...                    # 共 19 个 folds
├── logs/
│   └── results_lopo_YYYYMMDD_HHMMSS.json
└── vis/
    ├── fold_0/
    │   ├── epoch_1.png
    │   └── ...
    └── ...
```

---

## 📈 预期结果

基于预训练 MAE 编码器，预期分割性能：

| 指标 | 无预训练 | 有预训练 | 提升 |
|------|---------|---------|------|
| **Dice Score** | 0.70-0.72 | 0.74-0.77 | +2-5% |
| **IoU** | 0.55-0.58 | 0.59-0.63 | +4-5% |
| **Precision** | 0.75-0.78 | 0.78-0.82 | +3-4% |
| **Recall** | 0.68-0.72 | 0.72-0.76 | +4% |

---

## 🔍 监控训练

### 实时日志

```bash
# 查看最新日志
tail -f /root/CN_seg/seven/seg/logs/results_lopo_*.json

# 或者查看 Python 输出
ps aux | grep train_seg.py
```

### 检查进度

```bash
# 查看已完成的 folds
ls -lh /root/CN_seg/seven/seg/checkpoints/

# 查看可视化
ls -lh /root/CN_seg/seven/seg/vis/fold_*/
```

---

## 🐛 故障排除

### 问题 1：找不到 MAE checkpoint

**错误：**
```
FileNotFoundError: mae_v2_best.pth not found
```

**解决：**
```bash
# 检查 checkpoint 是否存在
ls -lh /root/CN_seg/seven/checkpoints_v2/mae_v2_best.pth

# 如果不存在，需要先完成预训练
cd /root/CN_seg
bash start_training.sh
```

### 问题 2：Patch size 不匹配

**错误：**
```
RuntimeError: size mismatch for pos_embed
```

**解决：**
确认 `seven/seg/config_seg.py` 中 `PATCH_SIZE = 16`

### 问题 3：显存不足

**错误：**
```
CUDA out of memory
```

**解决：**
```python
# 编辑 seven/seg/config_seg.py
BATCH_SIZE = 2  # 降低到 2
```

### 问题 4：数据集路径错误

**错误：**
```
No images found for patient P001
```

**解决：**
```bash
# 检查数据集结构
ls -lh /root/CN_seg/DATA/P001/Data/
ls -lh /root/CN_seg/DATA/P001/mask/

# 确认有 19 个患者目录
ls -d /root/CN_seg/DATA/P*/
```

---

## 📥 下载结果

训练完成后，下载结果到本地：

```bash
# 下载所有 checkpoints
uv run --with paramiko python scripts/remote_ops.py download \
  /root/CN_seg/seven/seg/checkpoints/ \
  seven/seg/checkpoints/

# 下载日志
uv run --with paramiko python scripts/remote_ops.py download \
  /root/CN_seg/seven/seg/logs/ \
  seven/seg/logs/

# 下载可视化（可选，文件较大）
uv run --with paramiko python scripts/remote_ops.py download \
  /root/CN_seg/seven/seg/vis/ \
  seven/seg/vis/
```

或者打包下载：

```bash
# 在服务器上打包
ssh -p 48198 root@connect.bjb2.seetacloud.com
cd /root/CN_seg/seven/seg
tar -czf seg_results.tar.gz checkpoints/ logs/ vis/

# 在本地下载
uv run --with paramiko python scripts/remote_ops.py download \
  /root/CN_seg/seven/seg/seg_results.tar.gz \
  seg_results.tar.gz
```

---

## 🎯 下一步

训练完成后：

1. **评估结果**
   ```bash
   cd /root/CN_seg/seven/seg
   python eval_seg.py
   ```

2. **分析性能**
   - 查看每个 fold 的 Dice score
   - 对比有/无预训练的差异
   - 分析失败案例

3. **可视化预测**
   - 查看 `vis/fold_*/` 中的预测结果
   - 对比 ground truth 和预测

4. **论文撰写**
   - 整理实验结果
   - 绘制性能对比图
   - 分析预训练的有效性

---

## 📚 相关文档

- **预训练指南**：`QUICKSTART.md`
- **适配器微调**：`ADAPTER_TUNING.md`
- **配置对比**：`CONFIG_COMPARISON.md`
- **实现总结**：`IMPLEMENTATION_SUMMARY.md`

---

**准备好了吗？开始分割训练吧！** 🚀

```bash
cd /root/CN_seg
bash start_segmentation.sh
```
