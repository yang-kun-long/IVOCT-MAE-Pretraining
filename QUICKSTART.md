# 🚀 快速启动指南

## 准备工作（5 分钟）

### 1. 获取 AutoDL Token（可选但推荐）

访问：https://www.autodl.com/console/account/info

复制"开发者 Token"，用于接收训练通知到微信。

### 2. SSH 连接到服务器

```bash
ssh -p 48198 root@connect.bjb2.seetacloud.com
# 密码：k3Kf0Yw3CtZI
```

### 3. 设置通知 Token（可选）

```bash
export AUTODL_TOKEN="your_token_here"

# 或者永久保存
echo 'export AUTODL_TOKEN="your_token_here"' >> ~/.bashrc
source ~/.bashrc
```

### 4. 测试通知（可选）

```bash
cd /root/CN_seg/seven
python -c "from utils.notifier import Notifier; n = Notifier(); n.send('🧪 Test', 'Ready to train!', force=True) if n.enabled else print('Notifications disabled')"
```

如果设置正确，你会在微信收到测试消息。

---

## 启动训练（2 种方法）

### 方法 1：快速启动脚本（推荐）

```bash
cd /root/CN_seg
bash start_training.sh
```

脚本会自动：
- ✅ 激活 conda 环境
- ✅ 检查 Python 和 PyTorch
- ✅ 安装依赖（tqdm, requests）
- ✅ 测试适配器实现
- ✅ 确认后启动训练

### 方法 2：直接启动

```bash
cd /root/CN_seg
source /root/miniconda3/bin/activate
cd seven
python train_mae_v2.py
```

---

## 训练配置

当前配置（`seven/config_v2.py`）：

```python
# 数据集
数据集：4,554 图像（19 患者）

# 训练参数
EPOCHS = 200              # 训练轮数
BATCH_SIZE = 64           # 批大小（优化 5090 32GB）
BASE_LR = 3e-4            # 学习率
MASK_RATIO = 0.65         # 掩码比例

# 适配器微调
USE_ADAPTER = True        # 启用适配器
ADAPTER_BOTTLENECK = 64   # 瓶颈维度
FREEZE_MODE = "adapter_only"  # 冻结 encoder
```

**预计训练时间**：~16 小时（RTX 5090）

---

## 监控训练

### 实时日志

```bash
# 方法 1：tail（推荐）
tail -f /root/CN_seg/seven/logs_v2/train_*.log

# 方法 2：less（可滚动）
less +F /root/CN_seg/seven/logs_v2/train_*.log
```

### GPU 监控

```bash
# 实时监控
watch -n 1 nvidia-smi

# 或者单次查看
nvidia-smi
```

### 检查点

```bash
# 查看已保存的检查点
ls -lh /root/CN_seg/seven/checkpoints_v2/

# 查看可视化
ls -lh /root/CN_seg/seven/recon_vis_v2/
```

### 微信通知（如果已配置）

你会在以下时机收到通知：
- 🚀 训练开始
- 📊 每 20 epochs（Epoch 20, 40, 60...）
- ⭐ 每次保存最佳模型
- 🎉 训练完成
- ❌ 训练错误

---

## 训练输出

### 检查点（`seven/checkpoints_v2/`）

```
mae_v2_best.pth           # 最佳模型（用于下游任务）
mae_v2_encoder_only.pth   # 仅编码器权重
mae_v2_epoch_10.pth       # 每 10 epochs 保存
mae_v2_epoch_20.pth
...
mae_v2_epoch_200.pth      # 最后一个 epoch
```

### 日志（`seven/logs_v2/`）

```
train_YYYYMMDD_HHMMSS.log  # 训练日志（文本）
train_log_v2.json          # 训练指标（JSON）
```

### 可视化（`seven/recon_vis_v2/`）

```
recon_v2_epoch_1.png       # 第 1 epoch 重建
recon_v2_epoch_10.png      # 每 10 epochs
...
recon_v2_epoch_200.png     # 最后一个 epoch
```

---

## 中断和恢复

### 安全中断

```bash
# 按 Ctrl+C 一次（等待当前 epoch 完成）
# 或按两次（立即停止）
```

### 恢复训练（暂不支持）

当前版本不支持从检查点恢复。如果需要恢复：
1. 修改 `config_v2.py` 中的 `EPOCHS`
2. 手动加载检查点（需要修改代码）

---

## 下载结果

训练完成后，从本地机器运行：

```bash
# 下载检查点
uv run --with paramiko python scripts/remote_ops.py download \
  /root/CN_seg/seven/checkpoints_v2/mae_v2_best.pth \
  seven/checkpoints_v2/mae_v2_best.pth

# 下载日志
uv run --with paramiko python scripts/remote_ops.py download \
  /root/CN_seg/seven/logs_v2/ \
  seven/logs_v2/

# 下载可视化
uv run --with paramiko python scripts/remote_ops.py download \
  /root/CN_seg/seven/recon_vis_v2/ \
  seven/recon_vis_v2/
```

或者打包下载：

```bash
# 在服务器上打包
ssh -p 48198 root@connect.bjb2.seetacloud.com
cd /root/CN_seg/seven
tar -czf training_results.tar.gz checkpoints_v2/ logs_v2/ recon_vis_v2/

# 在本地下载
uv run --with paramiko python scripts/remote_ops.py download \
  /root/CN_seg/seven/training_results.tar.gz \
  training_results.tar.gz
```

---

## 故障排除

### 问题 1：CUDA out of memory

**解决：** 降低 batch size

```python
# 编辑 seven/config_v2.py
BATCH_SIZE = 48  # 或 32
```

### 问题 2：训练太慢

**检查：**
```bash
nvidia-smi  # 确认 GPU 利用率
htop        # 确认 CPU 和内存
```

**可能原因：**
- DataLoader workers 太少：增加 `NUM_WORKERS`
- 磁盘 I/O 慢：检查数据集位置

### 问题 3：通知不工作

**检查：**
```bash
echo $AUTODL_TOKEN  # 确认 token 已设置
```

**测试：**
```bash
cd /root/CN_seg/seven
python -c "from utils.notifier import Notifier; Notifier().send('Test', 'Hello', force=True)"
```

### 问题 4：训练中断

**检查日志：**
```bash
tail -100 /root/CN_seg/seven/logs_v2/train_*.log
```

**常见原因：**
- 显存不足：降低 batch size
- 磁盘满：清理旧文件
- 网络断开：重新连接 SSH

---

## 预期结果

### 训练指标

| Epoch | Total Loss | MSE Loss | SSIM Loss | Grad Loss |
|-------|-----------|----------|-----------|-----------|
| 1 | ~2.5 | ~2.0 | ~0.4 | ~0.1 |
| 50 | ~0.5 | ~0.3 | ~0.15 | ~0.05 |
| 100 | ~0.3 | ~0.2 | ~0.08 | ~0.02 |
| 200 | ~0.15 | ~0.1 | ~0.04 | ~0.01 |

### 重建质量

| 指标 | 预期值 | 说明 |
|------|--------|------|
| SSIM | 0.83-0.86 | 结构相似度 |
| PSNR | 25-30 dB | 峰值信噪比 |
| MSE | 0.001-0.005 | 均方误差 |

### 下游分割

使用预训练编码器进行分割任务：
- 预期 Dice score：0.74-0.77
- 相比无预训练：+2-5%

---

## 下一步

训练完成后：

1. **评估重建质量**
   ```bash
   cd /root/CN_seg/seven
   python evaluate_v2.py
   ```

2. **运行分割任务**
   ```bash
   cd /root/CN_seg/seven/seg
   python train_seg.py
   ```

3. **对比实验**（可选）
   - 全量训练：`USE_ADAPTER = False`
   - 冻结编码器：`FREEZE_MODE = "full"`

---

## 参考文档

- **适配器微调**：`ADAPTER_TUNING.md`
- **配置对比**：`CONFIG_COMPARISON.md`
- **通知设置**：`NOTIFICATION_SETUP.md`
- **实现总结**：`IMPLEMENTATION_SUMMARY.md`

---

## 联系方式

遇到问题？检查：
1. 训练日志：`seven/logs_v2/train_*.log`
2. 错误信息：最后 100 行日志
3. GPU 状态：`nvidia-smi`
4. 磁盘空间：`df -h`

**准备好了吗？开始训练吧！** 🚀

```bash
cd /root/CN_seg
bash start_training.sh
```
