# 训练通知配置指南

## 概述

训练通知系统支持两种后端：
1. **AutoDL API**（推荐，适用于 AutoDL 平台）
2. **Server酱**（通用，适用于任何平台）

## AutoDL API（推荐）

### 1. 获取 Token

访问：https://www.autodl.com/console/account/info

在"开发者 Token"部分复制你的 token。

### 2. 设置环境变量

```bash
# 在远程服务器上
export AUTODL_TOKEN="your_token_here"

# 或者添加到 ~/.bashrc 永久生效
echo 'export AUTODL_TOKEN="your_token_here"' >> ~/.bashrc
source ~/.bashrc
```

### 3. 速率限制

- **限制**：1 条消息/分钟
- **配额**：100 条消息/天（免费版）
- **策略**：我们的实现会自动遵守速率限制

### 4. 通知时机

为了遵守速率限制，我们只在以下时机发送通知：

| 事件 | 频率 | 说明 |
|------|------|------|
| 训练开始 | 1 次 | 显示配置信息 |
| Epoch 完成 | 每 20 epochs | 显示损失指标 |
| 最佳模型 | 每次更新 | 仅当损失降低时 |
| 训练完成 | 1 次 | 显示最终结果 |
| 训练错误 | 每次 | 立即通知错误 |

**预计通知数量**（200 epochs）：
- 训练开始：1 条
- Epoch 里程碑：10 条（每 20 epochs）
- 最佳模型：~5-10 条（估计）
- 训练完成：1 条
- **总计**：~17-22 条（远低于 100 条/天限制）

## Server酱（备选）

### 1. 获取 SendKey

访问：https://sct.ftqq.com/

使用微信扫码登录，获取 SendKey。

### 2. 设置环境变量

```bash
export SERVERCHAN_KEY="your_sendkey_here"
```

### 3. 修改训练脚本

编辑 `seven/train_mae_v2.py`：

```python
# 将这一行
notifier = Notifier()  # 默认使用 autodl

# 改为
notifier = Notifier(backend="serverchan")
```

## 使用方法

### 方法 1：环境变量（推荐）

```bash
# SSH 到服务器
ssh -p 48198 root@connect.bjb2.seetacloud.com

# 设置 token
export AUTODL_TOKEN="your_token_here"

# 启动训练
cd /root/CN_seg/seven
python train_mae_v2.py
```

### 方法 2：添加到启动脚本

编辑 `start_training.sh`，在训练前添加：

```bash
# 在 "Starting training..." 之前添加
export AUTODL_TOKEN="your_token_here"
```

### 方法 3：添加到 bashrc（永久）

```bash
echo 'export AUTODL_TOKEN="your_token_here"' >> ~/.bashrc
source ~/.bashrc
```

## 测试通知

在启动训练前，测试通知是否工作：

```bash
cd /root/CN_seg/seven
python -c "
from utils.notifier import Notifier
notifier = Notifier(backend='autodl')
if notifier.enabled:
    notifier.send('🧪 Test', 'Notification system is working!', force=True)
else:
    print('Please set AUTODL_TOKEN environment variable')
"
```

如果成功，你会在微信收到测试消息。

## 通知示例

### 训练开始
```
🚀 MAE Training Started
[2026-05-04 15:30:00]
Dataset: 4554 images
Batch: 64, Epochs: 200
Mode: Adapter Tuning
GPU: NVIDIA GeForce RTX 5090
```

### Epoch 完成
```
📊 Epoch 20/200
[2026-05-04 16:45:00]
Loss: 0.123456
MSE: 0.045678
SSIM: 0.067890
```

### 最佳模型
```
⭐ New Best Model
[2026-05-04 17:30:00]
Epoch 45, Loss: 0.098765
```

### 训练完成
```
🎉 Training Completed
[2026-05-05 07:30:00]
Epochs: 200
Best Loss: 0.087654
Time: 16h 0m
```

### 训练错误
```
❌ Training Error
[2026-05-04 18:00:00]
Epoch 67 failed
CUDA out of memory...
```

## 故障排除

### 问题 1：没有收到通知

**检查：**
```bash
# 1. 确认 token 已设置
echo $AUTODL_TOKEN

# 2. 测试通知
cd /root/CN_seg/seven
python -c "from utils.notifier import Notifier; n = Notifier(); print(f'Enabled: {n.enabled}')"
```

**解决：**
- 确保 token 正确
- 确保网络连接正常
- 检查 AutoDL 配额是否用完

### 问题 2：速率限制错误

**现象：**
```
[Notifier] Rate limit: wait 45s before next notification
```

**说明：** 这是正常的，系统会自动跳过该通知以遵守速率限制。

### 问题 3：通知太频繁

**调整：** 编辑 `seven/utils/notifier.py`

```python
# 修改 epoch 通知频率
def epoch_completed(self, epoch: int, total: int, metrics: dict) -> bool:
    # 从每 20 epochs 改为每 50 epochs
    if epoch % 50 != 0 and epoch != total:
        return False
```

## 禁用通知

如果不需要通知，只需不设置环境变量：

```bash
# 不设置 AUTODL_TOKEN 或 SERVERCHAN_KEY
# 训练脚本会自动检测并禁用通知
python train_mae_v2.py
```

日志中会显示：
```
[Notifier] Warning: No token provided for autodl. Notifications disabled.
```

## 安全建议

1. **不要将 token 提交到 git**
   - Token 应该只存在于环境变量中
   - 不要写入代码或配置文件

2. **定期更换 token**
   - 如果 token 泄露，立即在 AutoDL 控制台重新生成

3. **限制 token 权限**
   - AutoDL token 仅用于消息推送
   - 不要在不安全的环境中使用

## 高级配置

### 自定义速率限制

```python
# 修改 train_mae_v2.py
notifier = Notifier(backend="autodl", rate_limit=120.0)  # 2 分钟
```

### 强制发送（跳过速率限制）

```python
# 仅用于重要通知
notifier.send("Critical Alert", "Something important", force=True)
```

### 多后端支持

```python
# 同时使用两个后端
notifier1 = Notifier(backend="autodl")
notifier2 = Notifier(backend="serverchan")

# 发送到两个后端
notifier1.send("Training Started", "...")
notifier2.send("Training Started", "...")
```

## 参考链接

- AutoDL 消息推送文档：https://www.autodl.com/docs/msg/
- AutoDL 控制台：https://www.autodl.com/console/account/info
- Server酱官网：https://sct.ftqq.com/
