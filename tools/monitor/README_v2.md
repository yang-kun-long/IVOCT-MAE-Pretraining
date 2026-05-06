# IVOCT 训练监控系统 v2

## 新功能

### 1. 实验管理
- 显示所有历史实验和正在运行的实验
- 实验卡片显示状态、时间、性能指标
- 点击实验卡片查看详细信息

### 2. 实时进度监控
- Epoch级别的实时进度条
- 每个fold的训练进度独立显示
- 实时更新的loss和dice曲线图
- 自动刷新（运行中实验每5秒，实验列表每10秒）

### 3. 实验配置
- 手动填写实验元数据（策略、参数、备注）
- 训练脚本自动读取并记录

## 使用方法

### 步骤1：配置实验信息（训练前）

编辑 `seven/seg/experiment_config.json`：

```json
{
  "experiment_name": "你的实验名称",
  "description": "实验描述",
  "start_time": "2026-05-05T10:00:00",
  "strategy": {
    "split_mode": "stratified_4fold",
    "num_folds": 4,
    "encoder_frozen": true,
    "use_adapter": true,
    "loss_function": "focal_tversky",
    "max_epochs": 180
  },
  "notes": "其他备注"
}
```

### 步骤2：修改训练脚本（集成 MonitorRun）

在训练脚本中添加进度追踪：

```python
from utils.monitoring import MonitorRun

monitor = MonitorRun(experiment_id="stratified_4fold_20260505", logs_dir=Path("logs"))

# 开始fold
monitor.start_fold(
    fold_idx=0,
    total_epochs=180,
    train_patients=train_patients,
    val_patients=val_patients
)

# 每个epoch结束后更新
monitor.update_epoch(
    fold_idx=0,
    epoch=epoch,
    train_loss=train_loss,
    val_dice=val_dice,
    val_iou=val_iou,
    is_best=(val_dice > best_dice)
)

# fold完成
monitor.finish_fold(fold_idx=0, best_dice=best_dice, metrics=metrics)

# 实验完成
monitor.finish(
    result_prefix="results_stratified",
    split_mode="stratified_4fold",
    mean_dice=mean_dice,
    fold_results=all_results,
)
```

### 步骤3：启动监控服务

```bash
# 在服务器上
cd /root/monitor
nohup python app.py > monitor.log 2>&1 &

# 或使用screen
screen -S monitor
python app.py
# Ctrl+A, D 分离
```

### 步骤4：访问监控界面

1. 配置autodl端口映射（容器端口6006）
2. 浏览器打开映射的公网URL
3. 查看实验列表，点击查看详情

## 文件结构

```
monitor/
├── app.py                    # Flask后端（已更新）
├── templates/
│   └── index.html           # 前端页面（已更新）
└── README.md                # 本文档

seven/seg/
├── experiment_config.json   # 实验配置（手动填写）
├── utils/
│   └── progress_tracker.py  # 进度追踪器（新增）
└── logs/
    ├── progress_*.json      # 实时进度文件（自动生成）
    └── results_*.json       # 最终结果文件（训练完成后）
```

## API接口

- `GET /api/experiments` - 获取所有实验列表
- `GET /api/experiment/<id>` - 获取实验详情
- `GET /api/experiment/<id>/realtime` - 获取实时进度数据
- `GET /api/config` - 获取实验配置模板

## 监控界面功能

### 实验列表
- 🟢 运行中：绿色边框，脉冲动画
- ✅ 已完成：蓝色标签，显示最终Dice
- ❌ 错误：红色标签

### 实时监控（运行中实验）
- 每个fold的进度条（当前epoch/总epoch）
- 实时loss曲线（按fold分组）
- 实时dice曲线（最佳epoch用黄色标记）
- 验证集患者列表

### 历史结果（已完成实验）
- 各fold的最终指标表格
- Dice, IoU, Sensitivity, Specificity
- 验证集患者分组

## 部署到服务器

```bash
# 本地打包
cd D:\学习\胡金亮\IVOCT-MAE-Pretraining
tar -czf monitor_v2.tar.gz monitor/ seven/seg/experiment_config.json seven/seg/utils/progress_tracker.py

# 上传
uv run python scripts/remote_ops.py upload monitor_v2.tar.gz /root/monitor_v2.tar.gz

# 服务器解压
uv run python scripts/remote_ops.py exec "cd /root && tar -xzf monitor_v2.tar.gz"

# 重启监控服务
uv run python scripts/remote_ops.py exec "pkill -f 'python.*app.py' && cd /root/monitor && nohup python app.py > monitor.log 2>&1 &"
```

## 注意事项

1. **实验ID唯一性**：每次实验使用不同的experiment_id，建议格式：`{策略}_{日期}`
2. **文件锁**：ProgressTracker使用fcntl文件锁，确保多进程安全
3. **刷新频率**：实时数据每5秒刷新，避免过于频繁的磁盘IO
4. **存储空间**：progress文件会随epoch增长，完成后可删除（保留results文件即可）

## 下一步改进

- [ ] 支持暂停/恢复实验
- [ ] 实验对比功能（多个实验并排对比）
- [ ] 导出实验报告（PDF/Markdown）
