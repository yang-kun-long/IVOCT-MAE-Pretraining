# 服务器部署指南

## 环境要求
- Python 3.8+
- CUDA 11.0+ (推荐 CUDA 11.8 或 12.1)
- GPU 显存 >= 8GB (batch_size=32 需要约 6-8GB)

## 部署步骤

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

如果遇到 OpenMP 冲突警告，设置环境变量：
```bash
export KMP_DUPLICATE_LIB_OK=TRUE
```

### 2. 验证环境
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import timm; print(f'timm: {timm.__version__}')"
```

### 3. 检查数据
确保 DATA 目录结构正确：
```
DATA/
├── P001/
│   ├── Data/       # 原始图像
│   ├── label/      # 标注文件
│   └── mask/       # 掩码图像
├── P002/
├── P003/
└── P004/
```

验证数据加载：
```bash
cd seven
python -c "from datasets.ivoct_pretrain_dataset_v2 import IVOCTPretrainDatasetV2; import config_v2 as config; ds = IVOCTPretrainDatasetV2(config.DATA_DIR, config.IMG_SIZE); print(f'Total images: {len(ds)}')"
```

### 4. 开始训练
```bash
cd seven
python train_mae_v2.py
```

训练过程会：
- 每 10 个 epoch 保存检查点到 `checkpoints_v2/`
- 每 10 个 epoch 保存可视化到 `recon_vis_v2/`
- 保存最佳模型为 `mae_v2_best.pth`
- 训练日志保存到 `logs_v2/train_log_v2.json`

### 5. 推理测试
训练完成后（或使用已有检查点）：
```bash
cd seven
python infer_reconstruct_v2.py
```

生成的重建可视化保存在 `recon_vis_v2/reconstruction_check_v2.png`

## 配置调整

如果显存不足，修改 `seven/config_v2.py`：
```python
BATCH_SIZE = 16  # 降低 batch size
NUM_WORKERS = 2  # 降低数据加载线程数
```

如果想加快训练（牺牲精度）：
```python
EPOCHS = 100     # 减少训练轮数
MASK_RATIO = 0.60  # 降低掩码比例
```

## 监控训练

使用 tmux 或 screen 保持训练会话：
```bash
tmux new -s mae_training
cd seven && python train_mae_v2.py
# Ctrl+B, D 分离会话
# tmux attach -t mae_training 重新连接
```

查看训练日志：
```bash
tail -f seven/logs_v2/train_log_v2.json
```

## 预期结果

- 训练时间：约 8-12 小时（200 epochs，单卡 V100/A100）
- 最终 loss：total_loss < 0.15（MSE + SSIM + Grad 加权和）
- 重建质量：SSIM > 0.85，能清晰重建组织边界
