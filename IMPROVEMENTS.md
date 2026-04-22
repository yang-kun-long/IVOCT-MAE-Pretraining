# 模型改进说明

## 版本历史

### v2 版本（初始）
- Patch size: 8
- Decoder: 4 层, 256 维
- Mask ratio: 70%
- Loss 权重: MSE=1.0, SSIM=0.2, Grad=0.1
- Epochs: 200
- **结果**: SSIM=0.253, PSNR=17.78 dB（较差）

### v3 版本（第一次改进）
- Decoder: 8 层, 384 维（增强重建能力）
- Mask ratio: 60%（降低难度）
- Loss 权重: MSE=1.0, SSIM=0.5, Grad=0.2
- Epochs: 300
- **结果**: SSIM=0.427, PSNR=23.77 dB（一般）
- **改进**: SSIM +68.8%, PSNR +33.7%

### v4 版本（第二次改进，当前）
- 保持 v3 的架构改进
- Mask ratio: 50%（进一步降低难度）
- Loss 权重: MSE=1.0, SSIM=0.7, Grad=0.2（更重视结构）
- Epochs: 400（更充分训练）
- **预期**: SSIM=0.50-0.55, PSNR=25-28 dB（一般-良好）

## v3 → v4 改进内容

| 参数 | v3 | v4 | 改进原因 |
|------|----|----|----------|
| MASK_RATIO | 0.60 | 0.50 | 进一步降低重建难度 |
| LAMBDA_SSIM | 0.5 | 0.7 | 更重视结构相似性 |
| EPOCHS | 300 | 400 | 更充分的训练 |

## 评估方法（新增两种脚本）

### 1. 遮挡重建评估（evaluate_masked.py）
```bash
cd seven
python evaluate_masked.py
```
- 测试模型在遮挡部分区域后的重建能力
- 这是 MAE 预训练的真实任务场景
- 使用配置的 mask_ratio 进行遮挡

### 2. 完整重建评估（evaluate_full.py）
```bash
cd seven
python evaluate_full.py
```
- 测试模型的编码-解码能力（自编码器性能）
- 不遮挡输入，评估完整重建质量
- 可以反映特征学习的质量

### 3. 旧版评估（evaluate_v2.py）
- 与 evaluate_masked.py 相同
- 保留用于兼容性
