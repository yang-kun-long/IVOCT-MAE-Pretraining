# 2026-04-24 中断记录

本记录用于保存本次会话中断时的服务器状态，便于下次直接续跑。

## 当前结论

- 原始 baseline 仍然是当前最强结果：
  - `seven/seg/logs/results_lopo_20260423_222429.json`
  - Mean Dice：`0.5145 ± 0.1426`
- 后续两轮调参结果都未超过 baseline：
  - `seven/seg/logs/results_lopo_20260423_233510.json`
    - `BN + early stopping`
    - Mean Dice：`0.4786 ± 0.1291`
  - `seven/seg/logs/results_lopo_20260423_235957.json`
    - `focal_tversky + BN + early stopping`
    - Mean Dice：`0.4994 ± 0.1648`

## 为什么要重跑 baseline

服务器上的 `seven/seg/checkpoints/seg_fold*_best.pth` 已经被后续实验覆盖，无法再直接对最早那版 baseline 做“真正可比”的阈值 sweep。

因此本次会话最后采取的策略是：

1. 在服务器上建立一个独立 worktree：`/root/CN_seg_baseline`
2. 将该 worktree 指向提交 `733df27`
3. 将主仓库的 `DATA/` 和 `seven/checkpoints_v2/` 以软链接方式接入该 worktree
4. 在该隔离目录中重新跑最早那版 baseline

## 服务器上的进行中任务

中断时，baseline 重跑任务仍在服务器上进行中。

- worktree 路径：`/root/CN_seg_baseline`
- 训练目录：`/root/CN_seg_baseline/seven/seg`
- 日志文件：`/root/CN_seg_baseline/seven/seg/baseline_repro.log`
- 命令：
  - `cd /root/CN_seg_baseline/seven/seg`
  - `nohup /root/miniconda3/bin/python train_seg.py --split lopo > baseline_repro.log 2>&1 < /dev/null &`

## 中断时观测到的进度

以中断前最后一次检查为准：

- baseline 重跑已推进到约 `Epoch 38/300`
- 日志表现与最早 baseline 的高波动现象一致
- 尚未产生新的最终 `results_lopo_*.json`

## 下次续接建议

1. 先检查 `baseline_repro.log` 是否已经跑完
2. 如果跑完，记录新的 baseline 结果文件位置
3. 立即基于这轮新生成的 baseline checkpoints 做 threshold sweep
4. 再决定是否继续做后处理或阈值优化
