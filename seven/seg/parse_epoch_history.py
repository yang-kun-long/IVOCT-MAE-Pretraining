"""
从训练日志解析epoch数据，为历史实验补充epoch历史
Parse epoch data from training logs for historical experiments
"""
import re
import json
from pathlib import Path
from datetime import datetime


def parse_training_log(log_file: Path):
    """解析训练日志，提取epoch数据"""
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()

    fold_header_re = re.compile(r"Fold\s+(\d+):\s*Train=\[(.*?)\],\s*Val=\[(.*?)\]")
    fold_matches = list(fold_header_re.finditer(content))

    all_folds = []

    for idx, fold_match in enumerate(fold_matches):
        fold_number = int(fold_match.group(1))
        section_start = fold_match.end()
        section_end = fold_matches[idx + 1].start() if idx + 1 < len(fold_matches) else len(content)
        section = content[section_start:section_end]

        train_patients = [p.strip().strip("'\"") for p in fold_match.group(2).split(',') if p.strip()]
        val_patients = [p.strip().strip("'\"") for p in fold_match.group(3).split(',') if p.strip()]

        # 提取所有epoch数据
        epochs = []
        epoch_blocks = re.finditer(r'Epoch (\d+)/(\d+)(.*?)(?=\nEpoch \d+/\d+|\Z)', section, re.DOTALL)

        best_dice = 0
        best_epoch = 0

        for match in epoch_blocks:
            epoch_num = int(match.group(1))
            total_epochs = int(match.group(2))
            epoch_text = match.group(3)
            train_loss_match = re.search(r'Train Loss:\s*([\d.]+)', epoch_text)
            val_dice_match = re.search(r'Val Dice:\s*([\d.]+)', epoch_text)
            if not train_loss_match or not val_dice_match:
                continue

            train_loss = float(train_loss_match.group(1))
            val_dice = float(val_dice_match.group(1))

            # 提取IoU（如果有）
            iou_match = re.search(r'Val IoU:\s*([\d.]+)', epoch_text)
            val_iou = float(iou_match.group(1)) if iou_match else 0.0

            is_best = val_dice > best_dice
            if is_best:
                best_dice = val_dice
                best_epoch = epoch_num

            epochs.append({
                "epoch": epoch_num,
                "timestamp": datetime.now().isoformat(),
                "train_loss": train_loss,
                "val_dice": val_dice,
                "val_iou": val_iou,
                "is_best": is_best
            })

        if epochs:
            fold_data = {
                "fold": fold_number - 1,
                "status": "completed",
                "start_time": datetime.now().isoformat(),
                "total_epochs": epochs[-1]["epoch"] if epochs else 0,
                "current_epoch": epochs[-1]["epoch"] if epochs else 0,
                "train_patients": train_patients,
                "val_patients": val_patients,
                "epochs": epochs,
                "best_dice": best_dice,
                "best_epoch": best_epoch
            }
            all_folds.append(fold_data)

    return all_folds


def add_epoch_history_to_results(logs_dir: Path):
    """为历史结果文件添加epoch历史"""
    logs_dir = Path(logs_dir)

    # 查找所有results文件
    for result_file in logs_dir.glob("results_*.json"):
        # 读取结果文件
        with open(result_file, 'r') as f:
            data = json.load(f)

        # 检查是否已有epoch_history
        if "epoch_history" in data:
            print(f"✓ {result_file.stem} 已有epoch历史，跳过")
            continue

        # 尝试找到对应的日志文件
        split_mode = data.get("split_mode", "")

        # 尝试多个可能的日志文件
        possible_logs = []
        if "stratified" in split_mode:
            possible_logs = [
                logs_dir.parent / "train_stratified.log",
                logs_dir.parent / "train_seg.log"
            ]
        elif "lopo" in split_mode:
            possible_logs = [
                logs_dir.parent / "stable_lopo_latest.log",
                logs_dir.parent / "train_lopo.log",
                logs_dir.parent / "train_seg.log"
            ]
        else:
            possible_logs = [logs_dir.parent / "train_seg.log"]

        log_file = None
        for log_path in possible_logs:
            if log_path.exists():
                log_file = log_path
                break

        if not log_file:
            print(f"⚠ {result_file.stem} 未找到日志文件，跳过")
            continue

        try:
            # 解析日志
            print(f"📖 解析 {log_file.name} 为 {result_file.stem}...")
            epoch_history = parse_training_log(log_file)

            if not epoch_history:
                print(f"⚠ {result_file.stem} 未能从日志中提取epoch数据")
                continue

            # 添加到结果文件
            data["epoch_history"] = epoch_history
            data["epoch_history_source"] = f"parsed_from_{log_file.name}"
            data["epoch_history_added_at"] = datetime.now().isoformat()

            # 写回文件
            with open(result_file, 'w') as f:
                json.dump(data, f, indent=2)

            print(f"✓ {result_file.stem} epoch历史已添加 ({len(epoch_history)} folds)")

        except Exception as e:
            print(f"✗ {result_file.stem} 解析失败: {e}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        logs_dir = Path(sys.argv[1])
    else:
        logs_dir = Path(__file__).parent / "logs"

    print(f"扫描目录: {logs_dir}")
    add_epoch_history_to_results(logs_dir)
    print("\n完成！")
