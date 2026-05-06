from flask import Flask, render_template, jsonify, request
import json
import os
from pathlib import Path
from datetime import datetime

app = Flask(__name__)

# Paths
LOGS_DIR = Path("/root/CN_seg/seven/seg/logs")
EXPERIMENT_CONFIG = Path("/root/CN_seg/seven/seg/experiment_config.json")


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def file_updated_at(path):
    return datetime.fromtimestamp(os.path.getmtime(path)).isoformat()


def now_iso():
    return datetime.now().isoformat()


def parse_iso(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def progress_summary(data):
    folds = data.get("folds", [])
    total_epochs = sum(int(fold.get("total_epochs") or 0) for fold in folds)
    completed_epochs = sum(int(fold.get("current_epoch") or len(fold.get("epochs", []))) for fold in folds)

    if total_epochs <= 0:
        percent = 0.0
    else:
        percent = min(100.0, round(completed_epochs / total_epochs * 100, 2))

    start = parse_iso(data.get("start_time"))
    now = parse_iso(now_iso())
    elapsed_seconds = None
    eta_seconds = None
    eta_at = None

    if start and now:
        elapsed_seconds = max(0, int((now - start).total_seconds()))
        if completed_epochs > 0 and total_epochs > completed_epochs and elapsed_seconds > 0:
            seconds_per_epoch = elapsed_seconds / completed_epochs
            eta_seconds = int(round((total_epochs - completed_epochs) * seconds_per_epoch))
            eta_at = datetime.fromtimestamp(now.timestamp() + eta_seconds).isoformat()
        elif total_epochs > 0 and completed_epochs >= total_epochs:
            eta_seconds = 0
            eta_at = now.isoformat()

    return {
        "completed_epochs": completed_epochs,
        "total_epochs": total_epochs,
        "remaining_epochs": max(0, total_epochs - completed_epochs),
        "percent": percent,
        "elapsed_seconds": elapsed_seconds,
        "eta_seconds": eta_seconds,
        "eta_at": eta_at,
    }


def timestamp_from_result(result_file, data):
    filename = result_file.stem
    parts = filename.split('_')
    if len(parts) >= 3 and parts[-2].isdigit() and parts[-1].isdigit():
        date_str = parts[-2]
        time_str = parts[-1]
        return (
            f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
            f"T{time_str[:2]}:{time_str[2:4]}:{time_str[4:6]}",
            "filename",
        )

    if data.get("timestamp"):
        return data["timestamp"], "result"

    metadata = data.get("metadata", {})
    if metadata.get("experiment_time"):
        return metadata["experiment_time"], "metadata.experiment_time"

    return file_updated_at(result_file), "file_mtime"


def strategy_description(strategy, fallback):
    strategy_parts = []
    if strategy.get("split_mode"):
        strategy_parts.append(strategy["split_mode"])
    if strategy.get("num_folds"):
        strategy_parts.append(f"{strategy['num_folds']} folds")
    if strategy.get("encoder_frozen"):
        strategy_parts.append("frozen encoder")
    if strategy.get("use_adapter"):
        strategy_parts.append("adapter")
    if strategy.get("loss_function"):
        strategy_parts.append(strategy["loss_function"])
    return " · ".join(strategy_parts) if strategy_parts else fallback


def expected_fold_count(data):
    metadata = data.get("metadata", {})
    strategy = metadata.get("strategy", {})
    return strategy.get("num_folds") or data.get("num_folds")


def normalized_chart_data(folds):
    chart_data = []
    for fold in folds:
        fold_idx = fold.get("fold")
        for epoch_data in fold.get("epochs", []):
            chart_data.append({
                "fold": fold_idx,
                "epoch": epoch_data.get("epoch"),
                "train_loss": epoch_data.get("train_loss"),
                "val_dice": epoch_data.get("val_dice"),
                "val_iou": epoch_data.get("val_iou"),
                "is_best": epoch_data.get("is_best", False)
            })
    return chart_data


def filter_history_to_result_folds(epoch_history, fold_results):
    if not fold_results:
        return epoch_history
    valid_fold_indices = {fr.get("fold") for fr in fold_results}
    return [eh for eh in epoch_history if eh.get("fold") in valid_fold_indices]


def history_quality(data):
    epoch_history = data.get("epoch_history") or []
    fold_results = data.get("fold_results") or []
    if not epoch_history:
        return "none", ["No epoch history is attached."]

    notes = []
    quality = "verified"
    by_fold = {fold.get("fold"): fold for fold in epoch_history}

    for fold_result in fold_results:
        fold_idx = fold_result.get("fold")
        history_fold = by_fold.get(fold_idx)
        if not history_fold:
            quality = "mismatch"
            notes.append(f"Fold {fold_idx} has result metrics but no matching history.")
            continue

        result_val = fold_result.get("val_patients")
        history_val = history_fold.get("val_patients")
        if result_val and history_val and result_val != history_val:
            quality = "mismatch"
            notes.append(f"Fold {fold_idx} validation patients differ between result and history.")

        result_best = fold_result.get("best_dice")
        history_best = history_fold.get("best_dice")
        if result_best is not None and history_best is not None:
            if abs(result_best - history_best) > 0.01:
                quality = "mismatch"
                notes.append(f"Fold {fold_idx} best Dice differs by more than 0.01.")

    if not fold_results:
        quality = "partial"
        notes.append("No fold_results are available for cross-checking.")
    elif len(epoch_history) != len(fold_results) and quality != "mismatch":
        quality = "partial"
        notes.append("History fold count differs from result fold count.")

    has_iou = any(
        (epoch.get("val_iou") or 0) != 0
        for fold in epoch_history
        for epoch in fold.get("epochs", [])
    )
    if not has_iou:
        if quality == "verified":
            quality = "partial"
        notes.append("IoU history is missing or all zero.")

    return quality, notes


def result_summary(result_file, data):
    timestamp, timestamp_source = timestamp_from_result(result_file, data)
    metadata = data.get("metadata", {})
    strategy = metadata.get("strategy", {})
    fold_results = data.get("fold_results", [])
    expected_folds = expected_fold_count(data)
    quality, quality_notes = history_quality(data)

    return {
        "id": result_file.stem,
        "experiment_id": data.get("experiment_id"),
        "name": metadata.get("name", result_file.stem),
        "description": metadata.get("description", ""),
        "status": "completed",
        "timestamp": timestamp,
        "timestamp_source": timestamp_source,
        "updated_at": file_updated_at(result_file),
        "split_mode": data.get("split_mode", "unknown"),
        "strategy": strategy_description(strategy, data.get("split_mode", "unknown")),
        "mean_dice": data.get("mean_dice", 0),
        "num_folds": len(fold_results),
        "expected_folds": expected_folds,
        "is_partial": bool(expected_folds and len(fold_results) < expected_folds),
        "has_epoch_history": bool(data.get("epoch_history")),
        "history_source": data.get("epoch_history_source", "progress_tracker"),
        "history_quality": quality,
        "history_quality_notes": quality_notes,
        "file": str(result_file.name)
    }


def build_experiments():
    experiments = []
    completed_experiment_ids = set()

    # 1. 已完成的实验（results_*.json）
    for result_file in LOGS_DIR.glob("results_*.json"):
        data = load_json(result_file)
        if data.get("experiment_id"):
            completed_experiment_ids.add(data["experiment_id"])
        experiments.append(result_summary(result_file, data))

    # 2. 正在运行的实验（progress_*.json）
    for progress_file in LOGS_DIR.glob("progress_*.json"):
        data = load_json(progress_file)
        if data.get("experiment_id") in completed_experiment_ids:
            continue

        experiments.append({
            "id": data.get("experiment_id", progress_file.stem),
            "name": data.get("experiment_id", progress_file.stem),
            "description": "",
            "status": data.get("status", "running"),
            "timestamp": data.get("start_time", ""),
            "timestamp_source": "progress.start_time",
            "updated_at": file_updated_at(progress_file),
            "split_mode": "unknown",
            "strategy": "",
            "current_fold": data.get("current_fold"),
            "num_folds": len(data.get("folds", [])),
            "expected_folds": data.get("total_folds"),
            "is_partial": False,
            "has_epoch_history": True,
            "history_source": "progress_tracker",
            "progress": progress_summary(data),
            "file": str(progress_file.name)
        })

    # 按时间倒序排序
    experiments.sort(key=lambda x: x["timestamp"], reverse=True)
    return experiments

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/experiments')
def list_experiments():
    """列出所有实验（已完成的results + 正在运行的progress）"""
    try:
        return jsonify(build_experiments())

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/training_history')
def legacy_training_history():
    """兼容旧监控入口：返回归一化后的实验历史列表。"""
    try:
        return jsonify(build_experiments())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/training_status')
def legacy_training_status():
    """兼容旧监控入口：报告当前是否有运行中的实验。"""
    try:
        experiments = build_experiments()
        running = [exp for exp in experiments if exp.get("status") == "running"]
        return jsonify({
            "status": "running" if running else "idle",
            "running_experiments": running,
            "latest_experiment": experiments[0] if experiments else None,
            "total_experiments": len(experiments),
            "updated_at": datetime.now().isoformat(),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/experiment/<experiment_id>')
def get_experiment(experiment_id):
    """获取单个实验的详细信息"""
    try:
        # 先尝试progress文件（正在运行）
        progress_file = LOGS_DIR / f"progress_{experiment_id}.json"
        if progress_file.exists():
            data = load_json(progress_file)
            return jsonify(data)

        # 再尝试results文件（已完成）
        result_file = LOGS_DIR / f"{experiment_id}.json"
        if result_file.exists():
            data = load_json(result_file)
            quality, quality_notes = history_quality(data)
            data["history_quality"] = quality
            data["history_quality_notes"] = quality_notes
            return jsonify(data)

        return jsonify({"error": "Experiment not found"}), 404

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/experiment/<experiment_id>/realtime')
def get_realtime_progress(experiment_id):
    """获取实验的实时进度（用于图表）"""
    try:
        # 先尝试progress文件（运行中）
        progress_file = LOGS_DIR / f"progress_{experiment_id}.json"
        if progress_file.exists():
            data = load_json(progress_file)

            # 提取所有fold的epoch数据用于绘图
            chart_data = normalized_chart_data(data.get("folds", []))

            return jsonify({
                "status": data.get("status"),
                "current_fold": data.get("current_fold"),
                "last_update": data.get("last_update"),
                "folds": data.get("folds", []),
                "history_source": "progress_tracker",
                "progress": progress_summary(data),
                "chart_data": chart_data
            })

        # 尝试results文件（已完成，但包含epoch_history）
        result_file = LOGS_DIR / f"{experiment_id}.json"
        if result_file.exists():
            data = load_json(result_file)

            # 检查是否有epoch_history
            epoch_history = data.get("epoch_history", [])
            fold_results = data.get("fold_results", [])
            epoch_history = filter_history_to_result_folds(epoch_history, fold_results)
            quality, quality_notes = history_quality(data)
            chart_data = [] if quality == "mismatch" else normalized_chart_data(epoch_history)
            timestamp, _ = timestamp_from_result(result_file, data)

            return jsonify({
                "status": "completed",
                "current_fold": None,
                "last_update": timestamp,
                "updated_at": file_updated_at(result_file),
                "folds": epoch_history,
                "fold_results": fold_results,
                "history_source": data.get("epoch_history_source", "progress_tracker"),
                "history_quality": quality,
                "history_quality_notes": quality_notes,
                "chart_data": chart_data
            })

        return jsonify({"error": "Experiment not found"}), 404

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/config')
def get_experiment_config():
    """获取实验配置模板"""
    try:
        if EXPERIMENT_CONFIG.exists():
            return jsonify(load_json(EXPERIMENT_CONFIG))
        return jsonify({"error": "Config not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6006, debug=False)
