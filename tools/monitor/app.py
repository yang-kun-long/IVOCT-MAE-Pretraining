from flask import Flask, render_template, jsonify, request
import json
import os
from pathlib import Path
from datetime import datetime
from copy import deepcopy

app = Flask(__name__)

# Paths
LOGS_DIR = Path("/root/CN_seg/seven/seg/logs")
EXPERIMENT_CONFIG = Path("/root/CN_seg/seven/seg/experiment_config.json")

CORE4_PATIENTS = ["P014", "P008", "P013", "P006"]
EXTRA4_PATIENTS = ["P015", "P010", "P016", "P009"]
FULL_REMAINING10_PATIENTS = ["P001", "P002", "P003", "P004", "P005", "P007", "P012", "P017", "P018", "P019"]
EXPANDED8_PATIENTS = CORE4_PATIENTS + EXTRA4_PATIENTS
FULL18_PATIENTS = EXPANDED8_PATIENTS + FULL_REMAINING10_PATIENTS


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


def first_val_patient(fold):
    patients = fold.get("val_patients") or []
    return patients[0] if patients else None


def result_pattern_for_split(split_mode):
    return f"results_{split_mode}_*.json"


def canonical_experiment_key(data):
    key = data.get("split_mode") or data.get("experiment_id") or ""
    parts = key.split("_")
    if len(parts) >= 3 and parts[-2].isdigit() and parts[-1].isdigit():
        return "_".join(parts[:-2])
    return key


def method_label(key):
    key = key.lower()
    if "smp_linknet_resnet34_imagenet" in key:
        return "SMP LinkNet-R34"
    if "smp_unetplusplus_resnet34_imagenet" in key:
        return "SMP U-Net++-R34"
    if "smp_deeplabv3plus_resnet34_imagenet" in key:
        return "SMP DeepLabV3+-R34"
    if "dinov2_linknet" in key:
        return "DINOv2 + LinkNet"
    if "dinov2" in key:
        return "DINOv2 ViT-S"
    if "lipid_warm" in key:
        return "Lipid-Warm Skip"
    if "v13_two_stage" in key:
        return "v13 Detect+Seg (已弃)"
    if "v7_skip" in key:
        return "v7 Skip"
    if "v8_residual_skip" in key:
        return "v8 Residual Skip"
    if "v9_warm_residual_skip" in key:
        return "v9 Warm Residual"
    return "IVOCT 分割"


def panel_label(data, key):
    panel = data.get("panel") or {}
    if panel.get("stage") == "expanded8":
        return "Expanded-8"
    if panel.get("stage") == "full_lopo18":
        return "Full LOPO18"
    if "_full_remaining10" in key:
        return "Full 剩余10"
    if "_extra4" in key:
        return "Extra-4"
    if "_sentinel" in key:
        return "Core-4"
    if "_lopo_" in key or "lopo" in key:
        return "LOPO"
    if "4fold" in key:
        return "4-Fold"
    return ""


def validation_patients_text(data):
    panel = data.get("panel") or {}
    patients = panel.get("expected_patients")
    if not patients:
        key = canonical_experiment_key(data).lower()
        # For _sentinel single-worker progress JSONs (each worker holds only
        # one val patient), surface the full Core-4 panel so the strategy
        # text matches what completed _sentinel result files display.
        if "_sentinel" in key:
            patients = CORE4_PATIENTS
        else:
            patients = [
                patient
                for fold in data.get("fold_results") or data.get("folds") or []
                for patient in (fold.get("val_patients") or [])
            ]
    return "/".join(patients[:8]) + ("..." if len(patients) > 8 else "")


def metadata_remark(metadata):
    for key in ("remark", "remarks", "note", "notes", "comment", "comments"):
        value = metadata.get(key)
        if isinstance(value, list):
            value = "；".join(str(item) for item in value if item)
        if value:
            return str(value)
    return ""


def generated_remark(data):
    key = canonical_experiment_key(data).lower()
    panel = data.get("panel") or {}
    copied = panel.get("copied_folds", 0)
    missing = panel.get("missing_sources") or []
    if panel.get("stage") == "expanded8":
        suffix = "；缺少历史来源，需检查文件匹配。" if missing else ""
        return f"分层续跑视图：复用 Core-4 的 {copied} 折，只新增 Extra-4 病人。{suffix}"
    if panel.get("stage") == "full_lopo18":
        suffix = "；缺少历史来源，需检查文件匹配。" if missing else ""
        return f"分层完整视图：复用 Core-4/Extra-4 的 {copied} 折，只新增剩余 10 个病人。{suffix}"
    if "smp_" in key:
        return "第三方成熟分割架构基线，用来判断架构和训练 recipe 的差距。"
    if "dinov2" in key:
        return "外部预训练 attribution 实验，用来判断表示学习是否是主要瓶颈。"
    if "lipid_warm" in key:
        return "自研：脂质 mask 监督预训练 → 钙化分割 fine-tune，单变量对照 v7 skip baseline。"
    if "v13_two_stage" in key:
        return "自研：joint detect+seg + 负样本（已证伪：Core-4 退步 -3.7pp）。"
    if "v7_skip" in key:
        return "当前自研主线参考结果，用作第三方方法和新方案的对照。"
    return ""


def presentation_fields(data, fallback_name):
    metadata = data.get("metadata") or {}
    key = canonical_experiment_key(data)
    panel = data.get("panel") or {}
    if panel.get("name"):
        name = f"{method_label(key)} · {panel['name']}"
    elif metadata.get("display_name"):
        name = metadata["display_name"]
    else:
        panel_name = panel_label(data, key)
        method = method_label(key)
        name = f"{method} · {panel_name}" if panel_name else method

    strategy = None
    if panel.get("name"):
        patients = validation_patients_text(data)
        strategy_parts = []
        if "smp_" in key.lower():
            strategy_parts.append("成熟分割架构")
        if "dinov2" in key.lower():
            strategy_parts.append("外部预训练")
        if "imagenet" in key.lower():
            strategy_parts.append("ImageNet 编码器")
        strategy_parts.append(f"验证面板: {panel['name']}")
        if patients:
            strategy_parts.append(f"病人: {patients}")
        strategy = "；".join(strategy_parts)
    else:
        strategy = metadata.get("strategy_cn")

    if not strategy:
        panel_name = panel_label(data, key)
        patients = validation_patients_text(data)
        strategy_parts = []
        if "smp_" in key.lower():
            strategy_parts.append("成熟分割架构")
        if "dinov2" in key.lower():
            strategy_parts.append("外部预训练")
        if "imagenet" in key.lower():
            strategy_parts.append("ImageNet 编码器")
        if panel_name:
            strategy_parts.append(f"验证面板: {panel_name}")
        if patients:
            strategy_parts.append(f"病人: {patients}")
        strategy = "；".join(strategy_parts) if strategy_parts else strategy_description(metadata.get("strategy", {}), data.get("split_mode", fallback_name))

    remark = metadata_remark(metadata) or generated_remark(data)
    description = metadata.get("description_cn") or metadata.get("description") or remark
    return {
        "name": name or fallback_name,
        "description": description or "",
        "strategy": strategy or "",
        "remark": remark or "",
        "source_id": data.get("experiment_id") or fallback_name,
    }


def latest_result(pattern):
    files = [path for path in LOGS_DIR.glob(pattern) if path.is_file()]
    if not files:
        return None
    return max(files, key=lambda path: path.stat().st_mtime)


def composite_config(data):
    """Infer progressive-panel composition from stable experiment naming."""
    key = canonical_experiment_key(data)
    if "_extra4" in key:
        source_split = key.replace("_extra4", "_sentinel")
        return {
            "stage": "expanded8",
            "name": "Expanded-8",
            "expected_patients": EXPANDED8_PATIENTS,
            "sources": [{"label": "Core-4", "pattern": result_pattern_for_split(source_split)}],
        }
    if "_full_remaining10" in key:
        base = key.replace("_full_remaining10", "")
        return {
            "stage": "full_lopo18",
            "name": "Full LOPO18",
            "expected_patients": FULL18_PATIENTS,
            "sources": [
                {"label": "Core-4", "pattern": result_pattern_for_split(f"{base}_sentinel")},
                {"label": "Extra-4", "pattern": result_pattern_for_split(f"{base}_extra4")},
            ],
        }
    return None


def fold_history_from_results(fold_results):
    history = []
    for row in fold_results:
        final_epoch = int(row.get("final_epoch") or row.get("best_epoch") or 0)
        history.append({
            "fold": row.get("fold"),
            "status": "completed",
            "total_epochs": final_epoch,
            "current_epoch": final_epoch,
            "train_patients": row.get("train_patients", []),
            "val_patients": row.get("val_patients", []),
            "epochs": [],
            "best_dice": row.get("best_dice", 0.0),
            "best_epoch": row.get("best_epoch", 0),
        })
    return history


def normalize_panel_fold(fold, expected_patients, label=None, copied=False):
    row = deepcopy(fold)
    patient = first_val_patient(row)
    if patient in expected_patients:
        row["fold"] = expected_patients.index(patient)
    if copied:
        row["copied_from"] = label or "prior panel"
        row.setdefault("status", "completed")
    return row


def merge_panel_folds(groups, expected_patients):
    by_patient = {}
    for folds, label, copied in groups:
        for fold in folds:
            patient = first_val_patient(fold)
            if not patient:
                continue
            by_patient[patient] = normalize_panel_fold(fold, expected_patients, label=label, copied=copied)
    return [by_patient[p] for p in expected_patients if p in by_patient]


def load_source_result(source):
    path = None
    if source.get("file"):
        path = LOGS_DIR / source["file"]
    elif source.get("pattern"):
        path = latest_result(source["pattern"])
    if not path or not path.exists():
        return None, None
    return path, load_json(path)


def apply_composite_panel(data):
    config = composite_config(data)
    if not config:
        return data

    expected_patients = config["expected_patients"]
    source_result_groups = []
    source_history_groups = []
    source_files = []
    missing_sources = []

    for source in config["sources"]:
        source_path, source_data = load_source_result(source)
        if source_data is None:
            missing_sources.append(source.get("pattern") or source.get("file") or "unknown")
            continue
        label = source.get("label")
        source_files.append(source_path.name)
        source_result_groups.append((source_data.get("fold_results", []), label, True))
        source_history = source_data.get("epoch_history") or fold_history_from_results(source_data.get("fold_results", []))
        source_history_groups.append((source_history, label, True))

    current_results = data.get("fold_results") or data.get("all_results") or []
    current_history = data.get("epoch_history")
    if current_history is None:
        current_history = data.get("folds") or fold_history_from_results(current_results)

    combined_results = merge_panel_folds(
        source_result_groups + [(current_results, None, False)],
        expected_patients,
    )
    combined_history = merge_panel_folds(
        source_history_groups + [(current_history, None, False)],
        expected_patients,
    )

    effective = deepcopy(data)
    effective["fold_results"] = combined_results
    effective["epoch_history"] = combined_history
    effective["folds"] = combined_history
    effective["num_folds"] = len(expected_patients)
    effective["total_folds"] = len(expected_patients)
    effective["panel"] = {
        "stage": config["stage"],
        "name": config["name"],
        "expected_patients": expected_patients,
        "source_files": source_files,
        "missing_sources": missing_sources,
        "copied_folds": sum(1 for fold in combined_results if fold.get("copied_from")),
    }
    effective["is_composite_panel"] = True
    if combined_results:
        dices = [float(row.get("best_dice", 0.0)) for row in combined_results]
        effective["mean_dice"] = sum(dices) / len(dices)
        effective["std_dice"] = float((sum((x - effective["mean_dice"]) ** 2 for x in dices) / len(dices)) ** 0.5)

    metadata = effective.setdefault("metadata", {})
    metadata.setdefault("display_name", f"{method_label(canonical_experiment_key(effective))} · {config['name']}")
    metadata.setdefault("description_cn", generated_remark(effective))
    metadata["composite_panel"] = effective["panel"]
    return effective


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
    if data.get("panel", {}).get("expected_patients"):
        return len(data["panel"]["expected_patients"])
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
    data = apply_composite_panel(data)
    timestamp, timestamp_source = timestamp_from_result(result_file, data)
    metadata = data.get("metadata", {})
    strategy = metadata.get("strategy", {})
    fold_results = data.get("fold_results", [])
    expected_folds = expected_fold_count(data)
    quality, quality_notes = history_quality(data)
    presentation = presentation_fields(data, result_file.stem)

    return {
        "id": result_file.stem,
        "experiment_id": data.get("experiment_id"),
        "batch_id": data.get("batch_id"),
        "name": presentation["name"],
        "description": presentation["description"],
        "remark": presentation["remark"],
        "source_id": presentation["source_id"],
        "status": "completed",
        "timestamp": timestamp,
        "timestamp_source": timestamp_source,
        "updated_at": file_updated_at(result_file),
        "split_mode": data.get("split_mode", "unknown"),
        "strategy": presentation["strategy"] or strategy_description(strategy, data.get("split_mode", "unknown")),
        "mean_dice": data.get("mean_dice", 0),
        "num_folds": len(fold_results),
        "expected_folds": expected_folds,
        "is_partial": bool(expected_folds and len(fold_results) < expected_folds),
        "is_composite_panel": bool(data.get("is_composite_panel")),
        "panel": data.get("panel"),
        "has_epoch_history": bool(data.get("epoch_history")),
        "history_source": data.get("epoch_history_source", "progress_tracker"),
        "history_quality": quality,
        "history_quality_notes": quality_notes,
        "file": str(result_file.name)
    }


def is_monitor_hidden(data):
    """Return True when a result/progress file should stay out of the UI."""
    metadata = data.get("metadata") or {}
    visibility = data.get("visibility", metadata.get("visibility"))
    monitor_visible = data.get("monitor_visible", metadata.get("monitor_visible"))
    if monitor_visible is False:
        return True
    if visibility in {"hidden", "debug", "diagnostic", "archive"}:
        return True
    if data.get("status") in {"stopped", "abandoned"} and monitor_visible is not True:
        return True
    return False


def build_experiments():
    experiments = []
    completed_experiment_ids = set()

    # 1. 已完成的实验（results_*.json）
    for result_file in LOGS_DIR.glob("results_*.json"):
        data = load_json(result_file)
        if is_monitor_hidden(data):
            continue
        if data.get("experiment_id"):
            completed_experiment_ids.add(data["experiment_id"])
        experiments.append(result_summary(result_file, data))

    # 2. 正在运行的实验（progress_*.json）
    for progress_file in LOGS_DIR.glob("progress_*.json"):
        data = load_json(progress_file)
        if is_monitor_hidden(data):
            continue
        if data.get("experiment_id") in completed_experiment_ids:
            continue
        data = apply_composite_panel(data)
        presentation = presentation_fields(data, data.get("experiment_id", progress_file.stem))

        experiments.append({
            "id": data.get("experiment_id", progress_file.stem),
            "batch_id": data.get("batch_id"),
            "name": presentation["name"],
            "description": presentation["description"],
            "remark": presentation["remark"],
            "source_id": presentation["source_id"],
            "status": data.get("status", "running"),
            "timestamp": data.get("start_time", ""),
            "timestamp_source": "progress.start_time",
            "updated_at": file_updated_at(progress_file),
            "split_mode": "unknown",
            "strategy": presentation["strategy"],
            "current_fold": data.get("current_fold"),
            "num_folds": len(data.get("folds", [])),
            "expected_folds": expected_fold_count(data) or data.get("total_folds"),
            "is_partial": bool((expected_fold_count(data) or 0) and len(data.get("folds", [])) < expected_fold_count(data)),
            "is_composite_panel": bool(data.get("is_composite_panel")),
            "panel": data.get("panel"),
            "has_epoch_history": True,
            "history_source": "progress_tracker",
            "progress": progress_summary(data),
            "file": str(progress_file.name)
        })

    # 按时间倒序排序
    experiments.sort(key=lambda x: x["timestamp"], reverse=True)
    return experiments


def group_experiments_by_batch(experiments):
    """Collect experiments sharing the same `batch_id` into virtual batch nodes.

    A `batch_id` is set by training scripts (via MonitorRun batch_id arg or the
    BATCH_ID env var) so a shell launcher that fans out N parallel single-fold
    workers can declare them as one logical batch. The monitor surfaces:
      - one summary card per batch (aggregated mean_dice, status, child count)
      - the individual member experiments inside the batch node
      - any experiment without batch_id stays at the top level unchanged

    Returns a dict:
      {"batches": [...batch nodes...], "ungrouped": [...standalone experiments...]}
    Each batch node has fields:
      is_batch, batch_id, name, status, num_members, member_status_counts,
      mean_dice_completed, timestamp (earliest member start),
      updated_at (latest member update), members (sorted by fold/timestamp).
    """
    batches = {}
    ungrouped = []
    for exp in experiments:
        batch_id = exp.get("batch_id")
        if not batch_id:
            ungrouped.append(exp)
            continue
        node = batches.setdefault(batch_id, {
            "is_batch": True,
            "batch_id": batch_id,
            "name": None,
            "members": [],
            "timestamps": [],
            "updated_ats": [],
            "status_counts": {"running": 0, "completed": 0, "error": 0, "other": 0},
            "completed_dices": [],
        })
        node["members"].append(exp)
        if exp.get("timestamp"):
            node["timestamps"].append(exp["timestamp"])
        if exp.get("updated_at"):
            node["updated_ats"].append(exp["updated_at"])
        status = exp.get("status", "other")
        if status in node["status_counts"]:
            node["status_counts"][status] += 1
        else:
            node["status_counts"]["other"] += 1
        if status == "completed" and exp.get("mean_dice") is not None:
            try:
                node["completed_dices"].append(float(exp["mean_dice"]))
            except (TypeError, ValueError):
                pass

    batch_nodes = []
    for batch_id, node in batches.items():
        members = node["members"]
        members.sort(key=lambda m: (m.get("timestamp") or "", m.get("id") or ""))
        first_member = members[0]
        node["name"] = first_member.get("name", batch_id)
        node["timestamp"] = min(node["timestamps"]) if node["timestamps"] else (first_member.get("timestamp") or "")
        node["updated_at"] = max(node["updated_ats"]) if node["updated_ats"] else first_member.get("updated_at", "")
        node["num_members"] = len(members)
        node["status"] = "running" if node["status_counts"]["running"] > 0 else \
                         ("completed" if node["status_counts"]["completed"] == len(members) else "partial")
        node["mean_dice_completed"] = (
            sum(node["completed_dices"]) / len(node["completed_dices"])
            if node["completed_dices"] else None
        )
        node["completed_count"] = node["status_counts"]["completed"]
        node["running_count"] = node["status_counts"]["running"]
        node["expected_members"] = first_member.get("expected_folds") or None
        # Cleanup helper fields
        node.pop("timestamps", None)
        node.pop("updated_ats", None)
        node.pop("completed_dices", None)
        batch_nodes.append(node)

    batch_nodes.sort(key=lambda b: b["timestamp"] or "", reverse=True)
    return {"batches": batch_nodes, "ungrouped": ungrouped}


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


@app.route('/api/batches')
def list_batches():
    """返回按 batch_id 分组的实验视图。

    每个 batch 节点聚合 N 个并发 worker 的状态/Dice/时间戳；无 batch_id
    的实验通过 `ungrouped` 字段返回。前端可据此渲染折叠的批次卡片。
    """
    try:
        return jsonify(group_experiments_by_batch(build_experiments()))
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
            data = apply_composite_panel(data)
            data.update(presentation_fields(data, experiment_id))
            return jsonify(data)

        # 再尝试results文件（已完成）
        result_file = LOGS_DIR / f"{experiment_id}.json"
        if result_file.exists():
            data = load_json(result_file)
            data = apply_composite_panel(data)
            data.update(presentation_fields(data, experiment_id))
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
            data = apply_composite_panel(data)
            presentation = presentation_fields(data, experiment_id)

            # 提取所有fold的epoch数据用于绘图
            chart_data = normalized_chart_data(data.get("folds", []))

            return jsonify({
                "name": presentation["name"],
                "description": presentation["description"],
                "strategy": presentation["strategy"],
                "remark": presentation["remark"],
                "source_id": presentation["source_id"],
                "status": data.get("status"),
                "current_fold": data.get("current_fold"),
                "last_update": data.get("last_update"),
                "folds": data.get("folds", []),
                "fold_results": data.get("fold_results", []),
                "is_composite_panel": bool(data.get("is_composite_panel")),
                "panel": data.get("panel"),
                "history_source": "progress_tracker",
                "progress": progress_summary(data),
                "chart_data": chart_data
            })

        # 尝试results文件（已完成，但包含epoch_history）
        result_file = LOGS_DIR / f"{experiment_id}.json"
        if result_file.exists():
            data = load_json(result_file)
            data = apply_composite_panel(data)
            presentation = presentation_fields(data, experiment_id)

            # 检查是否有epoch_history
            epoch_history = data.get("epoch_history", [])
            fold_results = data.get("fold_results", [])
            epoch_history = filter_history_to_result_folds(epoch_history, fold_results)
            quality, quality_notes = history_quality(data)
            chart_data = [] if quality == "mismatch" else normalized_chart_data(epoch_history)
            timestamp, _ = timestamp_from_result(result_file, data)

            return jsonify({
                "name": presentation["name"],
                "description": presentation["description"],
                "strategy": presentation["strategy"],
                "remark": presentation["remark"],
                "source_id": presentation["source_id"],
                "status": "completed",
                "current_fold": None,
                "last_update": timestamp,
                "updated_at": file_updated_at(result_file),
                "folds": epoch_history,
                "fold_results": fold_results,
                "is_composite_panel": bool(data.get("is_composite_panel")),
                "panel": data.get("panel"),
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
