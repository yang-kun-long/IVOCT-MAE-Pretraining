"""
Real-time training progress tracker
实时训练进度追踪器

在训练过程中调用，实时写入epoch级别的进度数据
"""
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import os

try:
    import fcntl
except ImportError:  # Windows fallback: atomic replace still protects readers from partial writes.
    fcntl = None


def _flock(file_obj, operation):
    if fcntl is not None:
        fcntl.flock(file_obj.fileno(), operation)


class ProgressTracker:
    """实时进度追踪器"""

    def __init__(self, experiment_id: str, logs_dir: Path):
        self.experiment_id = experiment_id
        self.logs_dir = Path(logs_dir)
        self.progress_file = self.logs_dir / f"progress_{experiment_id}.json"
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # 初始化进度文件
        if not self.progress_file.exists():
            self._init_progress_file()

    def _init_progress_file(self):
        """初始化进度文件"""
        data = {
            "experiment_id": self.experiment_id,
            "status": "running",
            "start_time": datetime.now().isoformat(),
            "last_update": datetime.now().isoformat(),
            "current_fold": None,
            "folds": []
        }
        self._write_json(data)

    def plan_folds(self, folds: list):
        """预先登记所有计划中的fold，用于总体进度和ETA计算"""
        data = self._read_json()
        data["last_update"] = datetime.now().isoformat()
        data["folds"] = []
        for fold in folds:
            data["folds"].append({
                "fold": fold["fold"],
                "status": "pending",
                "start_time": None,
                "total_epochs": fold["total_epochs"],
                "current_epoch": 0,
                "train_patients": fold["train_patients"],
                "val_patients": fold["val_patients"],
                "epochs": [],
                "best_dice": 0.0,
                "best_epoch": 0
            })
        self._write_json(data)

    def start_fold(self, fold_idx: int, total_epochs: int,
                   train_patients: list, val_patients: list):
        """开始新的fold"""
        data = self._read_json()
        data["current_fold"] = fold_idx
        data["last_update"] = datetime.now().isoformat()

        fold_data = None
        for f in data["folds"]:
            if f["fold"] == fold_idx:
                fold_data = f
                break

        if fold_data is None:
            fold_data = {
                "fold": fold_idx,
                "epochs": [],
                "best_dice": 0.0,
                "best_epoch": 0
            }
            data["folds"].append(fold_data)

        fold_data.update({
            "status": "running",
            "start_time": datetime.now().isoformat(),
            "total_epochs": total_epochs,
            "current_epoch": 0,
            "train_patients": train_patients,
            "val_patients": val_patients,
        })
        fold_data.setdefault("epochs", [])
        fold_data.setdefault("best_dice", 0.0)
        fold_data.setdefault("best_epoch", 0)
        self._write_json(data)

    def update_epoch(self, fold_idx: int, epoch: int,
                     train_loss: float, val_dice: float, val_iou: float,
                     is_best: bool = False):
        """更新epoch进度"""
        data = self._read_json()
        data["last_update"] = datetime.now().isoformat()

        # 找到对应的fold
        fold_data = None
        for f in data["folds"]:
            if f["fold"] == fold_idx:
                fold_data = f
                break

        if fold_data is None:
            return

        fold_data["current_epoch"] = epoch
        epoch_data = {
            "epoch": epoch,
            "timestamp": datetime.now().isoformat(),
            "train_loss": train_loss,
            "val_dice": val_dice,
            "val_iou": val_iou,
            "is_best": is_best
        }
        fold_data["epochs"].append(epoch_data)

        if is_best:
            fold_data["best_dice"] = val_dice
            fold_data["best_epoch"] = epoch

        self._write_json(data)

    def finish_fold(self, fold_idx: int, best_dice: float, metrics: Dict[str, float]):
        """完成fold"""
        data = self._read_json()
        data["last_update"] = datetime.now().isoformat()

        for f in data["folds"]:
            if f["fold"] == fold_idx:
                f["status"] = "completed"
                f["end_time"] = datetime.now().isoformat()
                f["best_dice"] = best_dice
                f["final_metrics"] = metrics
                break

        self._write_json(data)

    def finish_experiment(self, mean_dice: float, all_results: list):
        """完成整个实验"""
        data = self._read_json()
        data["status"] = "completed"
        data["end_time"] = datetime.now().isoformat()
        data["last_update"] = datetime.now().isoformat()
        data["mean_dice"] = mean_dice
        data["all_results"] = all_results
        self._write_json(data)

    def mark_error(self, error_msg: str):
        """标记错误"""
        data = self._read_json()
        data["status"] = "error"
        data["error"] = error_msg
        data["last_update"] = datetime.now().isoformat()
        self._write_json(data)

    def _read_json(self) -> Dict[str, Any]:
        """线程安全地读取JSON"""
        if not self.progress_file.exists():
            return {}

        with open(self.progress_file, 'r', encoding='utf-8') as f:
            _flock(f, fcntl.LOCK_SH if fcntl is not None else None)
            try:
                data = json.load(f)
            finally:
                _flock(f, fcntl.LOCK_UN if fcntl is not None else None)
        return data

    def _write_json(self, data: Dict[str, Any]):
        """线程安全地写入JSON"""
        tmp_file = self.progress_file.with_suffix(self.progress_file.suffix + ".tmp")
        with open(tmp_file, 'w', encoding='utf-8') as f:
            _flock(f, fcntl.LOCK_EX if fcntl is not None else None)
            try:
                json.dump(data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            finally:
                _flock(f, fcntl.LOCK_UN if fcntl is not None else None)
        os.replace(tmp_file, self.progress_file)
