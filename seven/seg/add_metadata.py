"""
为历史实验结果补充元数据
Add metadata to historical experiment results
"""
import json
from pathlib import Path
from datetime import datetime

# 历史实验的元数据（手动填写）
EXPERIMENT_METADATA = {
    "results_stratified_20260504_211828": {
        "name": "Stratified 4-Fold Baseline",
        "description": "完整的4折分层交叉验证，冻结编码器+adapter",
        "strategy": {
            "split_mode": "stratified_4fold",
            "num_folds": 4,
            "encoder_frozen": True,
            "use_adapter": True,
            "loss_function": "focal_tversky",
            "max_epochs": 180,
            "early_stopping_patience": 50
        },
        "notes": "最终完整版本，Mean Dice 0.4198"
    },
    "results_stratified_20260504_211238": {
        "name": "Stratified 4-Fold (Fold 0-2)",
        "description": "前3折的中间结果",
        "strategy": {
            "split_mode": "stratified_4fold",
            "num_folds": 4,
            "encoder_frozen": True,
            "use_adapter": True,
            "loss_function": "focal_tversky"
        },
        "notes": "训练中间保存的结果"
    },
    "results_stratified_20260504_210401": {
        "name": "Stratified 4-Fold (Fold 0-1)",
        "description": "前2折的中间结果",
        "strategy": {
            "split_mode": "stratified_4fold",
            "num_folds": 4,
            "encoder_frozen": True,
            "use_adapter": True,
            "loss_function": "focal_tversky"
        },
        "notes": "训练中间保存的结果"
    },
    "results_lopo_20260504_172640": {
        "name": "LOPO Test (Single Fold)",
        "description": "LOPO单折测试",
        "strategy": {
            "split_mode": "lopo",
            "encoder_frozen": True,
            "use_adapter": True,
            "loss_function": "focal_tversky"
        },
        "notes": "测试LOPO配置"
    },
    "results_lopo_20260504_171155": {
        "name": "LOPO Test (Single Fold)",
        "description": "LOPO单折测试",
        "strategy": {
            "split_mode": "lopo",
            "encoder_frozen": True,
            "use_adapter": True,
            "loss_function": "focal_tversky"
        },
        "notes": "测试LOPO配置"
    },
    "results_lopo_20260504_170513": {
        "name": "LOPO Test (Single Fold)",
        "description": "LOPO单折测试",
        "strategy": {
            "split_mode": "lopo",
            "encoder_frozen": True,
            "use_adapter": True,
            "loss_function": "focal_tversky"
        },
        "notes": "测试LOPO配置"
    },
    "results_lopo_baseline": {
        "name": "LOPO Baseline (4 patients)",
        "description": "基线实验，4个患者的LOPO交叉验证",
        "strategy": {
            "split_mode": "lopo",
            "num_folds": 4,
            "encoder_frozen": True,
            "use_adapter": False,
            "loss_function": "dice_bce",
            "max_epochs": 180
        },
        "notes": "基线对比实验，Mean Dice 0.5145"
    },
    "results_lopo_20260423_235957": {
        "name": "LOPO Full (4 patients)",
        "description": "完整的4患者LOPO实验",
        "strategy": {
            "split_mode": "lopo",
            "num_folds": 4,
            "encoder_frozen": True,
            "use_adapter": True,
            "loss_function": "focal_tversky"
        },
        "notes": "早期完整实验，Mean Dice 0.5145"
    },
    "results_lopo_20260423_233510": {
        "name": "LOPO Early Attempt",
        "description": "早期LOPO尝试",
        "strategy": {
            "split_mode": "lopo",
            "num_folds": 4,
            "encoder_frozen": True,
            "use_adapter": True,
            "loss_function": "focal_tversky"
        },
        "notes": "早期实验"
    },
    "results_lopo_20260423_231813": {
        "name": "LOPO Early Attempt",
        "description": "早期LOPO尝试",
        "strategy": {
            "split_mode": "lopo",
            "num_folds": 4,
            "encoder_frozen": True,
            "use_adapter": True,
            "loss_function": "focal_tversky"
        },
        "notes": "早期实验"
    },
    "results_lopo_20260423_222429": {
        "name": "LOPO Early Attempt",
        "description": "早期LOPO尝试",
        "strategy": {
            "split_mode": "lopo",
            "num_folds": 4,
            "encoder_frozen": True,
            "use_adapter": True,
            "loss_function": "focal_tversky"
        },
        "notes": "早期实验"
    },
    "results_lopo_20260423_184711": {
        "name": "LOPO Initial Test",
        "description": "最初的LOPO测试",
        "strategy": {
            "split_mode": "lopo",
            "num_folds": 4,
            "encoder_frozen": True,
            "use_adapter": True,
            "loss_function": "focal_tversky"
        },
        "notes": "最早期实验，Mean Dice 0.3272"
    }
}

def add_metadata_to_results(logs_dir: Path):
    """为历史结果文件添加元数据"""
    logs_dir = Path(logs_dir)

    for result_file in logs_dir.glob("results_*.json"):
        file_stem = result_file.stem

        # 读取原始结果
        with open(result_file, 'r') as f:
            data = json.load(f)

        # 检查是否已有元数据
        if "metadata" in data:
            print(f"✓ {file_stem} 已有元数据，跳过")
            continue

        # 获取元数据
        metadata = EXPERIMENT_METADATA.get(file_stem)
        if metadata:
            data["metadata"] = metadata
            data["metadata"]["added_at"] = datetime.now().isoformat()

            # 写回文件
            with open(result_file, 'w') as f:
                json.dump(data, f, indent=2)

            print(f"✓ {file_stem} 元数据已添加")
        else:
            print(f"⚠ {file_stem} 未找到元数据定义，跳过")

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        logs_dir = Path(sys.argv[1])
    else:
        logs_dir = Path(__file__).parent / "logs"

    print(f"扫描目录: {logs_dir}")
    add_metadata_to_results(logs_dir)
    print("\n完成！")
