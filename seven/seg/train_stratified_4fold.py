"""
Stratified 4-fold cross-validation training
按前景比例分层的4折交叉验证
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(1, str(Path(__file__).parent.parent))

import config_seg as config
from train_seg import train_fold
from utils.progress_tracker import ProgressTracker
import json
from datetime import datetime
import torch

# Stratified 4-fold split (from analysis)
STRATIFIED_FOLDS = [
    {
        "fold": 0,
        "val_patients": ["P014", "P015", "P010", "P011"],
        "train_patients": ["P008", "P013", "P009", "P001", "P012", "P017",
                          "P005", "P019", "P007", "P018", "P016", "P002",
                          "P003", "P004", "P006"]
    },
    {
        "fold": 1,
        "val_patients": ["P008", "P013", "P009", "P001"],
        "train_patients": ["P014", "P015", "P010", "P011", "P012", "P017",
                          "P005", "P019", "P007", "P018", "P016", "P002",
                          "P003", "P004", "P006"]
    },
    {
        "fold": 2,
        "val_patients": ["P012", "P017", "P005", "P019"],
        "train_patients": ["P014", "P015", "P010", "P011", "P008", "P013",
                          "P009", "P001", "P007", "P018", "P016", "P002",
                          "P003", "P004", "P006"]
    },
    {
        "fold": 3,
        "val_patients": ["P007", "P018", "P016", "P002", "P003", "P004", "P006"],
        "train_patients": ["P014", "P015", "P010", "P011", "P008", "P013",
                          "P009", "P001", "P012", "P017", "P005", "P019"]
    }
]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("="*80)
    print("Stratified 4-Fold Cross-Validation Training")
    print("="*80)

    # Initialize progress tracker
    experiment_id = f"stratified_4fold_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    tracker = ProgressTracker(
        experiment_id=experiment_id,
        logs_dir=config.SEG_LOG_DIR
    )
    print(f"Experiment ID: {experiment_id}")
    print(f"Progress tracking enabled: {config.SEG_LOG_DIR / f'progress_{experiment_id}.json'}")

    # Fold 0 already completed with best_dice=0.3227
    all_results = [
        {
            "fold": 0,
            "train_patients": ["P008", "P013", "P009", "P001", "P012", "P017",
                              "P005", "P019", "P007", "P018", "P016", "P002",
                              "P003", "P004", "P006"],
            "val_patients": ["P014", "P015", "P010", "P011"],
            "best_dice": 0.3227,
            "metrics": {
                "dice_mean": 0.3227,
                "dice_std": 0.0,
                "iou_mean": 0.0,
                "iou_std": 0.0,
                "sensitivity_mean": 0.0,
                "sensitivity_std": 0.0,
                "specificity_mean": 0.0,
                "specificity_std": 0.0
            }
        }
    ]
    print("✓ Fold 0 already completed (Dice=0.3227), continuing from Fold 1...")

    try:
        for fold_info in STRATIFIED_FOLDS[1:]:  # Skip Fold 0
            fold_idx = fold_info["fold"]
            val_patients = fold_info["val_patients"]
            train_patients = fold_info["train_patients"]

            print(f"\n{'='*80}")
            print(f"Training Fold {fold_idx}")
            print(f"Val patients: {val_patients}")
            print(f"Train patients ({len(train_patients)}): {train_patients}")
            print(f"{'='*80}\n")

            # Train this fold with progress tracking
            best_dice, best_metrics = train_fold(
                fold_idx=fold_idx,
                train_patients=train_patients,
                val_patients=val_patients,
                device=device,
                progress_tracker=tracker
            )

            fold_result = {
                "fold": fold_idx,
                "train_patients": train_patients,
                "val_patients": val_patients,
                "best_dice": best_dice,
                "metrics": best_metrics
            }

            all_results.append(fold_result)

            # Save intermediate results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_dict = {
                "split_mode": "stratified_4fold",
                "timestamp": timestamp,
                "mean_dice": sum(r["best_dice"] for r in all_results) / len(all_results),
                "fold_results": all_results
            }

            output_file = config.SEG_LOG_DIR / f"results_stratified_{timestamp}.json"
            with open(output_file, 'w') as f:
                json.dump(results_dict, f, indent=2)

            print(f"\n✓ Fold {fold_idx} completed. Intermediate results saved to {output_file}")

        # Final summary
        print("\n" + "="*80)
        print("Stratified 4-Fold Training Completed!")
        print("="*80)

        mean_dice = sum(r["best_dice"] for r in all_results) / len(all_results)
        print(f"\nMean Dice across all folds: {mean_dice:.4f}")

        for result in all_results:
            print(f"  Fold {result['fold']}: {result['best_dice']:.4f}")

        # Mark experiment as completed
        tracker.finish_experiment(mean_dice=mean_dice, all_results=all_results)

        # Copy epoch history from progress file to final results
        progress_file = config.SEG_LOG_DIR / f"progress_{experiment_id}.json"
        if progress_file.exists():
            with open(progress_file, 'r') as f:
                progress_data = json.load(f)

            # Add epoch history to final results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            final_results = {
                "split_mode": "stratified_4fold",
                "timestamp": timestamp,
                "experiment_id": experiment_id,
                "mean_dice": mean_dice,
                "fold_results": all_results,
                "epoch_history": progress_data.get("folds", [])  # Include full epoch data
            }

            output_file = config.SEG_LOG_DIR / f"results_stratified_{timestamp}.json"
            with open(output_file, 'w') as f:
                json.dump(final_results, f, indent=2)

            print(f"\n✓ Final results with epoch history saved to {output_file}")

        print(f"\n✓ Experiment completed. Progress saved to progress_{experiment_id}.json")

    except Exception as e:
        # Mark experiment as error
        tracker.mark_error(str(e))
        print(f"\n✗ Experiment failed: {e}")
        raise

if __name__ == "__main__":
    main()
