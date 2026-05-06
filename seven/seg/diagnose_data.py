#!/usr/bin/env python
"""
Data quality diagnostic script for all 19 patients.
Analyzes: foreground ratio, annotation count, image quality, etc.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(1, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image
import json
from collections import defaultdict

import config_seg as config

print("="*80)
print("Data Quality Diagnostics for All Patients")
print("="*80)

# Restore all 19 patients for analysis
all_patients = [f"P{i:03d}" for i in range(1, 20)]

results = {}

for patient_id in all_patients:
    print(f"\n[{patient_id}]")

    patient_dir = config.DATA_DIR / patient_id
    mask_dir = patient_dir / "mask"
    data_dir = patient_dir / "Data"

    if not mask_dir.exists():
        print(f"  ✗ No mask directory")
        results[patient_id] = {"status": "no_mask_dir"}
        continue

    # Count masks
    mask_files = list(mask_dir.glob("*_mask.png"))
    num_masks = len(mask_files)

    if num_masks == 0:
        print(f"  ✗ No masks found")
        results[patient_id] = {"status": "no_masks"}
        continue

    print(f"  Masks: {num_masks}")

    # Analyze foreground ratio
    fg_ratios = []
    mask_sizes = []
    image_qualities = []
    valid_samples = 0

    for mask_path in mask_files:
        # Check corresponding image
        img_name = mask_path.stem.replace("_mask", "") + ".jpg"
        img_path = data_dir / img_name

        if not img_path.exists():
            continue

        valid_samples += 1

        # Load mask
        mask = Image.open(mask_path).convert("L")
        mask_arr = np.array(mask) / 255.0
        fg_ratio = mask_arr.mean()
        fg_ratios.append(fg_ratio)

        # Mask size
        mask_sizes.append(mask.size)

        # Image quality (variance as proxy)
        img = Image.open(img_path).convert("L")
        img_arr = np.array(img) / 255.0
        img_var = img_arr.var()
        image_qualities.append(img_var)

    if valid_samples == 0:
        print(f"  ✗ No valid image-mask pairs")
        results[patient_id] = {"status": "no_valid_pairs"}
        continue

    # Statistics
    fg_mean = np.mean(fg_ratios)
    fg_std = np.std(fg_ratios)
    fg_min = np.min(fg_ratios)
    fg_max = np.max(fg_ratios)

    img_qual_mean = np.mean(image_qualities)

    print(f"  Valid samples: {valid_samples}")
    print(f"  Foreground ratio: {fg_mean:.4f} ± {fg_std:.4f} (min={fg_min:.4f}, max={fg_max:.4f})")
    print(f"  Image quality (variance): {img_qual_mean:.4f}")

    # Check for anomalies
    anomalies = []
    if fg_mean < 0.005:
        anomalies.append("very_low_foreground")
    if fg_mean > 0.10:
        anomalies.append("very_high_foreground")
    if fg_std > 0.05:
        anomalies.append("high_variance_foreground")
    if img_qual_mean < 0.01:
        anomalies.append("low_image_quality")

    if anomalies:
        print(f"  ⚠️  Anomalies: {', '.join(anomalies)}")
    else:
        print(f"  ✓ No anomalies detected")

    results[patient_id] = {
        "status": "ok",
        "num_masks": num_masks,
        "valid_samples": valid_samples,
        "fg_mean": float(fg_mean),
        "fg_std": float(fg_std),
        "fg_min": float(fg_min),
        "fg_max": float(fg_max),
        "img_quality_mean": float(img_qual_mean),
        "anomalies": anomalies,
    }

# Summary
print("\n" + "="*80)
print("Summary")
print("="*80)

# Group patients
old_patients = ["P001", "P002", "P003", "P004"]
new_patients = [p for p in all_patients if p not in old_patients]

def summarize_group(group_name, patient_list):
    print(f"\n{group_name}:")
    valid_patients = [p for p in patient_list if results.get(p, {}).get("status") == "ok"]

    if not valid_patients:
        print("  No valid patients")
        return

    total_samples = sum(results[p]["valid_samples"] for p in valid_patients)
    fg_means = [results[p]["fg_mean"] for p in valid_patients]

    print(f"  Patients: {len(valid_patients)}/{len(patient_list)}")
    print(f"  Total samples: {total_samples}")
    print(f"  Avg foreground ratio: {np.mean(fg_means):.4f} ± {np.std(fg_means):.4f}")
    print(f"  Range: [{np.min(fg_means):.4f}, {np.max(fg_means):.4f}]")

    # List patients with anomalies
    anomaly_patients = [p for p in valid_patients if results[p]["anomalies"]]
    if anomaly_patients:
        print(f"  ⚠️  Patients with anomalies: {', '.join(anomaly_patients)}")

summarize_group("Old Patients (P001-P004)", old_patients)
summarize_group("New Patients (P005-P019)", new_patients)

# Save results
output_path = config.SEG_LOG_DIR / "data_quality_report.json"
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Full report saved to: {output_path}")
print("="*80)
