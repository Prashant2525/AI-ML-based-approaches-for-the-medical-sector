"""
Download and prepare the Kvasir-VQA-x1 dataset from HuggingFace.

This script:
1. Downloads images from SimulaMet-HOST/Kvasir-VQA (raw split)
2. Downloads QA pairs from SimulaMet/Kvasir-VQA-x1 (train + test)
3. Saves images as JPGs and QA data as CSV files

Usage:
    python src/download_dataset.py
"""

import os
import json
import yaml
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset


def load_config(config_path="configs/config.yaml"):
    """Load project configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def download_images(config):
    """Download images from the Kvasir-VQA HuggingFace repository."""
    img_dir = Path(config["paths"]["image_dir"])
    img_dir.mkdir(exist_ok=True, parents=True)

    # Check if images are already downloaded
    existing_images = list(img_dir.glob("*.jpg"))
    if len(existing_images) > 100:
        print(f"[INFO] Found {len(existing_images)} images already downloaded. Skipping...")
        return len(existing_images)

    print("[INFO] Downloading images from SimulaMet-HOST/Kvasir-VQA...")
    ds_host = load_dataset(
        config["dataset"]["hf_image_repo"],
        split=config["dataset"]["image_split"]
    )

    seen = set()
    saved_count = 0
    for row in tqdm(ds_host, desc="Saving images"):
        img_id = row["img_id"]
        if img_id not in seen:
            img_path = img_dir / f"{img_id}.jpg"
            row["image"].save(img_path)
            seen.add(img_id)
            saved_count += 1

    print(f"[INFO] Saved {saved_count} unique images to {img_dir}")
    return saved_count


def download_qa_pairs(config):
    """Download QA pairs from the Kvasir-VQA-x1 dataset."""
    data_dir = Path(config["paths"]["data_dir"])
    data_dir.mkdir(exist_ok=True, parents=True)

    splits_info = {}

    for split in ["train", "test"]:
        csv_path = data_dir / f"kvasir_vqa_x1_{split}.csv"

        # Check if already downloaded
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            print(f"[INFO] {split} split already exists with {len(df)} samples. Skipping...")
            splits_info[split] = len(df)
            continue

        print(f"[INFO] Downloading {split} split from SimulaMet/Kvasir-VQA-x1...")
        ds = load_dataset(config["dataset"]["hf_qa_repo"], split=split)

        # Convert to DataFrame
        records = []
        for row in tqdm(ds, desc=f"Processing {split}"):
            records.append({
                "img_id": row["img_id"],
                "complexity": row["complexity"],
                "question": row["question"],
                "answer": row["answer"],
                "question_class": row["question_class"],
                "original": json.dumps(row["original"]) if row.get("original") else ""
            })

        df = pd.DataFrame(records)
        df.to_csv(csv_path, index=False)
        splits_info[split] = len(df)
        print(f"[INFO] Saved {len(df)} {split} samples to {csv_path}")

    return splits_info


def print_summary(num_images, splits_info):
    """Print dataset download summary."""
    print("\n" + "=" * 60)
    print("DATASET DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"  Images downloaded:  {num_images}")
    for split, count in splits_info.items():
        print(f"  {split:>5s} QA pairs:     {count:,}")
    print("=" * 60)


def main():
    config = load_config()

    # Create output directories
    for dir_key in ["data_dir", "image_dir", "results_dir", "eda_dir", "predictions_dir"]:
        Path(config["paths"][dir_key]).mkdir(parents=True, exist_ok=True)

    # Download images
    num_images = download_images(config)

    # Download QA pairs
    splits_info = download_qa_pairs(config)

    # Print summary
    print_summary(num_images, splits_info)


if __name__ == "__main__":
    main()
