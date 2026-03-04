"""
Image and text preprocessing utilities for Kvasir-VQA-x1.

Provides:
- Image preprocessing (resize, normalize, augment)
- Text cleaning and tokenization
- Stratified train/val splitting
- Preprocessing statistics and validation

Usage:
    python src/preprocessing.py
"""

import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from collections import Counter

from sklearn.model_selection import train_test_split
from torchvision import transforms


def load_config(config_path="configs/config.yaml"):
    """Load project configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def clean_text(text):
    """
    Clean and normalize text (questions/answers).

    - Strip whitespace
    - Normalize unicode
    - Remove extra spaces
    """
    if not isinstance(text, str):
        return str(text).strip()
    text = text.strip()
    text = " ".join(text.split())  # Normalize whitespace
    return text


def validate_images(df, image_dir):
    """
    Validate that all referenced images exist and are readable.

    Returns:
        tuple: (valid_df, invalid_ids)
    """
    image_dir = Path(image_dir)
    invalid_ids = []
    valid_ids = []

    unique_ids = df["img_id"].unique()
    print(f"[INFO] Validating {len(unique_ids)} unique images...")

    for img_id in tqdm(unique_ids, desc="Validating images"):
        img_path = image_dir / f"{img_id}.jpg"
        if img_path.exists():
            try:
                img = Image.open(img_path)
                img.verify()
                valid_ids.append(img_id)
            except Exception:
                invalid_ids.append(img_id)
        else:
            invalid_ids.append(img_id)

    if invalid_ids:
        print(f"[WARNING] {len(invalid_ids)} images are missing or corrupted.")
        valid_df = df[df["img_id"].isin(valid_ids)].copy()
    else:
        valid_df = df.copy()

    print(f"[INFO] {len(valid_ids)} valid images, {len(invalid_ids)} invalid.")
    return valid_df, invalid_ids





def create_stratified_split(train_df, val_ratio=0.1, seed=42):
    """
    Create stratified train/validation split.

    Stratifies by complexity to maintain distribution.

    Returns:
        tuple: (train_split, val_split)
    """
    train_split, val_split = train_test_split(
        train_df,
        test_size=val_ratio,
        random_state=seed,
        stratify=train_df["complexity"],
    )

    # Verify distribution
    print("\n[INFO] Split statistics:")
    print(f"  Train: {len(train_split):,} samples")
    print(f"  Val:   {len(val_split):,} samples")
    print("\n  Complexity distribution (Train -> Val):")
    for c in sorted(train_df["complexity"].unique()):
        train_pct = (train_split["complexity"] == c).mean() * 100
        val_pct = (val_split["complexity"] == c).mean() * 100
        print(f"    Level {c}: {train_pct:.1f}% -> {val_pct:.1f}%")

    return train_split, val_split


def compute_preprocessing_stats(train_df, val_df, test_df):
    """Compute and print preprocessing summary statistics."""
    print("\n" + "=" * 60)
    print("PREPROCESSING SUMMARY")
    print("=" * 60)

    for name, df in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
        print(f"\n  {name} Split:")
        print(f"    Samples:          {len(df):,}")
        print(f"    Unique images:    {df['img_id'].nunique():,}")
        print(f"    Avg Q length:     {df['question_length'].mean():.1f} words")
        print(f"    Avg A length:     {df['answer_length'].mean():.1f} words")
        print(f"    Complexity dist:  {dict(df['complexity'].value_counts().sort_index())}")

    print("=" * 60)


def main():
    config = load_config()
    data_dir = Path(config["paths"]["data_dir"])
    image_dir = Path(config["paths"]["image_dir"])
    val_ratio = config["dataset"]["val_split_ratio"]
    seed = config["dataset"]["random_seed"]

    # Load raw data
    print("[INFO] Loading raw data...")
    train_df = pd.read_csv(data_dir / "kvasir_vqa_x1_train.csv")
    test_df = pd.read_csv(data_dir / "kvasir_vqa_x1_test.csv")

    # Preprocess text
    print("\n[STEP 1] Cleaning text data...")
    train_df = preprocess_dataframe(train_df)
    test_df = preprocess_dataframe(test_df)

    # Validate images
    print("\n[STEP 2] Validating images...")
    if image_dir.exists():
        train_df, _ = validate_images(train_df, image_dir)
        test_df, _ = validate_images(test_df, image_dir)
    else:
        print("[SKIP] Image directory not found. Run download_dataset.py first.")

    # Stratified split
    print("\n[STEP 3] Creating stratified train/val split...")
    train_split, val_split = create_stratified_split(train_df, val_ratio, seed)

    # Save preprocessed splits
    print("\n[STEP 4] Saving preprocessed splits...")
    train_split.to_csv(data_dir / "preprocessed_train.csv", index=False)
    val_split.to_csv(data_dir / "preprocessed_val.csv", index=False)
    test_df.to_csv(data_dir / "preprocessed_test.csv", index=False)
    print(f"[INFO] Saved preprocessed CSVs to {data_dir}/")

    # Summary
    compute_preprocessing_stats(train_split, val_split, test_df)

    print("\n[DONE] Preprocessing complete!")


if __name__ == "__main__":
    main()
