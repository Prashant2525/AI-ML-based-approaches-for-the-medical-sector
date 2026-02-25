"""
PyTorch Dataset and DataLoader for Kvasir-VQA-x1.

Provides:
- KvasirVQADataset: PyTorch Dataset class
- get_image_transform: Image preprocessing transforms
- create_dataloaders: Train/val/test DataLoader factory

Usage:
    from src.dataset import create_dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(config)
"""

import os
import json
import yaml
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split


def load_config(config_path="configs/config.yaml"):
    """Load project configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_image_transform(config, augment=False):
    """
    Build image preprocessing transform pipeline.

    Args:
        config: Project configuration dict
        augment: If True, apply data augmentation (for training)

    Returns:
        torchvision.transforms.Compose
    """
    img_size = config["image"]["size"]
    mean = config["image"]["mean"]
    std = config["image"]["std"]

    if augment:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])


class KvasirVQADataset(Dataset):
    """
    PyTorch Dataset for Kvasir-VQA-x1.

    Each sample returns:
        - image: Preprocessed image tensor (3 x H x W)
        - question: Question string
        - answer: Answer string
        - metadata: Dict with img_id, complexity, question_class
    """

    def __init__(self, dataframe, image_dir, transform=None):
        """
        Args:
            dataframe: pandas DataFrame with columns [img_id, question, answer, complexity, question_class]
            image_dir: Path to the directory containing images
            transform: torchvision transform to apply to images
        """
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load image
        img_path = self.image_dir / f"{row['img_id']}.jpg"
        try:
            image = Image.open(img_path).convert("RGB")
        except (FileNotFoundError, OSError):
            # Return a black image as fallback
            image = Image.new("RGB", (224, 224), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        # Metadata
        metadata = {
            "img_id": row["img_id"],
            "complexity": int(row["complexity"]),
            "question_class": row["question_class"],
        }

        return {
            "image": image,
            "question": row["question"],
            "answer": row["answer"],
            "metadata": metadata,
        }


def create_dataloaders(config, config_path="configs/config.yaml"):
    """
    Create train, validation, and test DataLoaders.

    Steps:
        1. Load train and test CSVs
        2. Split train into train + validation (stratified by complexity)
        3. Create Dataset objects with appropriate transforms
        4. Wrap in DataLoaders

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    data_dir = Path(config["paths"]["data_dir"])
    image_dir = Path(config["paths"]["image_dir"])

    # Load data
    train_df = pd.read_csv(data_dir / "kvasir_vqa_x1_train.csv")
    test_df = pd.read_csv(data_dir / "kvasir_vqa_x1_test.csv")

    # Stratified train/val split
    val_ratio = config["dataset"]["val_split_ratio"]
    seed = config["dataset"]["random_seed"]

    train_split, val_split = train_test_split(
        train_df,
        test_size=val_ratio,
        random_state=seed,
        stratify=train_df["complexity"],
    )

    print(f"[INFO] Dataset splits:")
    print(f"  Train:      {len(train_split):,} samples")
    print(f"  Validation: {len(val_split):,} samples")
    print(f"  Test:       {len(test_df):,} samples")

    # Transforms
    train_transform = get_image_transform(config, augment=True)
    eval_transform = get_image_transform(config, augment=False)

    # Datasets
    train_dataset = KvasirVQADataset(train_split, image_dir, train_transform)
    val_dataset = KvasirVQADataset(val_split, image_dir, eval_transform)
    test_dataset = KvasirVQADataset(test_df, image_dir, eval_transform)

    # DataLoaders
    loader_cfg = config["dataloader"]

    def collate_fn(batch):
        """Custom collate to handle mixed types (tensors + strings)."""
        images = torch.stack([item["image"] for item in batch])
        questions = [item["question"] for item in batch]
        answers = [item["answer"] for item in batch]
        metadata = [item["metadata"] for item in batch]
        return {
            "images": images,
            "questions": questions,
            "answers": answers,
            "metadata": metadata,
        }

    train_loader = DataLoader(
        train_dataset,
        batch_size=loader_cfg["batch_size"],
        shuffle=True,
        num_workers=loader_cfg["num_workers"],
        pin_memory=loader_cfg["pin_memory"],
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=loader_cfg["batch_size"],
        shuffle=False,
        num_workers=loader_cfg["num_workers"],
        pin_memory=loader_cfg["pin_memory"],
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=loader_cfg["batch_size"],
        shuffle=False,
        num_workers=loader_cfg["num_workers"],
        pin_memory=loader_cfg["pin_memory"],
        collate_fn=collate_fn,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    config = load_config()
    train_loader, val_loader, test_loader = create_dataloaders(config)

    # Quick sanity check
    batch = next(iter(train_loader))
    print(f"\n[SANITY CHECK]")
    print(f"  Batch image shape: {batch['images'].shape}")
    print(f"  Sample question:   {batch['questions'][0][:80]}...")
    print(f"  Sample answer:     {batch['answers'][0][:80]}")
    print(f"  Sample metadata:   {batch['metadata'][0]}")
