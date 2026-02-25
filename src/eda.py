"""
Exploratory Data Analysis (EDA) for Kvasir-VQA-x1 dataset.

Generates visualizations and statistics saved to results/eda/:
- Dataset statistics summary
- Question class distribution (bar chart)
- Complexity level distribution (pie chart)
- Answer length distribution (histogram)
- Sample image grid with Q&A pairs

Usage:
    python src/eda.py
"""

import os
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from PIL import Image
from collections import Counter


# Set publication-quality plot style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.2,
})


def load_config(config_path="configs/config.yaml"):
    """Load project configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_data(config):
    """Load train and test DataFrames."""
    data_dir = Path(config["paths"]["data_dir"])
    train_df = pd.read_csv(data_dir / "kvasir_vqa_x1_train.csv")
    test_df = pd.read_csv(data_dir / "kvasir_vqa_x1_test.csv")
    return train_df, test_df


def print_and_save_statistics(train_df, test_df, output_dir):
    """Print and save dataset statistics."""
    full_df = pd.concat([train_df, test_df], ignore_index=True)

    stats_lines = [
        "=" * 60,
        "KVASIR-VQA-x1 DATASET STATISTICS",
        "=" * 60,
        f"  Total QA pairs:         {len(full_df):,}",
        f"  Training QA pairs:      {len(train_df):,}",
        f"  Test QA pairs:          {len(test_df):,}",
        f"  Unique images (total):  {full_df['img_id'].nunique():,}",
        f"  Unique images (train):  {train_df['img_id'].nunique():,}",
        f"  Unique images (test):   {test_df['img_id'].nunique():,}",
        "",
        "  Complexity distribution (full):",
    ]

    for c in sorted(full_df["complexity"].unique()):
        count = (full_df["complexity"] == c).sum()
        pct = count / len(full_df) * 100
        stats_lines.append(f"    Level {c}: {count:,} ({pct:.1f}%)")

    stats_lines.append("")
    stats_lines.append(f"  Number of question classes: {full_df['question_class'].nunique()}")
    stats_lines.append(f"  Avg answer length (words): {full_df['answer'].str.split().str.len().mean():.1f}")
    stats_lines.append(f"  Avg question length (words): {full_df['question'].str.split().str.len().mean():.1f}")
    stats_lines.append("=" * 60)

    stats_text = "\n".join(stats_lines)
    print(stats_text)

    # Save to file
    stats_path = output_dir / "dataset_statistics.txt"
    with open(stats_path, "w") as f:
        f.write(stats_text)
    print(f"\n[INFO] Statistics saved to {stats_path}")

    return full_df


def plot_question_class_distribution(full_df, output_dir, top_n=15):
    """Plot top-N question class distribution as horizontal bar chart."""
    fig, ax = plt.subplots(figsize=(10, 7))

    class_counts = full_df["question_class"].value_counts().head(top_n)

    colors = sns.color_palette("viridis", n_colors=len(class_counts))
    bars = ax.barh(
        range(len(class_counts)),
        class_counts.values,
        color=colors,
        edgecolor="white",
        linewidth=0.5,
    )

    ax.set_yticks(range(len(class_counts)))
    ax.set_yticklabels(class_counts.index, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Number of QA Pairs")
    ax.set_title(f"Top {top_n} Question Classes in Kvasir-VQA-x1")

    # Add count labels
    for bar, count in zip(bars, class_counts.values):
        ax.text(
            bar.get_width() + max(class_counts.values) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{count:,}",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()
    save_path = output_dir / "question_class_distribution.png"
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[INFO] Saved: {save_path}")


def plot_complexity_distribution(full_df, output_dir):
    """Plot complexity level distribution as pie chart + bar chart."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    complexity_counts = full_df["complexity"].value_counts().sort_index()
    labels = [f"Level {c}" for c in complexity_counts.index]
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]

    # Pie chart
    wedges, texts, autotexts = axes[0].pie(
        complexity_counts.values,
        labels=labels,
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
        textprops={"fontsize": 11},
    )
    axes[0].set_title("Complexity Distribution")

    # Bar chart
    axes[1].bar(labels, complexity_counts.values, color=colors, edgecolor="white", width=0.6)
    axes[1].set_ylabel("Number of QA Pairs")
    axes[1].set_title("Complexity Level Counts")
    for i, (label, count) in enumerate(zip(labels, complexity_counts.values)):
        axes[1].text(i, count + max(complexity_counts.values) * 0.02,
                     f"{count:,}", ha="center", fontsize=10)

    plt.tight_layout()
    save_path = output_dir / "complexity_distribution.png"
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[INFO] Saved: {save_path}")


def plot_answer_length_distribution(full_df, output_dir):
    """Plot answer and question length distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Answer length
    answer_lengths = full_df["answer"].str.split().str.len()
    axes[0].hist(answer_lengths, bins=50, color="#3498db", edgecolor="white", alpha=0.8)
    axes[0].set_xlabel("Answer Length (words)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Answer Length Distribution")
    axes[0].axvline(answer_lengths.mean(), color="#e74c3c", linestyle="--",
                    label=f"Mean: {answer_lengths.mean():.1f}")
    axes[0].legend()

    # Question length
    question_lengths = full_df["question"].str.split().str.len()
    axes[1].hist(question_lengths, bins=50, color="#9b59b6", edgecolor="white", alpha=0.8)
    axes[1].set_xlabel("Question Length (words)")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Question Length Distribution")
    axes[1].axvline(question_lengths.mean(), color="#e74c3c", linestyle="--",
                    label=f"Mean: {question_lengths.mean():.1f}")
    axes[1].legend()

    plt.tight_layout()
    save_path = output_dir / "text_length_distribution.png"
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[INFO] Saved: {save_path}")


def plot_train_test_comparison(train_df, test_df, output_dir):
    """Plot train vs test split comparison by complexity."""
    fig, ax = plt.subplots(figsize=(8, 5))

    complexity_levels = sorted(train_df["complexity"].unique())
    x = np.arange(len(complexity_levels))
    width = 0.35

    train_counts = [len(train_df[train_df["complexity"] == c]) for c in complexity_levels]
    test_counts = [len(test_df[test_df["complexity"] == c]) for c in complexity_levels]

    bars1 = ax.bar(x - width / 2, train_counts, width, label="Train", color="#3498db", edgecolor="white")
    bars2 = ax.bar(x + width / 2, test_counts, width, label="Test", color="#e74c3c", edgecolor="white")

    ax.set_xlabel("Complexity Level")
    ax.set_ylabel("Number of QA Pairs")
    ax.set_title("Train vs Test Distribution by Complexity")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Level {c}" for c in complexity_levels])
    ax.legend()

    # Add count labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + max(train_counts) * 0.01,
                    f"{int(height):,}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    save_path = output_dir / "train_test_comparison.png"
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[INFO] Saved: {save_path}")


def plot_sample_images_with_qa(train_df, image_dir, output_dir, grid_size=9):
    """Plot a grid of sample images with their Q&A pairs."""
    n = min(grid_size, len(train_df))
    cols = 3
    rows = (n + cols - 1) // cols

    # Sample diverse examples (one from each complexity if possible)
    samples = []
    for complexity in sorted(train_df["complexity"].unique()):
        subset = train_df[train_df["complexity"] == complexity]
        sample_n = min(3, len(subset))
        samples.append(subset.sample(n=sample_n, random_state=42))
    sample_df = pd.concat(samples).head(n)

    fig = plt.figure(figsize=(5 * cols, 6 * rows))
    gs = gridspec.GridSpec(rows, cols, hspace=0.4, wspace=0.3)

    for idx, (_, row) in enumerate(sample_df.iterrows()):
        if idx >= n:
            break

        ax = fig.add_subplot(gs[idx])

        # Load image
        img_path = Path(image_dir) / f"{row['img_id']}.jpg"
        try:
            img = Image.open(img_path)
            ax.imshow(img)
        except FileNotFoundError:
            ax.text(0.5, 0.5, "Image\nNot Found", ha="center", va="center",
                    transform=ax.transAxes, fontsize=14)

        ax.set_axis_off()

        # Truncate long text
        q_text = row["question"][:80] + ("..." if len(row["question"]) > 80 else "")
        a_text = str(row["answer"])[:60] + ("..." if len(str(row["answer"])) > 60 else "")

        ax.set_title(
            f"[Complexity {row['complexity']}]\n"
            f"Q: {q_text}\n"
            f"A: {a_text}",
            fontsize=8,
            loc="left",
            pad=5,
        )

    plt.suptitle("Sample Images with Q&A Pairs from Kvasir-VQA-x1", fontsize=16, y=1.02)
    save_path = output_dir / "sample_images_qa.png"
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[INFO] Saved: {save_path}")


def main():
    config = load_config()
    output_dir = Path(config["paths"]["eda_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    image_dir = config["paths"]["image_dir"]
    top_n = config["eda"]["top_n_question_classes"]
    grid_size = config["eda"]["sample_grid_size"]

    print("[INFO] Loading dataset...")
    train_df, test_df = load_data(config)

    # 1. Statistics
    print("\n[1/5] Dataset Statistics")
    full_df = print_and_save_statistics(train_df, test_df, output_dir)

    # 2. Question class distribution
    print("\n[2/5] Question Class Distribution")
    plot_question_class_distribution(full_df, output_dir, top_n=top_n)

    # 3. Complexity distribution
    print("\n[3/5] Complexity Distribution")
    plot_complexity_distribution(full_df, output_dir)

    # 4. Text length distributions
    print("\n[4/5] Text Length Distribution")
    plot_answer_length_distribution(full_df, output_dir)

    # 5. Train vs Test comparison
    print("\n[5/5] Train vs Test Comparison")
    plot_train_test_comparison(train_df, test_df, output_dir)

    # 6. Sample images (only if images exist)
    if Path(image_dir).exists() and any(Path(image_dir).glob("*.jpg")):
        print("\n[BONUS] Sample Images with Q&A")
        plot_sample_images_with_qa(train_df, image_dir, output_dir, grid_size=grid_size)
    else:
        print("\n[SKIP] Sample images — no images found. Run download_dataset.py first.")

    print(f"\n[DONE] All EDA outputs saved to {output_dir}/")


if __name__ == "__main__":
    main()
