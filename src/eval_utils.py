"""
Shared evaluation utilities for VSLM model comparison on Kvasir-VQA-x1.

Provides consistent evaluation metrics and result saving across all models:
- Exact Match Accuracy
- Word-level F1 Score
- Per-complexity breakdown
- Diverse sample selection

Usage:
    from src.eval_utils import compute_word_f1, evaluate_predictions, save_results
"""

import os
import json
import pandas as pd


def compute_word_f1(prediction, ground_truth):
    """
    Compute word-level F1 score between prediction and ground truth.
    This is the standard VQA evaluation metric that handles paraphrasing.

    Args:
        prediction: Model's predicted answer string
        ground_truth: Ground truth answer string

    Returns:
        float: Word-level F1 score between 0.0 and 1.0
    """
    pred_tokens = set(prediction.strip().lower().split())
    gt_tokens = set(ground_truth.strip().lower().split())

    if not pred_tokens or not gt_tokens:
        return 0.0

    common = pred_tokens & gt_tokens
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def compute_exact_match(prediction, ground_truth):
    """
    Check if prediction exactly matches ground truth (case-insensitive).

    Args:
        prediction: Model's predicted answer string
        ground_truth: Ground truth answer string

    Returns:
        bool: True if exact match
    """
    return prediction.strip().lower() == ground_truth.strip().lower()


def select_diverse_samples(df, n_samples, seed=42):
    """
    Select diverse samples across complexity levels using stratified sampling.

    Args:
        df: DataFrame with 'complexity' column
        n_samples: Total number of samples to select
        seed: Random seed for reproducibility

    Returns:
        pandas.DataFrame: Selected samples
    """
    samples = []
    for complexity in sorted(df["complexity"].unique()):
        subset = df[df["complexity"] == complexity]
        n = max(1, n_samples // 3)
        samples.append(subset.sample(n=min(n, len(subset)), random_state=seed))
    return pd.concat(samples).head(n_samples)


def evaluate_predictions(results):
    """
    Compute all evaluation metrics from a list of result dictionaries.

    Each result dict should have: 'prediction', 'ground_truth', 'complexity',
    'img_id', 'question', 'question_class'.

    Args:
        results: List of result dicts from model inference

    Returns:
        dict: Summary with exact_match_accuracy, average_word_f1,
              per_complexity breakdown, etc.
    """
    if not results:
        return {"error": "No results to evaluate"}

    total = len(results)
    exact_matches = sum(1 for r in results if r["exact_match"])
    total_f1 = sum(r["word_f1"] for r in results)
    partial_matches = sum(1 for r in results if r["word_f1"] >= 0.5)

    exact_acc = (exact_matches / total * 100) if total > 0 else 0
    avg_f1 = (total_f1 / total * 100) if total > 0 else 0

    # Per-complexity breakdown
    results_df = pd.DataFrame(results)
    per_complexity = {}
    for c in sorted(results_df["complexity"].unique()):
        c_df = results_df[results_df["complexity"] == c]
        per_complexity[f"level_{c}"] = {
            "exact_matches": int(c_df["exact_match"].sum()),
            "total": int(len(c_df)),
            "exact_accuracy": round(c_df["exact_match"].mean() * 100, 1),
            "avg_word_f1": round(c_df["word_f1"].mean() * 100, 1),
        }

    summary = {
        "num_samples": total,
        "exact_match_accuracy": round(exact_acc, 1),
        "average_word_f1": round(avg_f1, 1),
        "exact_matches": int(exact_matches),
        "partial_matches_f1_gte_50": int(partial_matches),
        "total": total,
        "per_complexity": per_complexity,
    }

    return summary


def process_single_result(row, prediction):
    """
    Process a single inference result into a standardized result dict.

    Args:
        row: DataFrame row with img_id, complexity, question_class, question, answer
        prediction: Model's predicted answer string

    Returns:
        dict: Standardized result dict
    """
    ground_truth = str(row["answer"])
    is_exact = compute_exact_match(prediction, ground_truth)
    f1_score = compute_word_f1(prediction, ground_truth)

    return {
        "img_id": row["img_id"],
        "complexity": int(row["complexity"]),
        "question_class": row["question_class"],
        "question": row["question"],
        "ground_truth": ground_truth,
        "prediction": prediction,
        "exact_match": is_exact,
        "word_f1": round(f1_score, 3),
    }


def print_result(idx, total, result):
    """Print a formatted single result during inference."""
    if result["exact_match"]:
        status = "✓ EXACT"
    elif result["word_f1"] >= 0.5:
        status = "~ PARTIAL"
    else:
        status = "✗ WRONG"

    print(f"[{idx + 1}/{total}] {status} (F1: {result['word_f1']:.2f})")
    print(f"  Complexity: {result['complexity']}")
    print(f"  Question:   {result['question'][:100]}")
    print(f"  GT Answer:  {result['ground_truth'][:80]}")
    print(f"  Predicted:  {result['prediction'][:80]}")
    print("-" * 80)


def print_summary(model_name, summary):
    """Print formatted evaluation summary."""
    print(f"\n{'=' * 70}")
    print(f"  {model_name.upper()} — EVALUATION SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Total samples:            {summary['total']}")
    print(f"  Exact matches:            {summary['exact_matches']}/{summary['total']} ({summary['exact_match_accuracy']:.1f}%)")
    print(f"  Partial matches (F1≥0.5): {summary['partial_matches_f1_gte_50']}/{summary['total']}")
    print(f"  Average Word F1:          {summary['average_word_f1']:.1f}%")
    print(f"\n  Per-Complexity:")
    for key, val in summary["per_complexity"].items():
        level = key.replace("level_", "")
        print(f"    Level {level}: Exact {val['exact_matches']}/{val['total']}, Avg F1: {val['avg_word_f1']:.1f}%")
    print(f"{'=' * 70}")


def save_results(results, model_name, output_dir):
    """
    Save predictions CSV and summary JSON for a model.

    Args:
        results: List of result dicts
        model_name: Name identifier (e.g., 'moe_tinymed')
        output_dir: Directory to save results

    Returns:
        dict: Summary metrics
    """
    os.makedirs(output_dir, exist_ok=True)

    results_df = pd.DataFrame(results)
    summary = evaluate_predictions(results)
    summary["model"] = model_name

    # Save predictions CSV
    pred_path = os.path.join(output_dir, f"{model_name}_predictions.csv")
    results_df.to_csv(pred_path, index=False)
    print(f"[INFO] Predictions saved to {pred_path}")

    # Save summary JSON
    summary_path = os.path.join(output_dir, f"{model_name}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[INFO] Summary saved to {summary_path}")

    return summary


def load_baseline_summary(predictions_dir):
    """
    Load the existing baseline summary for comparison.

    Args:
        predictions_dir: Path to results/predictions/

    Returns:
        dict: Baseline summary or None if not found
    """
    summary_path = os.path.join(predictions_dir, "baseline_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            return json.load(f)
    return None


def build_comparison_table(summaries):
    """
    Build a comparison DataFrame from multiple model summaries.

    Args:
        summaries: Dict mapping model_name -> summary dict

    Returns:
        pandas.DataFrame: Comparison table
    """
    rows = []
    for model_name, summary in summaries.items():
        row = {
            "Model": model_name,
            "Exact Match (%)": summary.get("exact_match_accuracy", 0),
            "Avg Word F1 (%)": summary.get("average_word_f1", 0),
            "Partial Matches": summary.get("partial_matches_f1_gte_50", 0),
            "Total Samples": summary.get("total", summary.get("num_samples", 0)),
        }

        # Add per-complexity F1
        per_complexity = summary.get("per_complexity", {})
        for key, val in per_complexity.items():
            level = key.replace("level_", "L")
            row[f"F1 {level} (%)"] = val.get("avg_word_f1", 0)

        rows.append(row)

    return pd.DataFrame(rows)
