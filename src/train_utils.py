"""
Training utilities for Kvasir-VQA-x1 Medical VQA project.

Provides shared metrics, evaluation, and data sampling functions
used across training notebooks and scripts.
"""

import re
import numpy as np
import pandas as pd
from collections import Counter


def normalize_text(text):
    """Normalize text for comparison: lowercase, strip, remove extra spaces."""
    text = str(text).lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text


def compute_word_f1(prediction, ground_truth):
    """
    Compute word-level F1 score between prediction and ground truth.
    
    Args:
        prediction: Model-generated answer string
        ground_truth: Ground truth answer string
        
    Returns:
        float: F1 score between 0 and 1
    """
    pred_tokens = normalize_text(prediction).split()
    gt_tokens = normalize_text(ground_truth).split()
    
    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0
    
    pred_counts = Counter(pred_tokens)
    gt_counts = Counter(gt_tokens)
    
    # Count common tokens
    common = sum((pred_counts & gt_counts).values())
    
    if common == 0:
        return 0.0
    
    precision = common / len(pred_tokens)
    recall = common / len(gt_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    
    return f1


def compute_exact_match(prediction, ground_truth):
    """Check if normalized prediction exactly matches ground truth."""
    return normalize_text(prediction) == normalize_text(ground_truth)


def select_stratified_subset(df, n_samples, complexity_col='complexity', seed=42):
    """
    Select a stratified subset balanced across complexity levels.
    
    Args:
        df: Full training DataFrame
        n_samples: Total number of samples to select
        complexity_col: Column name for stratification
        seed: Random seed
        
    Returns:
        pd.DataFrame: Stratified subset
    """
    np.random.seed(seed)
    
    levels = sorted(df[complexity_col].unique())
    per_level = n_samples // len(levels)
    remainder = n_samples % len(levels)
    
    subsets = []
    for i, level in enumerate(levels):
        level_df = df[df[complexity_col] == level]
        # Give remainder samples to later levels
        n = per_level + (1 if i >= len(levels) - remainder else 0)
        n = min(n, len(level_df))
        subsets.append(level_df.sample(n=n, random_state=seed))
    
    result = pd.concat(subsets, ignore_index=True)
    # Shuffle the combined result
    result = result.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    print(f"[INFO] Selected {len(result)} samples stratified by {complexity_col}:")
    for level in levels:
        count = len(result[result[complexity_col] == level])
        print(f"  Level {level}: {count}")
    
    return result


def evaluate_predictions(predictions, ground_truths, complexities=None):
    """
    Compute evaluation metrics for a batch of predictions.
    
    Args:
        predictions: List of predicted answer strings
        ground_truths: List of ground truth answer strings
        complexities: Optional list of complexity levels for breakdown
        
    Returns:
        dict: Evaluation metrics
    """
    assert len(predictions) == len(ground_truths)
    
    exact_matches = 0
    word_f1_scores = []
    partial_matches = 0  # F1 >= 0.5
    
    for pred, gt in zip(predictions, ground_truths):
        em = compute_exact_match(pred, gt)
        f1 = compute_word_f1(pred, gt)
        
        exact_matches += int(em)
        word_f1_scores.append(f1)
        if f1 >= 0.5:
            partial_matches += 1
    
    total = len(predictions)
    results = {
        'total': total,
        'exact_matches': exact_matches,
        'exact_match_accuracy': exact_matches / total * 100 if total > 0 else 0,
        'average_word_f1': np.mean(word_f1_scores) * 100 if word_f1_scores else 0,
        'partial_matches_gte_50': partial_matches,
        'partial_match_rate': partial_matches / total * 100 if total > 0 else 0,
    }
    
    # Per-complexity breakdown
    if complexities is not None:
        per_complexity = {}
        for level in sorted(set(complexities)):
            indices = [i for i, c in enumerate(complexities) if c == level]
            level_f1s = [word_f1_scores[i] for i in indices]
            level_ems = sum(1 for i in indices 
                          if compute_exact_match(predictions[i], ground_truths[i]))
            per_complexity[f'level_{level}'] = {
                'total': len(indices),
                'exact_matches': level_ems,
                'exact_accuracy': level_ems / len(indices) * 100 if indices else 0,
                'avg_word_f1': np.mean(level_f1s) * 100 if level_f1s else 0,
            }
        results['per_complexity'] = per_complexity
    
    return results


def print_evaluation_results(results, model_name="Model"):
    """Pretty-print evaluation results."""
    print(f"\n{'='*60}")
    print(f"  {model_name} — Evaluation Results")
    print(f"{'='*60}")
    print(f"  Total samples:       {results['total']}")
    print(f"  Exact Match:         {results['exact_matches']}/{results['total']} "
          f"({results['exact_match_accuracy']:.1f}%)")
    print(f"  Average Word F1:     {results['average_word_f1']:.1f}%")
    print(f"  Partial Match (≥50): {results['partial_matches_gte_50']}/{results['total']} "
          f"({results['partial_match_rate']:.1f}%)")
    
    if 'per_complexity' in results:
        print(f"\n  Per-Complexity Breakdown:")
        for level, stats in results['per_complexity'].items():
            print(f"    {level}: EM={stats['exact_accuracy']:.1f}%, "
                  f"F1={stats['avg_word_f1']:.1f}% "
                  f"(n={stats['total']})")
    print(f"{'='*60}\n")
