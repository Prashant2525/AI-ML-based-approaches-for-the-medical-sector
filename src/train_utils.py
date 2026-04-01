"""
Training utilities for Kvasir-VQA-x1 Medical VQA project.

Provides shared metrics, evaluation, and data sampling functions
used across training notebooks and scripts.

Metrics included:
  - Exact Match (EM)
  - Word F1 / Precision / Recall
  - BLEU-1, BLEU-2, BLEU-3, BLEU-4
  - ROUGE-L (F1)
  - METEOR
  - BERTScore (F1)
"""

import re
import numpy as np
import pandas as pd
from collections import Counter


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):
    """Normalize text for comparison: lowercase, strip, remove extra spaces."""
    text = str(text).lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text


# ============================================================
# CORE METRICS (no extra dependencies)
# ============================================================

def compute_word_f1(prediction, ground_truth):
    """Word-level F1 score between prediction and ground truth."""
    pred_tokens = normalize_text(prediction).split()
    gt_tokens = normalize_text(ground_truth).split()

    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0

    common = sum((Counter(pred_tokens) & Counter(gt_tokens)).values())
    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_word_recall(prediction, ground_truth):
    """
    Word-level recall: fraction of ground-truth tokens found in prediction.
    Critical in medical domain — missing a finding is worse than over-reporting.
    """
    pred_tokens = normalize_text(prediction).split()
    gt_tokens = normalize_text(ground_truth).split()

    if not gt_tokens:
        return 1.0
    if not pred_tokens:
        return 0.0

    common = sum((Counter(pred_tokens) & Counter(gt_tokens)).values())
    return common / len(gt_tokens)


def compute_word_precision(prediction, ground_truth):
    """Word-level precision: fraction of predicted tokens that are in ground truth."""
    pred_tokens = normalize_text(prediction).split()
    gt_tokens = normalize_text(ground_truth).split()

    if not pred_tokens:
        return 1.0 if not gt_tokens else 0.0
    common = sum((Counter(pred_tokens) & Counter(gt_tokens)).values())
    return common / len(pred_tokens)


def compute_exact_match(prediction, ground_truth):
    """Check if normalized prediction exactly matches ground truth."""
    return normalize_text(prediction) == normalize_text(ground_truth)


# ============================================================
# NLG METRICS (BLEU, ROUGE-L, METEOR)
# ============================================================

def compute_bleu_scores(prediction, ground_truth):
    """
    Compute BLEU-1 through BLEU-4 for a single prediction.
    Uses nltk.translate.bleu_score with smoothing.

    Returns:
        dict with keys 'bleu_1', 'bleu_2', 'bleu_3', 'bleu_4'
    """
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

    pred_tokens = normalize_text(prediction).split()
    gt_tokens = normalize_text(ground_truth).split()

    if not gt_tokens:
        return {'bleu_1': 1.0 if not pred_tokens else 0.0,
                'bleu_2': 1.0 if not pred_tokens else 0.0,
                'bleu_3': 1.0 if not pred_tokens else 0.0,
                'bleu_4': 1.0 if not pred_tokens else 0.0}
    if not pred_tokens:
        return {'bleu_1': 0.0, 'bleu_2': 0.0, 'bleu_3': 0.0, 'bleu_4': 0.0}

    smoothie = SmoothingFunction().method1
    ref = [gt_tokens]

    scores = {}
    for n in range(1, 5):
        weights = tuple([1.0 / n] * n + [0.0] * (4 - n))
        scores[f'bleu_{n}'] = sentence_bleu(ref, pred_tokens,
                                            weights=weights,
                                            smoothing_function=smoothie)
    return scores


def compute_rouge_l(prediction, ground_truth):
    """
    Compute ROUGE-L (Longest Common Subsequence) F1 score.
    Implemented without external dependency for portability.
    """
    pred_tokens = normalize_text(prediction).split()
    gt_tokens = normalize_text(ground_truth).split()

    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0

    # LCS via dynamic programming
    m, n = len(gt_tokens), len(pred_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if gt_tokens[i - 1] == pred_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_len = dp[m][n]
    if lcs_len == 0:
        return 0.0

    precision = lcs_len / n
    recall = lcs_len / m
    return 2 * precision * recall / (precision + recall)


def compute_meteor(prediction, ground_truth):
    """
    Compute METEOR score using nltk.
    METEOR considers stemming and synonym matching.
    """
    from nltk.translate.meteor_score import meteor_score as nltk_meteor
    import nltk
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)

    pred_tokens = normalize_text(prediction).split()
    gt_tokens = normalize_text(ground_truth).split()

    if not gt_tokens:
        return 1.0 if not pred_tokens else 0.0
    if not pred_tokens:
        return 0.0

    return nltk_meteor([gt_tokens], pred_tokens)


def compute_bertscore_batch(predictions, ground_truths):
    """
    Compute BERTScore (Precision, Recall, F1) for a batch of predictions.
    Uses the 'bert_score' library — call once for the full batch (efficient).

    Returns:
        dict with 'bertscore_precision', 'bertscore_recall', 'bertscore_f1'
              (each a list of floats, one per sample)
    """
    from bert_score import score as bert_score_fn

    P, R, F1 = bert_score_fn(
        predictions, ground_truths,
        lang="en",
        verbose=False,
        rescale_with_baseline=True,
    )
    return {
        'bertscore_precision': P.tolist(),
        'bertscore_recall': R.tolist(),
        'bertscore_f1': F1.tolist(),
    }


# ============================================================
# STRATIFIED SAMPLING
# ============================================================

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
        n = per_level + (1 if i >= len(levels) - remainder else 0)
        n = min(n, len(level_df))
        subsets.append(level_df.sample(n=n, random_state=seed))

    result = pd.concat(subsets, ignore_index=True)
    result = result.sample(frac=1, random_state=seed).reset_index(drop=True)

    print(f"[INFO] Selected {len(result)} samples stratified by {complexity_col}:")
    for level in levels:
        count = len(result[result[complexity_col] == level])
        print(f"  Level {level}: {count}")

    return result


# ============================================================
# COMPREHENSIVE EVALUATION
# ============================================================

def evaluate_all_metrics(predictions, ground_truths, complexities=None,
                         use_bertscore=True):
    """
    Compute ALL evaluation metrics for a batch of predictions.

    Metrics computed:
      - Exact Match accuracy
      - Word F1 / Precision / Recall
      - BLEU-1, BLEU-2, BLEU-3, BLEU-4
      - ROUGE-L
      - METEOR
      - BERTScore F1 (optional, requires GPU for speed)

    Args:
        predictions:   List of predicted answer strings
        ground_truths: List of ground truth answer strings
        complexities:  Optional list of complexity levels
        use_bertscore: Whether to compute BERTScore (slower)

    Returns:
        dict: All metrics
    """
    assert len(predictions) == len(ground_truths)
    total = len(predictions)

    # --- Per-sample metrics ---
    em_list = []
    f1_list = []
    precision_list = []
    recall_list = []
    bleu1_list = []
    bleu2_list = []
    bleu3_list = []
    bleu4_list = []
    rouge_l_list = []
    meteor_list = []

    for pred, gt in zip(predictions, ground_truths):
        em_list.append(int(compute_exact_match(pred, gt)))
        f1_list.append(compute_word_f1(pred, gt))
        precision_list.append(compute_word_precision(pred, gt))
        recall_list.append(compute_word_recall(pred, gt))

        bleu = compute_bleu_scores(pred, gt)
        bleu1_list.append(bleu['bleu_1'])
        bleu2_list.append(bleu['bleu_2'])
        bleu3_list.append(bleu['bleu_3'])
        bleu4_list.append(bleu['bleu_4'])

        rouge_l_list.append(compute_rouge_l(pred, gt))
        meteor_list.append(compute_meteor(pred, gt))

    # --- Aggregate ---
    results = {
        'total': total,
        'exact_match': sum(em_list),
        'exact_match_pct': sum(em_list) / total * 100,
        'word_f1': np.mean(f1_list) * 100,
        'word_precision': np.mean(precision_list) * 100,
        'word_recall': np.mean(recall_list) * 100,
        'bleu_1': np.mean(bleu1_list) * 100,
        'bleu_2': np.mean(bleu2_list) * 100,
        'bleu_3': np.mean(bleu3_list) * 100,
        'bleu_4': np.mean(bleu4_list) * 100,
        'rouge_l': np.mean(rouge_l_list) * 100,
        'meteor': np.mean(meteor_list) * 100,
        'partial_match_pct': sum(1 for f in f1_list if f >= 0.5) / total * 100,
    }

    # --- BERTScore (batch) ---
    if use_bertscore:
        try:
            bs = compute_bertscore_batch(predictions, ground_truths)
            results['bertscore_f1'] = np.mean(bs['bertscore_f1']) * 100
            results['bertscore_precision'] = np.mean(bs['bertscore_precision']) * 100
            results['bertscore_recall'] = np.mean(bs['bertscore_recall']) * 100
        except Exception as e:
            print(f"  [WARN] BERTScore failed: {e}")
            results['bertscore_f1'] = None

    # --- Per-sample scores (for saving to CSV) ---
    results['_per_sample'] = {
        'exact_match': em_list,
        'word_f1': f1_list,
        'word_precision': precision_list,
        'word_recall': recall_list,
        'bleu_1': bleu1_list,
        'bleu_4': bleu4_list,
        'rouge_l': rouge_l_list,
        'meteor': meteor_list,
    }

    # --- Per-complexity breakdown ---
    if complexities is not None:
        per_complexity = {}
        for level in sorted(set(complexities)):
            idx = [i for i, c in enumerate(complexities) if c == level]
            per_complexity[f'level_{level}'] = {
                'total': len(idx),
                'exact_match_pct': sum(em_list[i] for i in idx) / len(idx) * 100,
                'word_f1': np.mean([f1_list[i] for i in idx]) * 100,
                'word_recall': np.mean([recall_list[i] for i in idx]) * 100,
                'bleu_1': np.mean([bleu1_list[i] for i in idx]) * 100,
                'bleu_4': np.mean([bleu4_list[i] for i in idx]) * 100,
                'rouge_l': np.mean([rouge_l_list[i] for i in idx]) * 100,
                'meteor': np.mean([meteor_list[i] for i in idx]) * 100,
            }
        results['per_complexity'] = per_complexity

    return results


def print_full_results(results, model_name="Model"):
    """Pretty-print all evaluation metrics."""
    print(f"\n{'='*65}")
    print(f"  {model_name} — Evaluation Results")
    print(f"{'='*65}")
    print(f"  Samples:          {results['total']}")
    print(f"  Exact Match:      {results['exact_match']}/{results['total']} "
          f"({results['exact_match_pct']:.1f}%)")
    print(f"  Partial (F1≥0.5): {results['partial_match_pct']:.1f}%")
    print(f"  ─────────────────────────────────────────")
    print(f"  Word F1:          {results['word_f1']:.1f}%")
    print(f"  Word Precision:   {results['word_precision']:.1f}%")
    print(f"  Word Recall:      {results['word_recall']:.1f}%")
    print(f"  ─────────────────────────────────────────")
    print(f"  BLEU-1:           {results['bleu_1']:.1f}%")
    print(f"  BLEU-2:           {results['bleu_2']:.1f}%")
    print(f"  BLEU-3:           {results['bleu_3']:.1f}%")
    print(f"  BLEU-4:           {results['bleu_4']:.1f}%")
    print(f"  ─────────────────────────────────────────")
    print(f"  ROUGE-L:          {results['rouge_l']:.1f}%")
    print(f"  METEOR:           {results['meteor']:.1f}%")
    if results.get('bertscore_f1') is not None:
        print(f"  ─────────────────────────────────────────")
        print(f"  BERTScore F1:     {results['bertscore_f1']:.1f}%")

    if 'per_complexity' in results:
        print(f"\n  Per-Complexity:")
        header = f"    {'Level':<10} {'EM':>6} {'F1':>6} {'Recall':>7} {'BLEU-1':>7} {'BLEU-4':>7} {'ROUGE-L':>8} {'METEOR':>7}"
        print(header)
        print(f"    {'-'*len(header.strip())}")
        for level, s in sorted(results['per_complexity'].items()):
            print(f"    {level:<10} {s['exact_match_pct']:>5.1f}% {s['word_f1']:>5.1f}% "
                  f"{s['word_recall']:>6.1f}% {s['bleu_1']:>6.1f}% {s['bleu_4']:>6.1f}% "
                  f"{s['rouge_l']:>7.1f}% {s['meteor']:>6.1f}%")
    print(f"{'='*65}\n")


# Keep backward compatibility
def evaluate_predictions(predictions, ground_truths, complexities=None):
    """Legacy wrapper — calls evaluate_all_metrics without BERTScore."""
    return evaluate_all_metrics(predictions, ground_truths, complexities,
                                use_bertscore=False)


def print_evaluation_results(results, model_name="Model"):
    """Legacy wrapper."""
    print_full_results(results, model_name)
