"""
Abstention mechanism for uncertainty-aware Medical VQA.

Decides when the model should refuse to answer based on
uncertainty scores, and provides threshold tuning utilities.
"""

import numpy as np


def apply_abstention(predictions, uncertainty_scores, threshold):
    """
    Apply abstention: if uncertainty > threshold, abstain.

    Args:
        predictions: list of predicted answer strings
        uncertainty_scores: list of float uncertainty scores (higher = more uncertain)
        threshold: float threshold — abstain if score > threshold

    Returns:
        dict with:
            'final_answers': list (answer or "ABSTAIN")
            'abstained_indices': list of indices where model abstained
            'answered_indices': list of indices where model answered
            'coverage': fraction of questions answered
            'abstention_rate': fraction of questions abstained
    """
    assert len(predictions) == len(uncertainty_scores)

    final_answers = []
    abstained = []
    answered = []

    for i, (pred, unc) in enumerate(zip(predictions, uncertainty_scores)):
        if unc > threshold:
            final_answers.append("I am not confident enough to answer this question.")
            abstained.append(i)
        else:
            final_answers.append(pred)
            answered.append(i)

    total = len(predictions)
    return {
        'final_answers': final_answers,
        'abstained_indices': abstained,
        'answered_indices': answered,
        'coverage': len(answered) / total if total > 0 else 0,
        'abstention_rate': len(abstained) / total if total > 0 else 0,
    }


def tune_threshold(uncertainty_scores, correctness_scores, 
                   target_coverage=0.80, n_steps=100):
    """
    Find the optimal uncertainty threshold.

    Sweeps thresholds from min to max uncertainty, and finds the one
    that maximizes accuracy on answered questions while maintaining
    at least target_coverage.

    Args:
        uncertainty_scores: list of float uncertainty values
        correctness_scores: list of float quality scores (e.g., word F1)
        target_coverage: minimum fraction of questions to answer (default 0.80)
        n_steps: number of threshold values to try

    Returns:
        dict with:
            'optimal_threshold': best threshold value
            'optimal_coverage': coverage at that threshold
            'optimal_selective_accuracy': accuracy on answered questions
            'all_thresholds': list of tried thresholds
            'all_coverages': corresponding coverages
            'all_selective_accuracies': corresponding selective accuracies
    """
    scores = np.array(uncertainty_scores)
    correct = np.array(correctness_scores)

    min_unc = float(scores.min())
    max_unc = float(scores.max())
    thresholds = np.linspace(min_unc, max_unc, n_steps)

    coverages = []
    selective_accs = []

    for t in thresholds:
        answered_mask = scores <= t
        n_answered = answered_mask.sum()
        coverage = n_answered / len(scores)

        if n_answered > 0:
            sel_acc = correct[answered_mask].mean()
        else:
            sel_acc = 0.0

        coverages.append(float(coverage))
        selective_accs.append(float(sel_acc))

    # Find best threshold that meets coverage target
    best_idx = None
    best_acc = -1
    for i, (cov, acc) in enumerate(zip(coverages, selective_accs)):
        if cov >= target_coverage and acc > best_acc:
            best_acc = acc
            best_idx = i

    # Fallback: if no threshold meets target, pick highest coverage one
    if best_idx is None:
        best_idx = n_steps - 1  # highest threshold = maximum coverage

    return {
        'optimal_threshold': float(thresholds[best_idx]),
        'optimal_coverage': coverages[best_idx],
        'optimal_selective_accuracy': selective_accs[best_idx],
        'all_thresholds': thresholds.tolist(),
        'all_coverages': coverages,
        'all_selective_accuracies': selective_accs,
    }


def analyze_abstention_by_complexity(uncertainty_scores, complexities, threshold):
    """
    Break down abstention rates by complexity level.

    Returns:
        dict: {level: {'total', 'abstained', 'abstention_rate'}}
    """
    result = {}
    for unc, comp in zip(uncertainty_scores, complexities):
        key = f"level_{comp}"
        if key not in result:
            result[key] = {'total': 0, 'abstained': 0}
        result[key]['total'] += 1
        if unc > threshold:
            result[key]['abstained'] += 1

    for key in result:
        t = result[key]['total']
        a = result[key]['abstained']
        result[key]['abstention_rate'] = a / t if t > 0 else 0
        result[key]['coverage'] = (t - a) / t if t > 0 else 0

    return result
