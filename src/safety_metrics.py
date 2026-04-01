"""
Safety evaluation metrics for uncertainty-aware Medical VQA.

Metrics:
  - Risk-Coverage curve
  - AUROC (uncertainty separates correct/incorrect)
  - ECE (Expected Calibration Error)
  - Selective Accuracy at various coverage levels
"""

import numpy as np


def compute_risk_coverage(uncertainty_scores, correctness_scores, n_points=50):
    """
    Compute the Risk-Coverage curve.

    As we increase the threshold (answering more questions = higher coverage),
    the risk (error rate on answered questions) should increase.

    A good uncertainty estimator keeps risk low even at moderate coverage.

    Args:
        uncertainty_scores: list of floats (higher = more uncertain)
        correctness_scores: list of floats (e.g., word F1, 0-1)
        n_points: number of coverage levels to compute

    Returns:
        dict with:
            'coverages': list of coverage values (0 to 1)
            'risks': list of risk (1 - accuracy) at each coverage
            'accuracies': list of selective accuracy at each coverage
            'auc_risk': area under risk-coverage curve (lower is better)
    """
    unc = np.array(uncertainty_scores)
    corr = np.array(correctness_scores)

    # Sort by uncertainty (ascending: most confident first)
    sorted_idx = np.argsort(unc)
    sorted_corr = corr[sorted_idx]

    coverages = []
    risks = []
    accuracies = []

    for n in range(1, len(sorted_corr) + 1):
        coverage = n / len(sorted_corr)
        answered = sorted_corr[:n]
        acc = float(np.mean(answered))
        risk = 1.0 - acc

        coverages.append(coverage)
        risks.append(risk)
        accuracies.append(acc)

    # AUC of risk-coverage (lower is better)
    auc_risk = float(np.trapz(risks, coverages))

    return {
        'coverages': coverages,
        'risks': risks,
        'accuracies': accuracies,
        'auc_risk': auc_risk,
    }


def compute_auroc(uncertainty_scores, is_correct, threshold=0.5):
    """
    Compute AUROC: how well uncertainty separates correct from incorrect.

    A correct prediction has is_correct=True (or F1 >= threshold).
    AUROC > 0.5 means uncertainty is informative.
    AUROC = 1.0 means perfect separation.

    Args:
        uncertainty_scores: list of floats
        is_correct: list of bools or floats (F1 scores)
        threshold: F1 threshold to define "correct" (default 0.5)

    Returns:
        float: AUROC score
    """
    unc = np.array(uncertainty_scores)

    if isinstance(is_correct[0], (bool, np.bool_)):
        labels = np.array(is_correct, dtype=int)
    else:
        # Convert F1 scores to binary
        labels = (np.array(is_correct) >= threshold).astype(int)

    # Handle edge cases
    if labels.sum() == 0 or labels.sum() == len(labels):
        return 0.5  # No discrimination possible

    # Manual AUROC (to avoid sklearn dependency in notebook)
    # For each incorrect sample, count how many correct samples have lower uncertainty
    incorrect_idx = np.where(labels == 0)[0]
    correct_idx = np.where(labels == 1)[0]

    n_pairs = len(incorrect_idx) * len(correct_idx)
    if n_pairs == 0:
        return 0.5

    concordant = 0
    for i in incorrect_idx:
        for j in correct_idx:
            if unc[i] > unc[j]:
                concordant += 1
            elif unc[i] == unc[j]:
                concordant += 0.5

    return float(concordant / n_pairs)


def compute_ece(confidence_scores, correctness_scores, n_bins=10):
    """
    Compute Expected Calibration Error (ECE).

    Measures how well confidence aligns with actual accuracy.
    ECE = 0 means perfectly calibrated.

    Bins predictions by confidence, then compares mean confidence
    vs actual accuracy in each bin.

    Args:
        confidence_scores: list of floats in [0, 1] (higher = more confident)
        correctness_scores: list of floats (binary or F1 in [0, 1])
        n_bins: number of confidence bins

    Returns:
        dict with:
            'ece': float, the ECE value
            'bin_confidences': mean confidence per bin
            'bin_accuracies': actual accuracy per bin
            'bin_counts': samples per bin
    """
    conf = np.array(confidence_scores)
    corr = np.array(correctness_scores)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_confidences = []
    bin_accuracies = []
    bin_counts = []

    ece = 0.0
    total = len(conf)

    for i in range(n_bins):
        lo = bin_boundaries[i]
        hi = bin_boundaries[i + 1]

        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)

        n_in_bin = mask.sum()
        if n_in_bin == 0:
            bin_confidences.append(0.0)
            bin_accuracies.append(0.0)
            bin_counts.append(0)
            continue

        avg_conf = float(conf[mask].mean())
        avg_acc = float(corr[mask].mean())

        bin_confidences.append(avg_conf)
        bin_accuracies.append(avg_acc)
        bin_counts.append(int(n_in_bin))

        ece += (n_in_bin / total) * abs(avg_acc - avg_conf)

    return {
        'ece': float(ece),
        'bin_confidences': bin_confidences,
        'bin_accuracies': bin_accuracies,
        'bin_counts': bin_counts,
    }


def compute_selective_accuracy(correctness_scores, uncertainty_scores, 
                                coverage_levels=None):
    """
    Compute accuracy at various coverage levels.

    At coverage=0.5, we answer only the 50% most confident questions.

    Args:
        correctness_scores: list of floats (F1 or binary)
        uncertainty_scores: list of floats
        coverage_levels: list of coverage fractions (default: [0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    Returns:
        dict: {coverage: selective_accuracy}
    """
    if coverage_levels is None:
        coverage_levels = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    unc = np.array(uncertainty_scores)
    corr = np.array(correctness_scores)
    sorted_idx = np.argsort(unc)
    sorted_corr = corr[sorted_idx]

    result = {}
    total = len(sorted_corr)
    for cov in coverage_levels:
        n = max(1, int(cov * total))
        acc = float(sorted_corr[:n].mean())
        result[cov] = acc

    return result
