"""
Uncertainty estimation methods for Medical VQA.

Three complementary approaches:
  1. Predictive Entropy   — token-level entropy from softmax distributions
  2. MC Dropout           — variance across N stochastic forward passes
  3. Sequence Confidence  — normalized log-probability of generated sequence
"""

import torch
import numpy as np
import re
from collections import Counter


def normalize_text(text):
    """Normalize text for comparison."""
    text = str(text).lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text


# ============================================================
# 1. PREDICTIVE ENTROPY
# ============================================================

def compute_predictive_entropy(model, processor, image, question,
                                max_new_tokens=64, device="cuda"):
    """
    Generate an answer and compute token-level predictive entropy.

    Entropy at each step: H = -Σ p(t) log p(t) over vocabulary.
    Returns the mean entropy across all generated tokens.

    High entropy → model is uncertain (probability spread across tokens).
    Low entropy  → model is confident (probability concentrated on one token).

    Returns:
        dict with keys:
            'prediction': generated answer string
            'entropy_mean': average entropy across tokens
            'entropy_max': maximum single-token entropy
            'entropy_per_token': list of per-token entropies
            'num_tokens': number of generated tokens
    """
    prompt = f"Question: {question} Answer:"
    inputs = processor(
        images=image, text=prompt, return_tensors="pt"
    ).to(device, dtype=torch.float16)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
        )

    # Extract generated token IDs (skip prompt)
    prompt_len = inputs['input_ids'].shape[1]
    generated_ids = outputs.sequences[0][prompt_len:]
    prediction = processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # Compute entropy from logits at each generation step
    entropies = []
    for score in outputs.scores:
        # score shape: (batch=1, vocab_size)
        probs = torch.softmax(score[0], dim=-1)
        # Clamp to avoid log(0)
        log_probs = torch.log(probs.clamp(min=1e-10))
        entropy = -(probs * log_probs).sum().item()
        entropies.append(entropy)

    if not entropies:
        entropies = [0.0]

    return {
        'prediction': prediction,
        'entropy_mean': float(np.mean(entropies)),
        'entropy_max': float(np.max(entropies)),
        'entropy_per_token': entropies,
        'num_tokens': len(entropies),
    }


# ============================================================
# 2. MC DROPOUT
# ============================================================

def _enable_dropout(model):
    """Enable dropout layers for MC Dropout inference."""
    enabled = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()
            enabled += 1
    return enabled


def _disable_dropout(model):
    """Restore all dropout layers to eval mode."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.eval()


def compute_mc_dropout(model, processor, image, question,
                       n_passes=5, max_new_tokens=64, device="cuda"):
    """
    Run N stochastic forward passes with dropout enabled.

    Measures uncertainty as the **lexical variance** across N generated answers:
    - Generate N different answers (dropout creates different paths)
    - Compute pairwise word F1 among all N answers
    - Uncertainty = 1 - mean(pairwise F1)  (high variance = uncertain)

    Also returns the majority answer (most common) as the final prediction.

    Returns:
        dict with keys:
            'prediction': majority/first answer
            'all_predictions': list of N answers
            'mc_uncertainty': 1 - mean pairwise F1 (0=consistent, 1=random)
            'unique_ratio': fraction of unique answers
            'n_passes': number of passes
    """
    prompt = f"Question: {question} Answer:"
    inputs = processor(
        images=image, text=prompt, return_tensors="pt"
    ).to(device, dtype=torch.float16)

    prompt_len = inputs['input_ids'].shape[1]

    # Enable dropout for stochastic inference
    n_enabled = _enable_dropout(model)

    answers = []
    for _ in range(n_passes):
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # greedy, but dropout makes it stochastic
            )
        new_tokens = generated[0][prompt_len:]
        answer = processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        answers.append(answer)

    # Restore eval mode
    _disable_dropout(model)

    # Compute pairwise word F1
    n = len(answers)
    pairwise_f1s = []
    for i in range(n):
        for j in range(i + 1, n):
            f1 = _word_f1(answers[i], answers[j])
            pairwise_f1s.append(f1)

    mean_pairwise_f1 = np.mean(pairwise_f1s) if pairwise_f1s else 1.0
    mc_uncertainty = 1.0 - mean_pairwise_f1

    # Majority answer (most common normalized form)
    normalized = [normalize_text(a) for a in answers]
    most_common = Counter(normalized).most_common(1)[0][0]
    # Find original-case version of the most common
    prediction = answers[normalized.index(most_common)]

    unique_answers = len(set(normalized))

    return {
        'prediction': prediction,
        'all_predictions': answers,
        'mc_uncertainty': float(mc_uncertainty),
        'unique_ratio': unique_answers / n,
        'n_passes': n_passes,
    }


def _word_f1(pred, gt):
    """Quick word F1 for pairwise comparison."""
    p_tok = normalize_text(pred).split()
    g_tok = normalize_text(gt).split()
    if not p_tok and not g_tok:
        return 1.0
    if not p_tok or not g_tok:
        return 0.0
    common = sum((Counter(p_tok) & Counter(g_tok)).values())
    if common == 0:
        return 0.0
    prec = common / len(p_tok)
    rec = common / len(g_tok)
    return 2 * prec * rec / (prec + rec)


# ============================================================
# 3. SEQUENCE CONFIDENCE (Log-probability)
# ============================================================

def compute_sequence_confidence(model, processor, image, question,
                                 max_new_tokens=64, device="cuda"):
    """
    Generate an answer and compute its normalized log-probability.

    Confidence = mean(log p(token_i | token_{<i}, image, question))
    across all generated tokens.

    Higher (less negative) = more confident.
    Lower (more negative) = less confident.

    Returns:
        dict with keys:
            'prediction': generated answer string
            'log_prob_mean': average log-probability per token
            'log_prob_sum': total log-probability
            'confidence': exp(mean log-prob), normalized to [0, 1]
    """
    prompt = f"Question: {question} Answer:"
    inputs = processor(
        images=image, text=prompt, return_tensors="pt"
    ).to(device, dtype=torch.float16)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
        )

    prompt_len = inputs['input_ids'].shape[1]
    generated_ids = outputs.sequences[0][prompt_len:]
    prediction = processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # Compute log-prob of each chosen token
    log_probs = []
    for step_idx, score in enumerate(outputs.scores):
        probs = torch.softmax(score[0], dim=-1)
        if step_idx < len(generated_ids):
            token_id = generated_ids[step_idx].item()
            token_prob = probs[token_id].item()
            log_probs.append(np.log(max(token_prob, 1e-10)))

    if not log_probs:
        log_probs = [-10.0]

    mean_log_prob = float(np.mean(log_probs))
    sum_log_prob = float(np.sum(log_probs))
    confidence = float(np.exp(mean_log_prob))  # [0, 1]

    return {
        'prediction': prediction,
        'log_prob_mean': mean_log_prob,
        'log_prob_sum': sum_log_prob,
        'confidence': confidence,
    }


# ============================================================
# COMBINED UNCERTAINTY ESTIMATION
# ============================================================

def estimate_uncertainty(model, processor, image, question,
                         n_mc_passes=5, max_new_tokens=64, device="cuda"):
    """
    Run all three uncertainty methods and return combined scores.

    Returns:
        dict with all scores + final prediction
    """
    # 1. Entropy + log-prob (single forward pass with scores)
    entropy_result = compute_predictive_entropy(
        model, processor, image, question, max_new_tokens, device
    )

    # 2. Log-prob confidence (reuses the same generation)
    conf_result = compute_sequence_confidence(
        model, processor, image, question, max_new_tokens, device
    )

    # 3. MC Dropout (N passes)
    mc_result = compute_mc_dropout(
        model, processor, image, question, n_mc_passes, max_new_tokens, device
    )

    # Combined uncertainty score (weighted average, normalized to [0, 1])
    # Higher = more uncertain
    entropy_norm = min(entropy_result['entropy_mean'] / 10.0, 1.0)  # typical range 0-10
    mc_unc = mc_result['mc_uncertainty']                              # already [0, 1]
    conf_unc = 1.0 - conf_result['confidence']                       # flip: low conf = high unc

    combined_uncertainty = (0.4 * entropy_norm + 0.3 * mc_unc + 0.3 * conf_unc)

    return {
        'prediction': mc_result['prediction'],
        'prediction_entropy': entropy_result['prediction'],
        'all_mc_predictions': mc_result['all_predictions'],

        # Individual scores
        'entropy_mean': entropy_result['entropy_mean'],
        'entropy_max': entropy_result['entropy_max'],
        'mc_uncertainty': mc_result['mc_uncertainty'],
        'mc_unique_ratio': mc_result['unique_ratio'],
        'seq_confidence': conf_result['confidence'],
        'log_prob_mean': conf_result['log_prob_mean'],

        # Combined
        'combined_uncertainty': float(combined_uncertainty),
    }
