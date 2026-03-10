# Literature Review & Project Justification

## a) Understanding the Topic in Detail

**Visual Question Answering (VQA) for Medical Images with Hallucination Control**

Medical VQA sits at the intersection of computer vision, natural language processing, and clinical informatics. The task involves a system receiving a medical image (in our case, gastrointestinal endoscopy images) and a natural language question about it, and generating an accurate, clinically relevant answer.

The critical challenge in medical VQA is **hallucination** — when a model confidently generates factually incorrect medical information. Unlike general-domain VQA where a wrong answer might be inconsequential, medical hallucinations can lead to misdiagnoses, inappropriate treatments, or missed findings. There are two types of medical hallucinations:

- **Consistency Hallucination (Misdiagnosis)**: The model fabricates findings not present in the image. For example, in our baseline experiments, BLIP-2 identified a "urethral sphincter" in a GI endoscopy image — a completely wrong organ system.
- **Completeness Hallucination (Missed Diagnosis)**: The model fails to report actual findings. For instance, BLIP-2 claimed "all polyps have been successfully excised" when residual polyps were still present.

Our project addresses this by building a system that is **uncertainty-aware** — it quantifies its own confidence and *refuses to answer* when uncertain, rather than guessing. This paradigm shift from "always answer" to "answer safely" is known as **selective prediction** or **abstention**, and is a top-tier research direction in Safe AI.

---

## b) Literature in the Domain

The literature spans three interrelated areas:

### 1. Medical Visual Question Answering
- **VQA-RAD** (Lau et al., 2018): One of the earliest medical VQA datasets with radiology images. Limited in size (~3,500 QA pairs) and question diversity.
- **PathVQA** (He et al., 2020): Pathology image VQA with ~32,000 QA pairs. Focus on histopathology.
- **SLAKE** (Liu et al., 2021): Bilingual medical VQA dataset (English/Chinese) with semantic annotations.
- **Kvasir-VQA** (Gautam et al., 2023): GI endoscopy VQA with ~58,000 QA pairs from HyperKvasir and Kvasir-Instrument datasets.
- **Kvasir-VQA-x1** (Gautam et al., 2025): Major expansion to 159,549 QA pairs with complexity stratification (3 levels), augmented images, and naturalized clinical queries.

### 2. Multimodal Vision-Language Models
- **ViLT** (Kim et al., 2021): Vision-and-Language Transformer — lightweight, no region features.
- **BLIP-2** (Li et al., 2023): Bootstrapping Language-Image Pre-training with a Q-Former bridge. Strong zero-shot performance but prone to hallucination on domain-specific data.
- **LLaVA / LLaVA-Med** (Li et al., 2023/2024): Large Language and Vision Assistant; LLaVA-Med is fine-tuned for biomedical domains.
- **PaliGemma 2** (Google, 2024): Multimodal model used in Medico 2025 challenge submissions with LoRA fine-tuning.
- **BiomedCLIP** (Zhang et al., 2023): Domain-specific CLIP variant trained on biomedical literature and images.

### 3. Hallucination Detection & Uncertainty Estimation
- **VASE** — Vision-Amplified Semantic Entropy (2025): Incorporates image transformations to enhance uncertainty estimation for hallucination detection in medical VQA.
- **HEDGE** (2025): Unified framework combining visual perturbations, semantic clustering, and uncertainty metrics.
- **MediHallDetector** (ICLR 2025): First dedicated hallucination detection model for medical LVLMs with the MediHall Score metric.
- **MedAbstain** (2026): Benchmark for abstention in medical QA, integrating conformal prediction for uncertainty assessment.
- **Geifman & El-Yaniv** (2017): Foundational work on selective prediction — Risk-Coverage curves and SelectiveNet.

---

## c) Various Works Carried Out with the Kvasir-VQA-x1 Dataset

| Work | Authors | Approach | Key Results |
|------|---------|----------|-------------|
| **Medico 2025 Challenge — PaliGemma 2** | Challenge participants | Fine-tuned PaliGemma 2 with LoRA + Grad-CAM explanations + confidence scores | BLEU: 0.427, METEOR: 0.66 on Complexity 3 questions |
| **Fine-tuned PaliGemma 3B** | Challenge participants | PEFT with LoRA, 4-bit quantization, geometric + color augmentations | ROUGE-1: 0.723, METEOR: 0.70 on private test |
| **Florence Model** | Challenge participants | Large-scale multimodal transformer with domain-specific endoscopic augmentations | BLEU: 0.160, ROUGE-L: 0.880, METEOR: 0.490 |
| **Disease-Guided VQA** | Zeshan Khan et al., 2025 | Integrates 23-class GI disease classifier probabilities with VQA features | Improved context-aware answering across complexity levels |
| **Original Kvasir-VQA-x1 paper** | Gautam et al., 2025 | Dataset introduction with benchmark evaluation of existing VQA methods | Established complexity-stratified evaluation protocol |

The dataset has also been used for image captioning, synthetic medical image generation, object detection, and GI disease classification tasks.

---

## d) Drawbacks Identified in the Current Literature

### 1. No Uncertainty Awareness
All existing approaches on Kvasir-VQA-x1 optimize **solely for accuracy**. None implement mechanisms to detect when the model is uncertain or likely to hallucinate. A model that achieves 70% accuracy but is *equally confident* on the 30% it gets wrong is clinically dangerous.

### 2. Hallucination Goes Unaddressed
Current works do not quantify or mitigate hallucinations. As our baseline experiments demonstrated, generic models fabricate wrong procedures ("laparoscopic cholecystectomy" for colonoscopy), wrong organs ("urethral sphincter" in GI), and wrong modalities ("abdominal ultrasonography" for endoscopy) — all with full confidence.

### 3. No Abstention Mechanism
No existing work on Kvasir-VQA-x1 implements selective prediction. In clinical settings, saying "I don't know — please consult a doctor" is far safer than confidently guessing wrong. The MedAbstain benchmark (2026) shows that even state-of-the-art medical LLMs fail to abstain when uncertain.

### 4. Poor Calibration
Current models are poorly calibrated — their confidence scores do not correlate with actual correctness. A model may output 95% confidence on a hallucinated answer, making it impossible for downstream systems (or clinicians) to trust or filter predictions.

### 5. Lack of Complexity-Aware Evaluation
Most works report aggregate metrics. They do not analyze how performance degrades across complexity levels, missing the insight that multi-step reasoning questions (Level 3) are far more prone to hallucination than simple factual queries (Level 1).

### 6. Limited Explainability of Failures
While Medico 2025 Subtask 2 addresses explainability of correct answers, no work explains *why* a model hallucinates or *when* it is likely to fail. Understanding failure modes is critical for clinical trust.

---

## e) Novelty of Our Work

Our work introduces **three novel contributions** to the Kvasir-VQA-x1 literature:

### 1. Uncertainty-Aware Training Objective
Unlike all existing approaches that use standard cross-entropy loss, we propose incorporating uncertainty directly into the training process. Rather than just adding a post-hoc entropy threshold, the model *learns when to abstain* during training. This can be achieved through:
- An auxiliary "abstain" class in the output space
- Evidential deep learning loss (Dirichlet-based uncertainty)
- Uncertainty-regularized training that penalizes confident-but-wrong predictions

### 2. Confidence-Based Abstention Mechanism
The system outputs **"Requires Doctor Review"** when the model's predictive entropy exceeds a learned threshold, rather than producing a potentially harmful guess. This implements the concept of selective prediction in the medical VQA domain — a first for the Kvasir-VQA-x1 dataset.

### 3. Multi-Metric Safety Evaluation
We evaluate not just accuracy, but also:
- **Risk-Coverage Curves**: Accuracy as a function of the fraction of questions the model chooses to answer
- **Expected Calibration Error (ECE)**: Whether the model's confidence is meaningful
- **AUROC of Uncertainty**: How well uncertainty predicts incorrectness
- **Per-Complexity Hallucination Analysis**: Breakdown of failure modes across difficulty levels

This shifts the evaluation paradigm from *"how accurate is the model?"* to *"how safe is the model?"*

---

## f) Baseline Models in This Domain

### Models Used on Kvasir-VQA-x1 Specifically
| Model | Type | Key Characteristic |
|-------|------|-------------------|
| **PaliGemma 2** | Vision-Language Model | Fine-tuned with LoRA; best reported results on Medico 2025 |
| **PaliGemma 3B** | Vision-Language Model | PEFT with 4-bit quantization |
| **Florence** | Multimodal Transformer | Domain-specific augmentations |
| **Disease-Guided VQA** | Hybrid Classifier + VQA | Integrates GI disease classifier output |

### General Medical VQA Baselines
| Model | Type | Key Characteristic |
|-------|------|-------------------|
| **BLIP-2** | Vision-Language Model | Strong zero-shot capability; our baseline (0% exact match, 27% F1 on Kvasir-VQA-x1) |
| **LLaVA-Med** | Medical LLM | Fine-tuned LLaVA on biomedical data |
| **BiomedCLIP** | Domain CLIP | Trained on PubMed image-text pairs |
| **ViLT** | Vision-Language Transformer | Lightweight, no region features |
| **Qwen2.5-VL / MedGemma** | Recent VLMs | Listed as baselines in related MICCAI challenges |

### Our Baseline Results (BLIP-2 Zero-Shot on Kvasir-VQA-x1)
| Metric | Level 1 | Level 2 | Level 3 | Overall |
|--------|---------|---------|---------|---------|
| Exact Match | 0% | 0% | 0% | 0% |
| Word F1 | 18.9% | 27.0% | 35.1% | 27.0% |

---

## g) How the Proposed Approach Overcomes Limitations

| Limitation | How We Overcome It |
|-----------|-------------------|
| **No uncertainty awareness** | We integrate uncertainty estimation directly into the model through entropy-based methods (predictive entropy, MC Dropout) and train the model with an uncertainty-aware objective, so it *learns* its own knowledge boundaries. |
| **Unchecked hallucinations** | Our abstention mechanism catches high-uncertainty predictions *before* they reach the clinician. When entropy exceeds a calibrated threshold, the system outputs "Requires Doctor Review" instead of a potentially harmful hallucinated answer. |
| **No abstention capability** | We implement selective prediction — the model has the option to refuse answering. This is evaluated via Risk-Coverage curves, which show the accuracy-coverage tradeoff: *"If the model only answers its most confident 50% of questions, how accurate is it?"* |
| **Poor calibration** | We apply calibration techniques (temperature scaling, Platt scaling) and evaluate using ECE. A well-calibrated model's 80% confidence means it's correct ~80% of the time — making confidence scores clinically actionable. |
| **No complexity-aware analysis** | We evaluate all metrics stratified by complexity level (1–3), revealing where hallucinations concentrate and where abstention is most beneficial. Our baseline already shows L1 (18.9% F1) vs L3 (35.1% F1) performance gaps. |
| **No failure explanation** | We analyze hallucination patterns (wrong organ, wrong procedure, contradicting image) and correlate them with uncertainty scores, identifying *which types of questions* the model should abstain from. |

### The Key Paradigm Shift
> Existing work asks: *"How can we make the model more accurate?"*
>
> Our work asks: *"How can we make the model **safer** — accurate when it answers, and honest when it can't?"*

This framing — optimizing for safety rather than just accuracy — is directly aligned with the growing field of **Trustworthy AI** in healthcare, which is a top-tier research direction in 2025–2026.

---

## h) Technical Methodology — How We Achieve This

Our system is built as a **4-stage pipeline**. Each stage feeds into the next.

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TRAINING PHASE                               │
│                                                                     │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────────────────┐   │
│  │ Kvasir   │───▶│ Fine-tune    │───▶│ Uncertainty-Aware       │   │
│  │ VQA-x1   │    │ VLM with     │    │ Training Objective      │   │
│  │ Dataset  │    │ LoRA/QLoRA   │    │ (modified loss function) │   │
│  └──────────┘    └──────────────┘    └─────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       INFERENCE PHASE                                │
│                                                                     │
│  ┌───────┐   ┌─────────┐   ┌──────────────┐   ┌────────────────┐  │
│  │ Image │──▶│ Model   │──▶│ Uncertainty  │──▶│ Abstention     │  │
│  │   +   │   │ Forward │   │ Estimation   │   │ Decision       │  │
│  │ Query │   │ Pass(es)│   │ (Entropy/MC) │   │ (Threshold)    │  │
│  └───────┘   └─────────┘   └──────────────┘   └───────┬────────┘  │
│                                                        │           │
│                                          ┌─────────────┼──────┐    │
│                                          ▼             ▼      │    │
│                                     Low Entropy    High Entropy│    │
│                                     ┌────────┐    ┌──────────┐│    │
│                                     │ Answer │    │ "Requires││    │
│                                     │ + Conf │    │  Doctor  ││    │
│                                     │ Score  │    │  Review" ││    │
│                                     └────────┘    └──────────┘│    │
│                                          └────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

---

### Stage 1: Fine-Tuning a Base VLM with LoRA

**Why fine-tune instead of training from scratch?**
Large vision-language models (VLMs) like BLIP-2 or PaliGemma have already learned general visual-linguistic representations from billions of image-text pairs. Training from scratch on 143K QA pairs would be insufficient. Instead, we **adapt** the pre-trained model to the GI endoscopy domain through parameter-efficient fine-tuning.

**How LoRA works:**
Low-Rank Adaptation (LoRA) freezes all original model weights and injects small trainable rank-decomposition matrices into the attention layers. Instead of updating a weight matrix W (e.g., 4096 × 4096 = 16.7M parameters), we decompose the update as:

```
W_new = W_original + ΔW
ΔW = A × B     where A is (4096 × r), B is (r × 4096), and r = 16 or 32
```

This reduces trainable parameters from ~3 billion to ~10-20 million (< 1% of the model), making fine-tuning feasible on a single consumer GPU (T4/A100 on Colab).

**What gets fine-tuned:**
- The Q-Former / cross-attention layers (where visual and text features interact)
- The language model's attention projection matrices
- NOT the vision encoder (frozen — it already extracts good image features)

**Training setup:**
```python
# Pseudocode for LoRA fine-tuning
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,                    # Rank of decomposition
    lora_alpha=32,           # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Which attention matrices to adapt
    lora_dropout=0.1,        # Dropout for regularization
    task_type="CAUSAL_LM"
)

model = get_peft_model(base_model, lora_config)
# Only ~10M trainable params out of 3B total
```

**Training data:**
- Input: `(endoscopy_image, question)` pairs from Kvasir-VQA-x1 training set
- Target: Ground truth answers
- Loss: Standard cross-entropy on the generated answer tokens (initially — modified in Stage 2)
- Epochs: 3-5 with early stopping on validation F1
- Batch size: 4-8 (with gradient accumulation for effective batch size 32)

**Expected outcome:** The fine-tuned model should dramatically improve over the zero-shot baseline (currently 0% exact match, 27% F1) because it learns GI-specific vocabulary (polyps, z-line, erythema), clinical reasoning patterns, and the dataset's answer format.

---

### Stage 2: Uncertainty Estimation

This is the **core technical novelty**. We implement three complementary methods and compare them:

#### Method 1: Predictive Entropy (Token-Level)

**Intuition:** When a model is uncertain, the probability distribution over next tokens is spread out (high entropy). When confident, it concentrates on a single token (low entropy).

**How it works:**
At each decoding step, the model produces a probability distribution over its vocabulary (~32,000 tokens). We compute the Shannon entropy of this distribution:

```
H(y_t | y_{<t}, x) = -Σ p(token_i) × log(p(token_i))
```

For the full answer, we aggregate across all generated tokens:

```
H_total = (1/T) × Σ_{t=1}^{T} H(y_t)    # Mean entropy across T tokens
```

**Interpretation:**
- Low H_total (~0.1-0.5): Model is very sure → **allow answer**
- High H_total (~2.0+): Model is uncertain → **abstain**

**Implementation:**
```python
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=64,
        return_dict_in_generate=True,
        output_scores=True,           # Returns logits at each step
    )

# Compute entropy from logits
entropies = []
for step_logits in outputs.scores:
    probs = torch.softmax(step_logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
    entropies.append(entropy)

mean_entropy = torch.stack(entropies).mean()  # Uncertainty score
```

**Advantage:** Single forward pass, computationally cheap.
**Limitation:** Only captures surface-level uncertainty; model may be confidently wrong.

---

#### Method 2: Monte Carlo (MC) Dropout

**Intuition:** Run the same input through the model N times with dropout enabled. If the model gives different answers each time, it's uncertain. If all N runs agree, it's confident.

**How it works:**
1. Enable dropout at inference time (normally dropout is OFF during inference)
2. Run N forward passes (e.g., N=10) on the same `(image, question)` pair
3. Collect N different answer predictions
4. Measure how much the predictions disagree

**Disagreement metrics:**
```
# Lexical diversity — how many unique answers out of N runs
diversity = |unique_answers| / N

# Semantic entropy — cluster semantically similar answers, then compute entropy
# over cluster distribution

# Token-level variance — average variance of token probabilities across N runs
variance = (1/N) × Σ Var(p(token_i)) across N runs
```

**Implementation:**
```python
def enable_mc_dropout(model):
    """Keep dropout layers active during inference"""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()  # Keep dropout ON

enable_mc_dropout(model)

predictions = []
for _ in range(N):
    output = model.generate(**inputs, do_sample=True, temperature=0.7)
    prediction = processor.decode(output[0], skip_special_tokens=True)
    predictions.append(prediction)

# Measure disagreement
unique_answers = set(predictions)
uncertainty = len(unique_answers) / N  # 0.1 = consistent, 1.0 = all different
```

**Advantage:** Captures model-level uncertainty, not just token-level; works well for detecting hallucinations.
**Limitation:** N times slower (N forward passes). With N=10 and 5s/pass, each sample takes ~50s.

---

#### Method 3: Uncertainty-Aware Training (Evidential / Modified Loss)

**Intuition:** Instead of just post-hoc uncertainty (Methods 1 and 2 above), we modify the training process itself so the model *learns* to produce calibrated confidence alongside its answers.

**Approach A — Auxiliary Confidence Head:**
Add a small neural network head that predicts a confidence score (0-1) alongside the answer. During training, we penalize predictions where confidence is high but the answer is wrong.

```
Total Loss = λ₁ × CE_loss(answer) + λ₂ × confidence_penalty

Where:
  confidence_penalty = -[correct × log(c) + (1-correct) × log(1-c)]
  correct = 1 if prediction matches ground truth, 0 otherwise
  c = sigmoid(confidence_head(hidden_state))
```

**Approach B — Evidential Deep Learning:**
Replace the standard softmax output with a Dirichlet distribution. Instead of outputting point probabilities, the model outputs *parameters of a distribution over probabilities*, encoding both the prediction and the model's uncertainty about that prediction.

```
Standard model:  p(answer | image, question) = softmax(logits)
Evidential model: α(answer | image, question) = softplus(logits) + 1

Uncertainty = K / Σ(α_k)   where K = number of classes
                             High Σ(α) → low uncertainty (many "evidence")
                             Low Σ(α) → high uncertainty (little "evidence")
```

**Approach C — Regularized Cross-Entropy:**
Modify the standard cross-entropy loss with an entropy regularization term that penalizes over-confidence:

```
Loss = CE_loss + β × max(0, confidence - margin) for incorrect predictions
```

This tells the model: "When you're wrong, you should NOT be confident."

**Advantage:** The model is uncertainty-aware from the ground up; no post-hoc calibration needed.
**Limitation:** More complex to implement; requires careful hyperparameter tuning (λ, β values).

---

### Stage 3: Abstention Decision

Given an uncertainty score U from any of the above methods, we need a **decision rule**:

```
if U < threshold:
    return model_answer, confidence_score
else:
    return "This question requires further review by a medical professional."
```

**How to set the threshold?**

We use the **validation set** to find the optimal threshold by plotting the Risk-Coverage curve:

```python
# For each possible threshold τ
for tau in np.linspace(0, max_entropy, 100):
    # Coverage: fraction of questions answered
    answered = [s for s in samples if s.uncertainty < tau]
    coverage = len(answered) / len(samples)
    
    # Risk: error rate on answered questions
    errors = [s for s in answered if s.prediction != s.ground_truth]
    risk = len(errors) / len(answered)
    
    # We want: high coverage + low risk
    # The curve shows the tradeoff
```

**Ideal behavior:**
- At coverage = 100% (answer everything): risk = baseline error rate
- At coverage = 50% (answer only confident half): risk should drop significantly
- The steeper the drop, the better the uncertainty estimation

**Clinical deployment setting:**
In a real clinical system, the threshold would be set by the hospital's risk tolerance. For example:
- **Conservative** (τ = low): Model only answers when very confident → high accuracy but low coverage (30-40%)
- **Moderate** (τ = medium): Balanced → moderate accuracy and coverage (60-70%)
- **Permissive** (τ = high): Model answers most questions → lower accuracy but high coverage (90%+)

---

### Stage 4: Evaluation Framework

We evaluate on **safety**, not just accuracy:

#### 1. Risk-Coverage Curve (Primary Metric)
```
        Accuracy
    1.0 ┤████
        │   ████
    0.8 ┤      ████         ← Our model (steep curve = good)
        │         ████
    0.6 ┤            ████
        │     ─────────────  ← Baseline (flat = no useful uncertainty)
    0.4 ┤
        │
    0.2 ┤
        └────────────────────
        0%   25%  50%  75% 100%
              Coverage →
```

**What it shows:** As we decrease coverage (i.e., the model abstains more), how much does accuracy improve? A steep curve = the model's uncertainty is meaningful. A flat curve = uncertainty is random noise.

The **Area Under the Risk-Coverage Curve (AURC)** is a single-number summary. Lower AURC = better.

#### 2. Expected Calibration Error (ECE)
```
ECE = Σ (|bin_count| / N) × |accuracy(bin) - confidence(bin)|
```
This measures whether the model's confidence is reliable. An ECE of 0 means perfect calibration (80% confidence → correct 80% of the time).

#### 3. AUROC of Uncertainty
Treats "is this prediction correct?" as a binary classification problem where uncertainty is the classifier's score. High AUROC (~0.8+) means uncertainty reliably predicts when the model will be wrong.

#### 4. Per-Complexity Analysis
All metrics are computed separately for Level 1, 2, and 3 questions. We expect:
- Level 1 (simple): High accuracy, low abstention rate
- Level 3 (complex): Lower accuracy, higher abstention rate, but *among answered questions, accuracy should still be high* (that's the point)

#### 5. Hallucination Case Study
Manual analysis of 30-50 predictions categorized as:
- **True Positive**: Correct answer, low uncertainty ✓
- **True Negative**: Wrong answer, high uncertainty → correctly abstained ✓
- **False Positive**: Wrong answer, low uncertainty → hallucination slipped through ✗
- **False Negative**: Correct answer, high uncertainty → unnecessarily abstained ✗

---

### Putting It All Together — End-to-End Example

```
Input:
  Image: GI endoscopy showing erythematous mucosa with a small polyp
  Question: "Is there any abnormality visible in the image?"

Step 1: Forward pass through fine-tuned model
  → Raw answer: "Yes, erythematous mucosa with a small polyp is visible"
  → Token-level entropy: 0.32 (low)

Step 2: MC Dropout (10 runs)
  → 9/10 runs: "Yes, erythematous mucosa..." (consistent)
  → 1/10 runs: "Yes, a polyp is visible"
  → MC uncertainty: 0.15 (low — answers agree)

Step 3: Abstention decision
  → Combined uncertainty: 0.24 (below threshold τ = 1.5)
  → Decision: ANSWER

Output: "Yes, erythematous mucosa with a small polyp is visible"
        Confidence: 87%

────────────────────────────────────────────────

Input:
  Image: Blurry/ambiguous endoscopy image
  Question: "What is the size of the polyp and is there a green/black box artefact?"

Step 1: Forward pass
  → Raw answer: "The polyp is about 5mm with no artefact visible"
  → Token-level entropy: 2.8 (HIGH)

Step 2: MC Dropout (10 runs)
  → Run 1: "The polyp is about 5mm..."
  → Run 2: "No polyp is visible in the image"
  → Run 3: "The polyp measures 10-15mm..."
  → ... (all different)
  → MC uncertainty: 0.8 (HIGH — answers disagree)

Step 3: Abstention decision
  → Combined uncertainty: 1.8 (above threshold τ = 1.5)
  → Decision: ABSTAIN

Output: "This question requires further review by a medical professional."
        Reason: Model uncertainty exceeds safety threshold.
```

---

### Technology Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Base Model | BLIP-2 or PaliGemma 2 | Best available VLMs with medical VQA applicability |
| Fine-tuning | LoRA via HuggingFace PEFT | Parameter-efficient; feasible on Colab T4/A100 |
| Uncertainty | Custom PyTorch modules | Entropy, MC Dropout, confidence head |
| Training | PyTorch + HuggingFace Accelerate | Mixed-precision (fp16), gradient accumulation |
| Evaluation | scikit-learn + custom metrics | Risk-Coverage curves, ECE, AUROC |
| Compute | Google Colab (T4/A100) | Free/affordable GPU access |
| Dataset | Kvasir-VQA-x1 via HuggingFace | 159K QA pairs, 6.4K images |
