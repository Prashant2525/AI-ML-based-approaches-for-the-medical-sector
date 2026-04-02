# Uncertainty-Aware Visual Question Answering on Kvasir-VQA-x1

## AI/ML-based Approaches for the Medical Sector

**Capstone Project (12 Credits) | 8th Semester, B.Tech CSE**
**Project ID:** CAPSTONE_2022_22172_4

---

## About the Project

This project develops a **Visual Question Answering (VQA)** system for gastrointestinal endoscopy images that prioritizes **safety over accuracy**. Generic multimodal models (like GPT-4V, BLIP-2) are known to hallucinate on medical data — confidently producing incorrect clinical information. Our system addresses this by implementing an **uncertainty-aware abstention mechanism** that refuses to answer when confidence is low, rather than fabricating potentially dangerous medical responses.

### Research Question
> *How effective are uncertainty-aware training objectives in reducing hallucinations for VQA tasks on the Kvasir-VQA-x1 dataset?*

### Key Innovation
Instead of optimizing solely for accuracy, we optimize for **safety** through:
- **Confidence threshold mechanism** — if the model's internal entropy is high, it outputs *"Requires Doctor Review"* instead of guessing
- **Uncertainty estimation** — predictive entropy, MC Dropout variance, and sequence log-probability
- **Risk-Coverage evaluation** — measuring accuracy when the model is allowed to decline answering uncertain questions

---

## Dataset

We use the [**Kvasir-VQA-x1**](https://huggingface.co/datasets/SimulaMet/Kvasir-VQA-x1) dataset — a large-scale multimodal benchmark for medical VQA in gastrointestinal endoscopy.

| Statistic | Value |
|-----------|-------|
| Total QA pairs | 159,549 |
| Training QA pairs | 143,594 |
| Test QA pairs | 15,955 |
| Unique GI endoscopy images | 6,449 |
| Complexity levels | 3 (simple → complex reasoning) |
| Question classes | 3,892 unique categories |
| Avg question length | 13.7 words |
| Avg answer length | 10.1 words |

**Complexity Distribution:**
- Level 1 (simple): 54,856 pairs (34.4%)
- Level 2 (medium): 52,349 pairs (32.8%)
- Level 3 (complex): 52,344 pairs (32.8%)

**Source:** [SimulaMet/Kvasir-VQA-x1](https://huggingface.co/datasets/SimulaMet/Kvasir-VQA-x1) | [GitHub](https://github.com/simula/Kvasir-VQA-x1) | [Paper](https://huggingface.co/papers/2506.09958)

---

## Methodology

### Architecture
- **Base Model:** BLIP-2 (`Salesforce/blip2-opt-2.7b`) — a vision-language model with ViT encoder, Q-Former bridge, and OPT-2.7B language model
- **Fine-tuning:** LoRA (Low-Rank Adaptation) applied to OPT attention layers (`q_proj`, `v_proj`), with 8-bit quantization via `bitsandbytes`
- **Uncertainty Estimation:** Three complementary methods (predictive entropy, MC Dropout, sequence confidence)
- **Abstention:** Threshold-based selective prediction — model abstains when combined uncertainty exceeds τ

### Evaluation Metrics
- **VQA Quality:** Exact Match, Word F1/Precision/Recall, BLEU-1/2/3/4, ROUGE-L, METEOR, BERTScore
- **Safety:** Risk-Coverage curves, AUROC, Expected Calibration Error (ECE), Selective Accuracy

---

## Work Completed

### Phase 1: Literature Review & Problem Formulation (February 2026)
- Reviewed 10+ research papers on medical VQA, multimodal learning, and hallucination mitigation
- Studied the Kvasir-VQA-x1 dataset structure including image modalities, question types, and annotation format
- Explored uncertainty estimation techniques: Monte Carlo Dropout, temperature scaling, evidential deep learning
- Identified candidate base architectures: BiomedCLIP, LLaVA-Med, CLIP-based adapter models
- Defined the research question and evaluation metrics

### Phase 2: Project Setup & Data Pipeline (February 2026)
- Set up project repository with modular codebase and YAML-based configuration management
- Downloaded and prepared the full Kvasir-VQA-x1 dataset (6,449 images, 159,549 QA pairs)
- Built data preprocessing pipeline with text cleaning, image validation, and stratified train/validation/test splitting
- Implemented PyTorch `Dataset` and `DataLoader` classes with image augmentation transforms

### Phase 3: Exploratory Data Analysis (February 2026)
- Generated publication-quality visualizations of question class distributions, complexity levels, and text length statistics
- Analyzed dataset balance across complexity levels and train/test splits

### Phase 4: Baseline Inference (February–March 2026)
- Ran zero-shot VQA inference using **BLIP-2** (`Salesforce/blip2-opt-2.7b`) on 48 test samples
- Established baseline performance with comprehensive metrics

**Baseline Results (BLIP-2 Zero-Shot, 48 samples):**

| Metric | Value |
|--------|-------|
| Exact Match | 0.0% |
| Word F1 | 23.4% |
| Word Precision | 21.3% |
| Word Recall | 29.8% |
| BLEU-1 | 19.5% |
| BLEU-4 | 4.2% |
| ROUGE-L | 20.8% |
| METEOR | 21.8% |
| BERTScore F1 | 25.9% |

**Hallucination examples confirmed:**
- Fabricating wrong procedures (e.g., "laparoscopic cholecystectomy" for colonoscopy images)
- Identifying wrong organ systems (e.g., "urethral sphincter" in GI endoscopy)
- Confidently contradicting visible findings (e.g., claiming polyps are absent when present)

### Phase 5: LoRA Fine-Tuning (March 2026)
- Implemented LoRA fine-tuning pipeline with 8-bit quantization for GPU efficiency
- Trained on a stratified subset (~500 samples) for 3 epochs (~18 minutes on T4 GPU)
- Identified and fixed a critical causal LM label alignment bug (prompt tokens must be masked in labels)
- Implemented comprehensive evaluation metrics (BLEU, ROUGE-L, METEOR, BERTScore) alongside existing F1

**Fine-Tuned Results (LoRA, ~500 training samples, 20 eval samples):**

| Metric | Baseline | Fine-Tuned | Δ |
|--------|----------|------------|---|
| Exact Match | 0.0% | 5.0% | +5.0% |
| Word F1 | 23.4% | 49.6% | **+26.2%** |
| Partial Match (F1≥0.5) | 5.6% | 50.0% | **+44.4%** |

**Training curve:** Loss 2.20 → 0.87 → 0.76 across 3 epochs.

### Phase 6: Uncertainty Estimation & Abstention (March–April 2026)
- Implemented three uncertainty estimation methods:
  - **Predictive Entropy:** Token-level softmax entropy during generation
  - **MC Dropout:** 5 stochastic forward passes measuring lexical variance
  - **Sequence Confidence:** Normalized log-probability of generated tokens
- Built abstention mechanism with threshold tuning (target: 80% coverage)
- Implemented safety metrics: Risk-Coverage curves, AUROC, ECE, selective accuracy

---

## Planned Work (April 2026)

1. **Scale fine-tuning** to 2,000+ stratified training samples
2. **Run uncertainty evaluation** on the scaled model and resolve remaining notebook issues
3. **Analyze safety metrics** — Risk-Coverage curves, AUROC, ECE
4. **Compare** three configurations: baseline vs. fine-tuned vs. uncertainty-aware
5. **Final project report** and documentation

---

## Project Structure

```
├── configs/
│   └── config.yaml                     # Central configuration (model, LoRA, training)
├── src/
│   ├── download_dataset.py             # Dataset download from HuggingFace
│   ├── dataset.py                      # PyTorch Dataset & DataLoaders
│   ├── eda.py                          # Exploratory Data Analysis
│   ├── preprocessing.py                # Image & text preprocessing pipeline
│   ├── baseline_inference.py           # Zero-shot BLIP-2 inference
│   ├── train_utils.py                  # Shared metrics (BLEU, ROUGE, METEOR, BERTScore, etc.)
│   ├── uncertainty.py                  # Uncertainty estimation (entropy, MC Dropout, log-prob)
│   ├── abstention.py                   # Threshold-based abstention mechanism
│   └── safety_metrics.py               # Risk-Coverage, AUROC, ECE, selective accuracy
├── notebooks/
│   ├── baseline_inference_colab.ipynb  # Zero-shot baseline (Colab)
│   ├── finetune_blip2_colab.ipynb      # LoRA fine-tuning (Colab)
│   └── uncertainty_eval_colab.ipynb    # Uncertainty + abstention eval (Colab)
├── data/                               # Downloaded dataset (gitignored)
│   ├── images/                         # 6,449 GI endoscopy images
│   ├── kvasir_vqa_x1_train.csv         # 143,594 training QA pairs
│   └── kvasir_vqa_x1_test.csv          # 15,955 test QA pairs
├── results/
│   ├── eda/                            # EDA plots and statistics
│   ├── predictions/                    # Baseline & fine-tuned predictions
│   │   ├── baseline_summary.json
│   │   ├── baseline_predictions.csv
│   │   ├── finetuned_summary.json
│   │   └── finetuned_predictions.csv
│   ├── uncertainty/                    # Uncertainty analysis results
│   └── training_log.json              # Fine-tuning loss curve
├── monthly_report_february.tex         # February progress report
├── monthly_report_march.tex            # March progress report
├── literature_review.md                # Full literature review
├── requirements.txt                    # Python dependencies
├── .gitignore
└── README.md
```

## Setup & Usage

### Prerequisites
- Python 3.10+
- CUDA-compatible GPU (for inference and training)
- Google Colab (recommended for training — T4 GPU sufficient)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Dataset
```bash
python src/download_dataset.py
```

### 3. Run EDA
```bash
python src/eda.py
```

### 4. Run Baseline Inference
Use the Colab notebook: `notebooks/baseline_inference_colab.ipynb`

### 5. Fine-Tune with LoRA
Use the Colab notebook: `notebooks/finetune_blip2_colab.ipynb`

### 6. Run Uncertainty Evaluation
Use the Colab notebook: `notebooks/uncertainty_eval_colab.ipynb`

---

## References

1. **Kvasir-VQA-x1 Dataset** — Gautam et al., "Visual Question Answering for Gastrointestinal Imaging" (2025). [Paper](https://huggingface.co/papers/2506.09958)
2. **MediaEval Medico 2025 Challenge** — [GitHub](https://github.com/simula/MediaEval-Medico-2025)
3. **BLIP-2** — Li et al., "BLIP-2: Bootstrapping Language-Image Pre-training" (2023). [HuggingFace](https://huggingface.co/Salesforce/blip2-opt-2.7b)
4. **LoRA** — Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2022). [Paper](https://arxiv.org/abs/2106.09685)
5. **MC Dropout** — Gal & Ghahramani, "Dropout as a Bayesian Approximation" (2016). [Paper](https://arxiv.org/abs/1506.02142)

---

## Team

| Member | Roll Number |
|--------|-------------|
| Sunit Soni | AP22110011494 |
| Prashant Dhimal | AP22110011492 |
| Jayash Shrestha | AP22110011481 |

**Supervisor:** Dr. M Krishna Siva Prasad

**Institution:** Department of CSE, SRM University AP