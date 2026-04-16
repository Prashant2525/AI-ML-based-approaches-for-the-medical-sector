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

## Results

### Final Comparison — Baseline vs Fine-Tuned vs Uncertainty-Aware (50 eval samples)

| Metric | Baseline (Zero-Shot) | Fine-Tuned (2000 samples) | Uncertainty-Aware | Selective @84% Coverage |
|--------|:--------------------:|:-------------------------:|:-----------------:|:-----------------------:|
| Exact Match | 0.0% | 2.0% | **8.0%** | — |
| Word F1 | 28.9% | 45.2% | 55.5% | **61.0%** |
| Word Precision | 25.8% | 41.8% | — | — |
| Word Recall | 38.0% | **61.9%** | — | — |
| BLEU-1 | 24.2% | 39.7% | **50.5%** | — |
| BLEU-4 | 4.7% | 20.9% | **25.9%** | — |
| ROUGE-L | 23.8% | 41.9% | **50.9%** | — |
| METEOR | 27.5% | 48.5% | **50.9%** | — |
| BERTScore F1 | 32.9% | **48.3%** | — | — |

**Key improvements from Baseline → Uncertainty-Aware:**
- Word F1: **+26.5%** (28.9% → 55.5%)
- BLEU-4: **+21.2%** (4.7% → 25.9%) — accurate multi-word phrase generation
- Word Recall: **+23.9%** (38.0% → 61.9%) — critical for medical safety
- Selective F1 at 84% coverage: **61.0%** — +5.6% over the overall F1 from abstention alone

### Safety Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| AUROC | **0.622** | Good — uncertainty discriminates correct vs. incorrect answers |
| AUC-Risk | **0.380** | Lower is safer; model takes less risk at any given coverage |
| ECE | **0.312** | Moderate calibration (lower = confidence matches accuracy better) |
| Abstention Threshold τ | 0.423 | Combined uncertainty cutoff |
| Coverage | **84%** | 42/50 answered, 8 abstained |
| Selective F1 | **61.0%** | F1 on answered samples only |
| Abstention Gain | **+5.6%** | Improvement from refusing hardest questions |

### Selective Accuracy at Coverage Levels

| Coverage | Selective Word F1 |
|:--------:|:-----------------:|
| 50% | 61.2% |
| 60% | 61.4% |
| 70% | 60.4% |
| **80%** | **60.5%** ← target |
| 90% | 60.1% |
| 100% | 55.5% |

### Per-Complexity Breakdown (Fine-Tuned, 2000 samples)

| Complexity | Baseline F1 | Fine-Tuned F1 | Δ |
|:----------:|:-----------:|:-------------:|:-:|
| Level 1 (simple) | 14.8% | 26.1% | +11.3% |
| Level 2 (medium) | 37.4% | 50.1% | +12.7% |
| Level 3 (complex) | 32.3% | 54.4% | **+22.1%** |

### Training Details

| Parameter | Value |
|-----------|-------|
| Training samples | 2,000 (stratified by complexity) |
| Epochs | 3 |
| LoRA rank / alpha | 16 / 32 |
| Effective batch size | 16 (4 × 4 gradient accumulation) |
| Learning rate | 2e-4 (cosine schedule with warmup) |
| Final training loss | 0.561 |
| Training time | ~77 minutes (T4 GPU) |
| MC Dropout passes | 5 |

**Training Loss Curve:**

| Epoch | Loss | Elapsed |
|:-----:|:----:|:-------:|
| 1 | 1.770 | 25.8 min |
| 2 | 0.627 | 51.6 min |
| 3 | 0.561 | 77.4 min |

### Safety Analysis Plots

![Safety Analysis — Risk-Coverage, Uncertainty vs Quality, Reliability Diagram, Uncertainty by Complexity](results/uncertainty/safety_plots.png)

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
- Ran zero-shot VQA inference using **BLIP-2** (`Salesforce/blip2-opt-2.7b`) on 50 test samples
- Established baseline: Word F1 = 28.9%, Exact Match = 0.0%
- Confirmed hallucination patterns: fabricating procedures, identifying wrong organs, contradicting visible findings

### Phase 5: LoRA Fine-Tuning (March–April 2026)
- Implemented LoRA fine-tuning pipeline with 8-bit quantization for GPU efficiency
- Identified and fixed a critical causal LM label alignment bug (prompt tokens must be masked in labels)
- Trained on 2,000 stratified samples for 3 epochs (~77 min on T4 GPU)
- Achieved Word F1 = 45.2%, Word Recall = 61.9%, BLEU-4 = 20.9% (up from 4.7% baseline)

### Phase 6: Uncertainty Estimation & Abstention (April 2026)
- Implemented three uncertainty estimation methods:
  - **Predictive Entropy:** Token-level softmax entropy during generation
  - **MC Dropout:** 5 stochastic forward passes measuring lexical variance
  - **Sequence Confidence:** Normalized log-probability of generated tokens
- Built combined uncertainty score (0.4 × entropy + 0.3 × MC + 0.3 × confidence)
- Tuned abstention threshold τ = 0.423 for 84% coverage
- **Selective F1 = 61.0%** vs 55.5% overall — abstention delivers +5.6% improvement
- AUROC = 0.622 confirms uncertainty is informative for error detection

### Phase 7: Pipeline Consolidation (April 2026)
- Consolidated three separate notebooks (baseline, fine-tuning, uncertainty) into a **single unified pipeline** (`complete_pipeline_colab.ipynb`)
- Added skip-flags (`SKIP_BASELINE`, `SKIP_TRAINING`, `SKIP_UNCERTAINTY`) for modular execution
- Shared evaluation functions, consistent eval subset across all phases
- Final 3-way comparison table (Baseline → Fine-Tuned → Uncertainty-Aware → Selective)

---

## Project Structure

```
├── configs/
│   └── config.yaml                        # Central configuration (model, LoRA, training)
├── src/
│   ├── download_dataset.py                # Dataset download from HuggingFace
│   ├── dataset.py                         # PyTorch Dataset & DataLoaders
│   ├── eda.py                             # Exploratory Data Analysis
│   ├── preprocessing.py                   # Image & text preprocessing pipeline
│   ├── baseline_inference.py              # Zero-shot BLIP-2 inference
│   ├── train_utils.py                     # Shared metrics (BLEU, ROUGE, METEOR, BERTScore)
│   ├── uncertainty.py                     # Uncertainty estimation (entropy, MC Dropout, log-prob)
│   ├── abstention.py                      # Threshold-based abstention mechanism
│   └── safety_metrics.py                  # Risk-Coverage, AUROC, ECE, selective accuracy
├── notebooks/
│   ├── complete_pipeline_colab.ipynb      # ★ Consolidated pipeline (recommended)
│   ├── baseline_inference_colab.ipynb     # Zero-shot baseline (standalone)
│   ├── finetune_blip2_colab.ipynb         # LoRA fine-tuning (standalone)
│   └── uncertainty_eval_colab.ipynb       # Uncertainty + abstention (standalone)
├── data/                                   # Downloaded dataset (gitignored)
│   ├── images/                            # 6,449 GI endoscopy images
│   ├── kvasir_vqa_x1_train.csv            # 143,594 training QA pairs
│   └── kvasir_vqa_x1_test.csv             # 15,955 test QA pairs
├── results/
│   ├── eda/                               # EDA plots and statistics
│   ├── predictions/                       # Baseline & fine-tuned predictions
│   │   ├── baseline_summary.json          # Zero-shot metrics
│   │   ├── baseline_predictions.csv       # Per-sample baseline predictions
│   │   ├── finetuned_summary.json         # LoRA fine-tuned metrics (2000 samples)
│   │   └── finetuned_predictions.csv      # Per-sample fine-tuned predictions
│   ├── uncertainty/                       # Uncertainty analysis outputs
│   │   ├── uncertainty_summary.json       # Safety metrics, abstention results
│   │   └── safety_plots.png              # 4-panel safety visualization
│   └── training_log.json                 # Fine-tuning loss curve (3 epochs)
├── monthly_report_february.tex            # February progress report
├── monthly_report_march.tex               # March progress report
├── literature_review.md                   # Full literature review
├── project_novelty.md                     # Novelty documentation
├── requirements.txt                       # Python dependencies
├── .gitignore
└── README.md
```

---

## Setup & Usage

### Prerequisites
- Python 3.10+
- CUDA-compatible GPU (for inference and training)
- Google Colab (recommended for training — T4 GPU sufficient)

### Quick Start (Recommended)

Use the consolidated notebook — it handles everything:

1. Open `notebooks/complete_pipeline_colab.ipynb` in Google Colab
2. Set `USE_DRIVE = True` in Cell 2 and mount your Drive
3. Configure skip flags in Cell 3:
   - `SKIP_BASELINE = True` — load saved baseline results
   - `SKIP_TRAINING = True` — load saved LoRA checkpoint
   - `SKIP_UNCERTAINTY = True` — load saved uncertainty results
4. Run all cells — the notebook will run or skip each phase accordingly

### Local Setup

```bash
pip install -r requirements.txt
python src/download_dataset.py
python src/eda.py
```

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