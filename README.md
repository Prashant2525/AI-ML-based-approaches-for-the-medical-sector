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
- **Base Models:** BLIP-2 (OPT-2.7B), InstructBLIP (Flan-T5-XL, 3.5B), SmolVLM2 (2.2B)
- **Fine-tuning:** LoRA / QLoRA (4-bit) applied to attention layers, with gradient checkpointing
- **Uncertainty Estimation:** Three complementary methods — predictive entropy, MC Dropout (5 passes), sequence confidence
- **Abstention:** Threshold-based selective prediction — model abstains when combined uncertainty exceeds τ

### Training Infrastructure
| | Colab Pipeline | DDP Pipeline |
|---|---|---|
| GPU | 1× T4 (16GB) | 5–6× V100-SXM2 (32GB) |
| Training data | 2,000 samples | **143,594** (full dataset) |
| Quantization | 8-bit | 4-bit QLoRA (NF4) |
| Parallelism | Single GPU | Distributed Data Parallel |

### Evaluation Metrics
- **VQA Quality:** Accuracy, Word F1, BLEU-1/2/3/4, ROUGE-1/2/L, METEOR, BERTScore, CHRF++, BLEURT
- **Safety:** AUROC, AUC-Risk, ECE, Selective Accuracy, Risk-Coverage curves

---

## Results

### Multi-Model Comparison — Full Dataset Training (143K samples, 5000 eval)

| Metric | InstructBLIP ZS | InstructBLIP FT | SmolVLM2 ZS | SmolVLM2 FT (5ep) |
|--------|:---:|:---:|:---:|:---:|
| Accuracy | 0.0% | 15.4% | 0.0% | **17.3%** |
| Word F1 | 12.5% | 70.9% | 33.8% | **73.3%** |
| BLEU-1 | 3.1% | 66.3% | 25.0% | **68.9%** |
| BLEU-4 | 0.7% | 40.3% | 5.2% | **44.0%** |
| ROUGE-L | 11.7% | 67.7% | 26.6% | **70.6%** |
| METEOR | 5.4% | 66.0% | 28.5% | **68.7%** |
| BERTScore | 67.8% | 92.7% | 80.9% | **93.3%** |
| CHRF++ | 5.6 | 63.2 | 30.3 | **65.9** |
| BLEURT | 0.081 | 0.749 | 0.526 | **0.768** |

### Uncertainty-Aware Evaluation (999 eval samples, 5 MC passes)

| Metric | InstructBLIP | SmolVLM2 (5ep) |
|--------|:---:|:---:|
| Overall F1 | 68.75% | **72.47%** |
| Selective F1 (@80% cov) | 73.25% | **75.45%** |
| **Abstention Gain** | **+4.50%** | **+2.98%** |
| AUROC | **0.770** | 0.688 |
| ECE | 0.688 | **0.079** |
| AUC-Risk | 0.222 | **0.202** |
| Coverage | 80.6% | 80.4% |
| Abstained | 194/999 | 196/999 |

### Selective Accuracy at Coverage Levels (SmolVLM2 5ep)

| Coverage | Selective F1 |
|:--------:|:-----------:|
| 50% | 79.3% |
| 60% | 78.4% |
| 70% | 77.3% |
| **80%** | **75.6%** ← target |
| 90% | 74.1% |
| 100% | 72.5% |

---

## Work Completed

### Phase 1: Literature Review & Problem Formulation (February 2026)
- Reviewed 10+ research papers on medical VQA, multimodal learning, and hallucination mitigation
- Studied the Kvasir-VQA-x1 dataset structure including image modalities, question types, and annotation format
- Explored uncertainty estimation techniques: Monte Carlo Dropout, temperature scaling, evidential deep learning
- Defined the research question and evaluation metrics

### Phase 2: Project Setup & Data Pipeline (February 2026)
- Set up project repository with modular codebase and YAML-based configuration management
- Downloaded and prepared the full Kvasir-VQA-x1 dataset (6,449 images, 159,549 QA pairs)
- Built data preprocessing pipeline with text cleaning, image validation, and stratified splitting

### Phase 3: Exploratory Data Analysis (February 2026)
- Generated publication-quality visualizations of question class distributions, complexity levels, and text length statistics

### Phase 4: Baseline Inference (February–March 2026)
- Ran zero-shot VQA inference using BLIP-2 on 50 test samples
- Established baseline: Word F1 = 28.9%, Exact Match = 0.0%
- Confirmed hallucination patterns: fabricating procedures, identifying wrong organs, contradicting visible findings

### Phase 5: LoRA Fine-Tuning — Colab (March–April 2026)
- Implemented LoRA fine-tuning pipeline with 8-bit quantization on Colab (T4 GPU)
- Trained BLIP-2 on 2,000 stratified samples for 3 epochs (~77 min)
- Achieved Word F1 = 45.2%, BLEU-4 = 20.9%

### Phase 6: Uncertainty Estimation & Abstention — Colab (April 2026)
- Implemented three uncertainty estimation methods (entropy, MC Dropout, sequence confidence)
- Built combined uncertainty score: `0.4 × entropy + 0.3 × MC + 0.3 × (1 - confidence)`
- Tuned abstention threshold, achieving Selective F1 = 61.0% vs 55.5% overall (+5.6% gain)

### Phase 7: Pipeline Consolidation (April 2026)
- Consolidated three separate notebooks into a single unified pipeline (`complete_pipeline_colab.ipynb`)
- Added skip-flags for modular execution across all phases

### Phase 8: Multi-GPU DDP Training (April 2026)
- Built `train_ddp.py` — a production-grade DDP pipeline for multi-GPU fine-tuning on NVIDIA DGX (V100s)
- Trained **InstructBLIP** (5 epochs, 6 GPUs) and **SmolVLM2** (3ep + 5ep, 5 GPUs) on the **full 143K dataset**
- QLoRA 4-bit quantization with NF4, cosine LR scheduling, early stopping
- SmolVLM2 (5ep) achieved **73.3% Word F1** (vs 33.8% zero-shot) — best model overall
- Comprehensive evaluation: 14+ metrics including BERTScore, CHRF++, BLEURT

### Phase 9: Uncertainty Evaluation at Scale (April 2026)
- Built `uncertainty_eval.py` — standalone script applying uncertainty estimation to DDP-trained models
- Ran on 999 stratified test samples with 5 MC Dropout passes (~2.3–4.5 hours per model on V100)
- **SmolVLM2 (5ep):** AUROC = 0.688, Selective F1 = 75.5% (+3.0%), ECE = 0.079
- **InstructBLIP:** AUROC = 0.770, Selective F1 = 73.3% (+4.5%), ECE = 0.688
- Generated 4-panel safety analysis plots for both models

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
│   ├── complete_pipeline_colab.ipynb      # ★ Consolidated pipeline (Colab, T4 GPU)
│   ├── baseline_inference_colab.ipynb     # Zero-shot baseline (standalone)
│   ├── finetune_blip2_colab.ipynb         # LoRA fine-tuning (standalone)
│   └── uncertainty_eval_colab.ipynb       # Uncertainty + abstention (standalone)
├── paralleltrain/                         # ★ Multi-GPU DDP pipeline (DGX, V100s)
│   ├── train_ddp.py                       # DDP fine-tuning + evaluation
│   ├── uncertainty_eval.py                # Uncertainty estimation + abstention
│   ├── README.md                          # DDP pipeline documentation
│   └── Model Results/                     # Saved outputs per run
│       ├── instructblip_..._ep5_.../      # InstructBLIP (5ep, 6 GPUs)
│       ├── smolvlm2_..._r32_ep3_.../     # SmolVLM2 (3ep, r=32)
│       ├── smolvlm2_..._r16_ep3_.../     # SmolVLM2 (3ep, r=16)
│       └── smolvlm2_..._r32_ep5_.../     # SmolVLM2 (5ep, r=32) — best model
├── data/                                   # Downloaded dataset (gitignored)
├── results/                               # Colab pipeline results
│   ├── predictions/                       # Baseline & fine-tuned predictions (BLIP-2)
│   ├── uncertainty/                       # Uncertainty analysis (BLIP-2)
│   └── training_log.json                  # BLIP-2 fine-tuning loss curve
├── sn-article.tex                         # Final project report (Springer format)
├── monthly_report_february.tex            # February progress report
├── monthly_report_march.tex               # March progress report
├── literature_review.md                   # Full literature review
├── project_novelty.md                     # Novelty documentation
├── project_deep_dive.md                   # Technical deep dive
├── requirements.txt                       # Python dependencies
├── .gitignore
└── README.md
```

---

## Setup & Usage

### Prerequisites
- Python 3.10+
- CUDA-compatible GPU
- Google Colab (T4 GPU) or NVIDIA DGX (V100s for DDP)

### Quick Start — Colab (single GPU)

1. Open `notebooks/complete_pipeline_colab.ipynb` in Google Colab
2. Set `USE_DRIVE = True` in Cell 2 and mount your Drive
3. Configure skip flags in Cell 3 to run/skip phases
4. Run all cells

### Full Training — DDP (multi-GPU)

```bash
# Fine-tune on full dataset
CUDA_VISIBLE_DEVICES=1,2,3,4,5 torchrun --nproc_per_node=5 paralleltrain/train_ddp.py --model smolvlm2

# Run uncertainty evaluation
python paralleltrain/uncertainty_eval.py --model smolvlm2 --eval_samples 500 --gpu 0
```

See [paralleltrain/README.md](paralleltrain/README.md) for full DDP documentation.

---

## References

1. **Kvasir-VQA-x1 Dataset** — Gautam et al., "Visual Question Answering for Gastrointestinal Imaging" (2025). [Paper](https://huggingface.co/papers/2506.09958)
2. **MediaEval Medico 2025 Challenge** — [GitHub](https://github.com/simula/MediaEval-Medico-2025)
3. **BLIP-2** — Li et al., "BLIP-2: Bootstrapping Language-Image Pre-training" (2023). [HuggingFace](https://huggingface.co/Salesforce/blip2-opt-2.7b)
4. **InstructBLIP** — Dai et al., "InstructBLIP: Towards General-purpose Vision-Language Models" (2023)
5. **SmolVLM2** — HuggingFace, "SmolVLM2-2.2B-Instruct" (2025). [HuggingFace](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct)
6. **LoRA** — Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2022). [Paper](https://arxiv.org/abs/2106.09685)
7. **MC Dropout** — Gal & Ghahramani, "Dropout as a Bayesian Approximation" (2016). [Paper](https://arxiv.org/abs/1506.02142)

---

## Team

| Member | Roll Number |
|--------|-------------|
| Sunit Soni | AP22110011494 |
| Prashant Dhimal | AP22110011492 |
| Jayash Shrestha | AP22110011481 |

**Supervisor:** Dr. M Krishna Siva Prasad

**Institution:** Department of CSE, SRM University AP