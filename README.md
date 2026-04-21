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
- **Uncertainty-aware training objectives** — the model learns *when* to abstain, not just *what* to answer
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

## Work Completed (February 2026)

### Phase 1: Literature Review & Problem Formulation
- Reviewed 10+ research papers on medical VQA, multimodal learning, and hallucination mitigation in vision-language models
- Studied the Kvasir-VQA-x1 dataset structure including image modalities, question types, and annotation format
- Explored uncertainty estimation techniques: Monte Carlo Dropout, temperature scaling, evidential deep learning
- Identified candidate base architectures: BiomedCLIP, LLaVA-Med, CLIP-based adapter models
- Defined the research question and evaluation metrics (Risk-Coverage Curve, Expected Calibration Error)

### Phase 2: Project Setup & Data Pipeline
- Set up project repository with modular codebase and YAML-based configuration management
- Downloaded and prepared the full Kvasir-VQA-x1 dataset (6,449 images, 159,549 QA pairs)
- Built data preprocessing pipeline with text cleaning, image validation, and stratified train/validation/test splitting
- Implemented PyTorch `Dataset` and `DataLoader` classes with image augmentation transforms

### Phase 3: Exploratory Data Analysis
- Generated publication-quality visualizations of question class distributions, complexity levels, and text length statistics
- Analyzed dataset balance across complexity levels and train/test splits
- All EDA outputs saved to `results/eda/`

### Phase 4: Baseline Inference
- Ran zero-shot VQA inference using pre-trained **BLIP-2** (`Salesforce/blip2-opt-2.7b`) on test samples
- Evaluated using exact-match accuracy and word-level F1 score

**Baseline Results (BLIP-2 Zero-Shot):**

| Metric | Value |
|--------|-------|
| Exact Match Accuracy | 0.0% |
| Average Word F1 | 27.0% |
| Partial Matches (F1 ≥ 0.5) | 1/18 (5.6%) |

| Complexity Level | Avg Word F1 |
|-----------------|-------------|
| Level 1 (simple) | 18.9% |
| Level 2 (medium) | 27.0% |
| Level 3 (complex) | 35.1% |

**Key Finding:** The generic BLIP-2 model exhibited dangerous hallucination behavior on medical data, including:
- Fabricating wrong procedures (e.g., "laparoscopic cholecystectomy" for colonoscopy images)
- Identifying wrong organ systems (e.g., "urethral sphincter" in GI endoscopy)
- Confidently contradicting visible findings (e.g., claiming polyps are absent when present)

These results confirm the need for domain-specific fine-tuning and uncertainty-aware mechanisms.

---

## Planned Work (March 2026)

1. **Fine-tune** the selected multimodal model on Kvasir-VQA-x1 training data using LoRA/QLoRA
2. **Implement uncertainty estimation** — entropy thresholding, Monte Carlo Dropout, and uncertainty-aware training objectives
3. **Build the abstention mechanism** — model outputs "Requires Doctor Review" when confidence is below threshold
4. **Evaluate** using Risk-Coverage curves, Expected Calibration Error (ECE), and AUROC of uncertainty
5. **Compare** fine-tuned model performance against the zero-shot baseline (0% exact match, 27% F1)

---

## Project Structure

```
├── configs/
│   └── config.yaml                  # Central project configuration
├── src/
│   ├── download_dataset.py          # Dataset download from HuggingFace
│   ├── dataset.py                   # PyTorch Dataset & DataLoaders
│   ├── eda.py                       # Exploratory Data Analysis
│   ├── preprocessing.py             # Image & text preprocessing pipeline
│   └── baseline_inference.py        # Zero-shot BLIP-2 baseline inference
├── notebooks/
│   └── baseline_inference_colab.ipynb  # Google Colab version of baseline
├── data/                            # Downloaded dataset (gitignored)
│   ├── images/                      # 6,449 GI endoscopy images
│   ├── kvasir_vqa_x1_train.csv      # 143,594 training QA pairs
│   ├── kvasir_vqa_x1_test.csv       # 15,955 test QA pairs
│   ├── preprocessed_train.csv       # Cleaned & split training data
│   ├── preprocessed_val.csv         # Stratified validation set
│   └── preprocessed_test.csv        # Cleaned test data
├── results/
│   ├── eda/                         # EDA plots and statistics
│   │   ├── dataset_statistics.txt
│   │   ├── question_class_distribution.png
│   │   ├── complexity_distribution.png
│   │   ├── text_length_distribution.png
│   │   ├── train_test_comparison.png
│   │   └── sample_images_qa.png
│   └── predictions/                 # Baseline model predictions
│       ├── baseline_predictions.csv
│       └── baseline_summary.json
├── monthly_report_february.tex      # Monthly progress report
├── requirements.txt                 # Python dependencies
├── .gitignore
└── README.md
```

## Setup & Usage

### Prerequisites
- Python 3.10+
- CUDA-compatible GPU (for baseline inference and training)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Dataset
```bash
python src/download_dataset.py
```
Downloads 6,449 images from `SimulaMet-HOST/Kvasir-VQA` and 159,549 QA pairs from `SimulaMet/Kvasir-VQA-x1`.

### 3. Run Exploratory Data Analysis
```bash
python src/eda.py
```
Generates visualizations and statistics in `results/eda/`.

### 4. Run Data Preprocessing
```bash
python src/preprocessing.py
```
Cleans text, validates images, creates stratified train/val/test splits.

### 5. Run Baseline Inference
```bash
python src/baseline_inference.py
```
Runs zero-shot BLIP-2 inference on test samples. Requires GPU. Alternatively, use the Colab notebook at `notebooks/baseline_inference_colab.ipynb`.

---

## References

1. **Kvasir-VQA-x1 Dataset** — Gautam et al., "Visual Question Answering for Gastrointestinal Imaging" (2025). [Paper](https://huggingface.co/papers/2506.09958)
2. **MediaEval Medico 2025 Challenge** — [GitHub](https://github.com/simula/MediaEval-Medico-2025)
3. **BLIP-2** — Li et al., "BLIP-2: Bootstrapping Language-Image Pre-training" (2023). [HuggingFace](https://huggingface.co/Salesforce/blip2-opt-2.7b)

---

## Team

| Member | Roll Number |
|--------|-------------|
| Sunit Soni | AP22110011494 |
| Prashant Dhimal | AP22110011492 |
| Jayash Shrestha | AP22110011481 |

**Supervisor:** Dr. M Krishna Siva Prasad

**Institution:** Department of CSE, SRM University AP