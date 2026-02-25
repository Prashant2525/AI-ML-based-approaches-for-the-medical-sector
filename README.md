# AI/ML-based Approaches for the Medical Sector

## Uncertainty-Aware Visual Question Answering on Kvasir-VQA-x1

A medical Visual Question Answering (VQA) system for gastrointestinal endoscopy images using the **Kvasir-VQA-x1** dataset. This project implements an uncertainty-aware multimodal model with a confidence-based abstention mechanism to reduce hallucinations in medical AI.

### Research Question
> *How effective are uncertainty-aware training objectives in reducing hallucinations for VQA tasks on the Kvasir-VQA-x1 dataset?*

---

## Project Structure

```
├── configs/
│   └── config.yaml              # Central configuration
├── src/
│   ├── download_dataset.py      # Dataset download from HuggingFace
│   ├── dataset.py               # PyTorch Dataset & DataLoaders
│   ├── eda.py                   # Exploratory Data Analysis
│   ├── preprocessing.py         # Image & text preprocessing
│   └── baseline_inference.py    # Zero-shot baseline model inference
├── data/                        # Downloaded dataset (gitignored)
│   └── images/                  # GI endoscopy images
├── results/                     # Output results
│   ├── eda/                     # EDA plots and statistics
│   └── predictions/             # Model predictions
├── requirements.txt             # Python dependencies
└── README.md
```

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Dataset
```bash
python src/download_dataset.py
```

### 3. Run Exploratory Data Analysis
```bash
python src/eda.py
```

### 4. Run Baseline Inference
```bash
python src/baseline_inference.py
```

## Dataset

**Kvasir-VQA-x1** is a large-scale dataset for medical VQA in gastrointestinal endoscopy:
- **6,500** GI endoscopic images
- **159,549** complex QA pairs
- **3 complexity levels** (1–3)
- Multiple question classes (Yes/No, counting, descriptive, reasoning)

Source: [SimulaMet/Kvasir-VQA-x1](https://huggingface.co/datasets/SimulaMet/Kvasir-VQA-x1)

## Team

| Member | Roll Number |
|--------|-------------|
| Sunit Soni | AP22110011494 |
| Prashant Dhimal | AP22110011492 |
| Jayash Shrestha | AP22110011481 |

**Supervisor:** Dr. M Krishna Siva Prasad

**Institution:** Department of CSE, SRM University AP