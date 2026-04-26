# Parallel Training & Uncertainty Evaluation (DDP)

Multi-GPU fine-tuning pipeline (DDP) and standalone uncertainty-aware evaluation for VLMs on Kvasir-VQA-x1.

**Hardware:** NVIDIA DGX — V100-SXM2-32GB GPUs

## Prerequisites

```bash
pip install transformers>=4.45.0 datasets accelerate pillow pandas tqdm
pip install peft nltk rouge-score matplotlib seaborn
pip install qwen-vl-utils protobuf sentencepiece bert-score sacrebleu
```

---

## 1. Training (`train_ddp.py`)

DDP-based fine-tuning with QLoRA (4-bit NF4) on the full 143K training set.

### Supported Models

| Model | Key | Params | Architecture |
|-------|-----|--------|-------------|
| InstructBLIP (Flan-T5-XL) | `instructblip` | 3.5B | Encoder-Decoder |
| SmolVLM2-2.2B-Instruct | `smolvlm2` | 2.2B | Causal (Decoder-only) |

### Usage

```bash
# 5 GPUs
CUDA_VISIBLE_DEVICES=1,2,3,4,5 torchrun --nproc_per_node=5 paralleltrain/train_ddp.py --model smolvlm2

# 6 GPUs
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 torchrun --nproc_per_node=6 paralleltrain/train_ddp.py --model instructblip
```

### Key DDP Differences vs Notebook

| Feature | Notebook (DataParallel) | Script (DDP) |
|---------|:-:|:-:|
| Scaling | ~2-3x | ~3.8x (near-linear) |
| Process model | 1 process, GIL bottleneck | 1 process per GPU |
| Gradient sync | Manual gather on GPU 0 | NCCL all-reduce (fast) |
| Data split | Batch scatter | DistributedSampler |
| Val loss | Per-GPU only | all_reduce averaged |

### Training Output

Saved to `results/predictions/`:
- `{model}_zs.csv` / `{model}_ft.csv` — Per-sample ZS and FT predictions
- `{model}_comparison.json` — Full comparison with config + uncertainty (if run)
- `{model}_loss.png` — Training loss curves
- `{model}_metrics.png` — ZS vs FT bar chart
- `{model}_bleu_rouge.png` / `{model}_complexity.png` / `{model}_heatmap.png` — Breakdowns

---

## 2. Uncertainty Evaluation (`uncertainty_eval.py`)

Standalone, single-GPU script that applies **uncertainty-aware abstention** to the DDP-trained models.

Three uncertainty estimation methods:
1. **Predictive Entropy** — token-level softmax entropy during generation
2. **MC Dropout** — 5 stochastic forward passes measuring lexical variance (via LoRA dropout)
3. **Sequence Confidence** — normalized log-probability of generated tokens

Combined score: `0.4 × entropy + 0.3 × mc_dropout + 0.3 × (1 - confidence)`

### Usage

```bash
# Smoke test (~5 min)
python paralleltrain/uncertainty_eval.py --model smolvlm2 --eval_samples 10 --gpu 0

# Full run — SmolVLM2 5ep (~2.3 hours on V100)
python paralleltrain/uncertainty_eval.py --model smolvlm2 \
  --checkpoint_dir "paralleltrain/Model Results/smolvlm2_loraNone_ep5_lr1e-05_lora_r32_lora_alpha64_bs4_ga1_eval5000" \
  --eval_samples 1000 --gpu 0

# InstructBLIP (~4.5 hours on V100)
python paralleltrain/uncertainty_eval.py --model instructblip \
  --checkpoint_dir "paralleltrain/Model Results/instructblip_loraNone_ep5_lr1e-05_lora_r32_lora_alpha64_bs4_ga1_eval5000" \
  --eval_samples 1000 --gpu 1

# Detached (nohup)
nohup python paralleltrain/uncertainty_eval.py --model smolvlm2 --eval_samples 1000 --gpu 0 > unc.log 2>&1 &
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `smolvlm2` | Model key: `instructblip` or `smolvlm2` |
| `--checkpoint_dir` | auto-detect | Path to LoRA adapter directory |
| `--eval_samples` | 500 | Number of test samples (stratified by complexity) |
| `--mc_passes` | 5 | MC Dropout forward passes per sample |
| `--target_coverage` | 0.80 | Abstention coverage target |
| `--gpu` | 0 | GPU index |
| `--max_new_tokens` | 64 | Max generated tokens |
| `--seed` | 42 | Random seed |

### Output

Saved to `Model Results/{run}/results/uncertainty/`:
- `uncertainty_predictions.csv` — Per-sample: prediction, F1, entropy, MC unc, combined, abstained flag
- `uncertainty_summary.json` — Safety metrics, abstention stats, selective accuracy table
- `safety_plots.png` — 4-panel visualization (Risk-Coverage, Uncertainty vs Quality, Reliability, Complexity)

Also non-destructively updates `{model}_comparison.json` with `"uncertainty"` key.

---

## Model Results

### Training Runs

| Run | Model | Epochs | GPUs | LoRA | Best Val Loss | FT Word F1 |
|-----|-------|:------:|:----:|:----:|:---:|:---:|
| `instructblip_..._ep5` | InstructBLIP | 5 | 6 | r=32, α=64 | 0.424 | 70.9% |
| `smolvlm2_..._r32_ep3` | SmolVLM2 | 3 | 5 | r=32, α=64 | 0.210 | 72.5% |
| `smolvlm2_..._r16_ep3` | SmolVLM2 | 3 | 5 | r=16, α=32 | 0.229 | 71.8% |
| `smolvlm2_..._r32_ep5` | SmolVLM2 | **5** | 5 | r=32, α=64 | **0.194** | **73.3%** ★ |

All evaluated on 4,998 stratified test samples.

### Uncertainty Evaluation (999 eval samples)

| Metric | InstructBLIP | SmolVLM2 (5ep) |
|--------|:---:|:---:|
| Overall F1 | 68.8% | **72.5%** |
| Selective F1 (@80%) | 73.3% | **75.5%** |
| Abstention Gain | **+4.5%** | +3.0% |
| AUROC | **0.770** | 0.688 |
| ECE | 0.688 | **0.079** |
| AUC-Risk | 0.222 | **0.202** |
| n_answered / n_abstained | 805 / 194 | 803 / 196 |
| Elapsed | 4.5 hrs | 2.3 hrs |

---

## File Structure

```
paralleltrain/
├── train_ddp.py              # DDP fine-tuning + evaluation (multi-GPU)
├── uncertainty_eval.py       # Uncertainty estimation + abstention (single GPU)
├── README.md
└── Model Results/
    ├── instructblip_loraNone_ep5_.../
    │   ├── adapter_config.json
    │   └── results/
    │       ├── predictions/    # ZS & FT CSVs, comparison JSON, plots
    │       └── uncertainty/    # safety_plots.png, uncertainty_summary.json, predictions CSV
    ├── smolvlm2_..._r32_ep3_.../
    │   └── (same structure)
    ├── smolvlm2_..._r16_ep3_.../
    │   └── (same structure)
    └── smolvlm2_..._r32_ep5_.../   ★ best model
        └── (same structure)
```
