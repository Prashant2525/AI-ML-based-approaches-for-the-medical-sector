# Parallel Training (DDP)

Distributed Data Parallel version of the VLM fine-tuning pipeline.

## Prerequisites

```bash
pip install transformers>=4.45.0 datasets accelerate pillow pandas tqdm
pip install peft nltk rouge-score matplotlib seaborn
pip install qwen-vl-utils protobuf sentencepiece bert-score
```

## Usage

### 4 GPUs (GPUs 1-4)
```bash
CUDA_VISIBLE_DEVICES=1,2,3,4 torchrun --nproc_per_node=4 paralleltrain/train_ddp.py --model llava_med
```

### 6 GPUs (GPUs 1-6)
```bash
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 torchrun --nproc_per_node=6 paralleltrain/train_ddp.py --model llava_med
```

### Choose model
```bash
CUDA_VISIBLE_DEVICES=1,2,3,4 torchrun --nproc_per_node=4 paralleltrain/train_ddp.py --model instructblip
CUDA_VISIBLE_DEVICES=1,2,3,4 torchrun --nproc_per_node=4 paralleltrain/train_ddp.py --model qwen2_vl
CUDA_VISIBLE_DEVICES=1,2,3,4 torchrun --nproc_per_node=4 paralleltrain/train_ddp.py --model llava_med
```

## Key DDP Differences vs Notebook

| Feature | Notebook (DataParallel) | Script (DDP) |
|---------|:-:|:-:|
| Scaling | ~2-3x | ~3.8x (near-linear) |
| Process model | 1 process, GIL bottleneck | 1 process per GPU |
| Gradient sync | Manual gather on GPU 0 | NCCL all-reduce (fast) |
| Data split | Batch scatter | DistributedSampler |
| Val loss | Per-GPU only | all_reduce averaged |

## Output Files

All outputs go to `results/predictions/`:
- `{model}_zs.csv` — Zero-shot per-sample results
- `{model}_ft.csv` — Fine-tuned per-sample results
- `{model}_comparison.json` — Full comparison with config
- `{model}_loss.png` — Training loss curves
- `{model}_metrics.png` — ZS vs FT bar chart
