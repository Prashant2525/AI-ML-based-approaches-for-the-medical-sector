#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Federated Fine-Tuning of VLMs on Kvasir-VQA-x1 (FedAvg)
=========================================================
Single-GPU simulation of Federated Learning using FedAvg.
Each "client" trains on its own data partition, then LoRA weights are averaged.

Models: LLaVA-Med v1.5 (7B) | InstructBLIP (Flan-T5-XL)
System: NVIDIA DGX V100-SXM2-32GB (single GPU)

Usage:
    CUDA_VISIBLE_DEVICES=6 python federated_train.py --model llava_med --partition iid
    CUDA_VISIBLE_DEVICES=6 python federated_train.py --model instructblip --partition noniid
    CUDA_VISIBLE_DEVICES=6 python federated_train.py --model llava_med --partition noniid --num_clients 4 --num_rounds 20
"""

# ============================================================================
# 1. IMPORTS
# ============================================================================
import os, json, gc, math, re, time, warnings, random, argparse, copy
warnings.filterwarnings('ignore')

import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score as _meteor
from nltk.stem import PorterStemmer
from rouge_score import rouge_scorer

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.rcParams['figure.dpi'] = 120

# ============================================================================
# 2. CONFIGURATION
# ============================================================================
parser = argparse.ArgumentParser(description='Federated VLM Fine-Tuning (FedAvg)')
parser.add_argument('--model', type=str, default='llava_med',
                    choices=['instructblip', 'llava_med'],
                    help='Model to fine-tune')
parser.add_argument('--partition', type=str, default='iid',
                    choices=['iid', 'noniid'],
                    help='Data partition strategy')
parser.add_argument('--num_clients', type=int, default=4,
                    help='Total number of federated clients')
parser.add_argument('--num_rounds', type=int, default=20,
                    help='Number of federated rounds')
parser.add_argument('--fraction_fit', type=float, default=0.5,
                    help='Fraction of clients selected per round')
parser.add_argument('--local_epochs', type=int, default=1,
                    help='Local training epochs per client per round')
parser.add_argument('--max_train', type=int, default=None,
                    help='Max training samples (None = use all)')
parser.add_argument('--num_eval', type=int, default=5000,
                    help='Number of evaluation samples')
args = parser.parse_args()

MODEL_KEY = args.model

# ---- Model Registry (same as train_ddp.py) ----
MODEL_REGISTRY = {
    'instructblip': {
        'model_id':     'Salesforce/instructblip-flan-t5-xl',
        'model_name':   'InstructBLIP (Flan-T5-XL)',
        'model_type':   'encoder_decoder',
        'learning_rate': 1e-5,
        'lora_targets': ['q', 'k', 'v', 'o'],
        'lora_task':    'SEQ_2_SEQ_LM',
        'lora_r':       32,
        'lora_alpha':   64,
        'grad_ckpt':    False,
        'batch_size':   4,
        'grad_accum':   4,
    },
    'llava_med': {
        'model_id':     'chaoyinshe/llava-med-v1.5-mistral-7b-hf',
        'model_name':   'LLaVA-Med v1.5 (7B)',
        'model_type':   'causal',
        'learning_rate': 1e-5,
        'lora_targets': ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        'lora_task':    'CAUSAL_LM',
        'lora_r':       32,
        'lora_alpha':   64,
        'grad_ckpt':    True,
        'batch_size':   4,
        'grad_accum':   4,
    },
}

config = MODEL_REGISTRY[MODEL_KEY]
MODEL_ID     = config['model_id']
MODEL_NAME   = config['model_name']
LORA_R       = config['lora_r']
LORA_ALPHA   = config['lora_alpha']
BATCH_SIZE   = config['batch_size']
GRAD_ACCUM   = config['grad_accum']
LR_MAX       = config['learning_rate']
LR_MIN       = 0.0
LORA_DROPOUT = 0.1
MAX_SEQ_LEN  = 256
MAX_ANSWER_LEN = 64
MAX_NEW_TOKENS = 64
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

PROJECT_DIR = os.getcwd()
DATA_DIR    = os.path.join(PROJECT_DIR, 'data')
IMAGE_DIR   = os.path.join(DATA_DIR, 'images')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results', 'predictions')
CKPT_DIR    = os.path.join(PROJECT_DIR, 'checkpoints',
              f'{MODEL_KEY}_fl_{args.partition}_c{args.num_clients}_r{args.num_rounds}')
CACHE_DIR   = os.path.join(PROJECT_DIR, 'hf_cache')

for d in [RESULTS_DIR, CKPT_DIR, CACHE_DIR]:
    os.makedirs(d, exist_ok=True)

os.environ['HF_HOME'] = CACHE_DIR
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

TAG = f'{MODEL_KEY}_fl_{args.partition}'   # file naming prefix

print(f'\n{"="*60}')
print(f'  Federated Fine-Tuning (FedAvg)')
print(f'{"="*60}')
print(f'  Model:      {MODEL_NAME}')
print(f'  Partition:  {args.partition.upper()}')
print(f'  Clients:    {args.num_clients}')
print(f'  Rounds:     {args.num_rounds}')
print(f'  Fraction:   {args.fraction_fit} ({max(1,int(args.num_clients * args.fraction_fit))} clients/round)')
print(f'  Local epochs: {args.local_epochs}')
print(f'  Device:     {DEVICE}')
print(f'{"="*60}\n')

# ============================================================================
# 3. EVALUATION METRICS (same as train_ddp.py)
# ============================================================================
bleu_smoother = SmoothingFunction().method1
rouge_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
stemmer = PorterStemmer()

def normalize_text(text):
    text = text.strip().lower()
    text = re.sub(r'[^\w\s]', '', text)
    return [stemmer.stem(w) for w in text.split()]

def compute_word_f1(pred, gt):
    p, g = set(normalize_text(pred)), set(normalize_text(gt))
    if not p or not g: return 0.0
    c = p & g
    if not c: return 0.0
    return 2 * len(c) / (len(p) + len(g))

def compute_bleu(pred, gt, n):
    ref, hyp = normalize_text(gt), normalize_text(pred)
    if not ref or not hyp: return 0.0
    weights = tuple([1.0/n]*n + [0.0]*(4-n))
    try: return sentence_bleu([ref], hyp, weights=weights, smoothing_function=bleu_smoother)
    except: return 0.0

def compute_rouge(pred, gt):
    if not pred.strip() or not gt.strip():
        return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    s = rouge_obj.score(gt.strip().lower(), pred.strip().lower())
    return {k: s[k].fmeasure for k in ['rouge1', 'rouge2', 'rougeL']}

def compute_meteor(pred, gt):
    try: return _meteor([normalize_text(gt)], normalize_text(pred))
    except: return 0.0

def compute_ece(confs, accs, bins=10):
    if not confs: return 0.0
    edges = np.linspace(0, 1, bins+1)
    total, ece = len(confs), 0.0
    for i in range(bins):
        mask = [(edges[i] <= c < edges[i+1]) for c in confs]
        cnt = sum(mask)
        if cnt:
            ece += (cnt/total) * abs(
                np.mean([a for a, m in zip(accs, mask) if m]) -
                np.mean([c for c, m in zip(confs, mask) if m]))
    return ece

METRIC_KEYS = ['word_f1','bleu_1','bleu_2','bleu_3','bleu_4','rouge_1','rouge_2','rouge_l']

def evaluate_one(row, pred):
    gt = str(row['answer'])
    r = compute_rouge(pred, gt)
    em = ' '.join(normalize_text(pred)) == ' '.join(normalize_text(gt))
    return {
        'img_id': row['img_id'], 'complexity': int(row['complexity']),
        'question_class': row['question_class'], 'question': row['question'],
        'ground_truth': gt, 'prediction': pred, 'exact_match': em,
        'word_f1': round(compute_word_f1(pred, gt), 3),
        'bleu_1': round(compute_bleu(pred, gt, 1), 3),
        'bleu_2': round(compute_bleu(pred, gt, 2), 3),
        'bleu_3': round(compute_bleu(pred, gt, 3), 3),
        'bleu_4': round(compute_bleu(pred, gt, 4), 3),
        'rouge_1': round(r['rouge1'], 3),
        'rouge_2': round(r['rouge2'], 3),
        'rouge_l': round(r['rougeL'], 3),
        'meteor': round(compute_meteor(pred, gt), 3),
    }

def summarize(results, name):
    if not results: return {'model': name, 'error': 'none'}
    n = len(results)
    em = sum(r['exact_match'] for r in results)
    s = {'model': name, 'n': n,
         'accuracy': round(em/n*100, 1),
         'ece': round(compute_ece(
             [r['word_f1'] for r in results],
             [1.0 if r['exact_match'] else 0.0 for r in results]) * 100, 2)}
    for k in METRIC_KEYS:
        s[f'avg_{k}'] = round(np.mean([r[k] for r in results]) * 100, 1)
    s['avg_meteor'] = round(np.mean([r['meteor'] for r in results]) * 100, 1)
    return s

def print_results(s):
    print(f"\n{'='*55}")
    print(f"  {s['model']}")
    print(f"{'='*55}")
    for label, key in [('Accuracy','accuracy'),('F1','avg_word_f1'),
        ('BLEU-1','avg_bleu_1'),('BLEU-2','avg_bleu_2'),('BLEU-3','avg_bleu_3'),
        ('BLEU-4','avg_bleu_4'),('ROUGE-1','avg_rouge_1'),('ROUGE-2','avg_rouge_2'),
        ('ROUGE-L','avg_rouge_l'),('METEOR','avg_meteor'),('ECE','ece')]:
        val = s.get(key, None)
        if val is not None:
            print(f"  {label:<12} {val:>6.1f}%")
    print(f"{'='*55}")

# ============================================================================
# 4. DATA LOADING & PARTITIONING
# ============================================================================
print('Loading data...')
train_df = pd.read_csv(os.path.join(DATA_DIR, 'kvasir_vqa_x1_train.csv'))
test_df  = pd.read_csv(os.path.join(DATA_DIR, 'kvasir_vqa_x1_test.csv'))

# Filter to existing images
train_df = train_df[train_df['img_id'].apply(
    lambda x: os.path.exists(os.path.join(IMAGE_DIR, f'{x}.jpg')))].reset_index(drop=True)
test_df = test_df[test_df['img_id'].apply(
    lambda x: os.path.exists(os.path.join(IMAGE_DIR, f'{x}.jpg')))].reset_index(drop=True)

if args.max_train and len(train_df) > args.max_train:
    train_df = train_df.sample(n=args.max_train, random_state=SEED).reset_index(drop=True)
    print(f'  Subsampled to {args.max_train} training samples')

# Eval subset (same logic as train_ddp.py)
def diverse_sample(df, n, seed=42):
    parts = []
    for c in sorted(df['complexity'].unique()):
        sub = df[df['complexity']==c]
        parts.append(sub.sample(n=min(max(1, n//3), len(sub)), random_state=seed))
    return pd.concat(parts).head(n)

eval_df = diverse_sample(test_df, args.num_eval)
print(f'  Train: {len(train_df)} | Test: {len(test_df)} | Eval: {len(eval_df)}')


def partition_data(df, num_clients, mode='iid', seed=42):
    """Split training data across federated clients.

    IID:     Random shuffle, split equally.
    Non-IID: Dirichlet distribution (alpha=0.5) over question_class.
             Each client gets a skewed distribution of question types.
    """
    np.random.seed(seed)

    if mode == 'iid':
        # Simple random split
        df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        splits = np.array_split(df_shuffled, num_clients)
        return [s.reset_index(drop=True) for s in splits]

    else:  # non-iid
        # Dirichlet-based split: each client gets a skewed class distribution
        alpha = 0.5  # lower = more non-IID (0.5 is moderately heterogeneous)
        classes = sorted(df['question_class'].unique())
        client_indices = [[] for _ in range(num_clients)]

        for cls in classes:
            cls_indices = df[df['question_class'] == cls].index.tolist()
            np.random.shuffle(cls_indices)

            # Sample proportions from Dirichlet distribution
            proportions = np.random.dirichlet([alpha] * num_clients)
            proportions = (proportions * len(cls_indices)).astype(int)
            # Fix rounding error
            proportions[-1] = len(cls_indices) - proportions[:-1].sum()

            start = 0
            for i, count in enumerate(proportions):
                client_indices[i].extend(cls_indices[start:start+count])
                start += count

        return [df.loc[idx].reset_index(drop=True) for idx in client_indices]


# Split data across clients
client_dfs = partition_data(train_df, args.num_clients, args.partition, SEED)

print(f'\n  Data partition ({args.partition.upper()}):')
for i, cdf in enumerate(client_dfs):
    classes = cdf['question_class'].value_counts()
    print(f'    Client {i}: {len(cdf)} samples, {len(classes)} classes')

# ============================================================================
# 5. VQA DATASET (same as train_ddp.py)
# ============================================================================
class VQADataset(Dataset):
    def __init__(self, df, proc, img_dir, model_key, max_len=256, ans_max=64):
        self.df = df.reset_index(drop=True)
        self.proc = proc; self.img_dir = img_dir
        self.mk = model_key; self.ml = max_len; self.am = ans_max

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(os.path.join(self.img_dir, f"{row['img_id']}.jpg")).convert('RGB')
        q, a = row['question'], str(row['answer'])

        if self.mk == 'instructblip':
            prompt = f'Answer concisely. Question: {q} Answer:'
            inputs = self.proc(images=img, text=prompt, return_tensors='pt',
                               padding='max_length', max_length=self.ml, truncation=True)
            labels = self.proc.tokenizer(a, return_tensors='pt',
                                         padding='max_length', max_length=self.am, truncation=True).input_ids
            labels[labels == self.proc.tokenizer.pad_token_id] = -100
            item = {k: v.squeeze(0) for k, v in inputs.items()}
            item['labels'] = labels.squeeze(0)

        elif self.mk == 'llava_med':
            prompt = f'USER: <image>\nAnswer concisely: {q}\nASSISTANT:'
            full = f'USER: <image>\nAnswer concisely: {q}\nASSISTANT: {a}'
            inputs = self.proc(text=full, images=img, return_tensors='pt')
            prompt_inputs = self.proc(text=prompt, images=img, return_tensors='pt')
            pl = prompt_inputs['input_ids'].shape[-1]
            labels = inputs['input_ids'].clone().squeeze(0)
            labels[:pl] = -100
            if self.proc.tokenizer.pad_token_id is not None:
                labels[labels == self.proc.tokenizer.pad_token_id] = -100
            item = {k: v.squeeze(0) for k, v in inputs.items()}
            item['labels'] = labels

        return item


def collate_fn(batch):
    result = {}
    for k in batch[0].keys():
        vals = [b[k] for b in batch]
        if not isinstance(vals[0], torch.Tensor):
            result[k] = vals[0]; continue
        if len(vals) == 1:
            result[k] = vals[0].unsqueeze(0); continue
        shapes = [v.shape for v in vals]
        if all(s == shapes[0] for s in shapes):
            result[k] = torch.stack(vals)
        else:
            max_shape = [max(s[d] for s in shapes) for d in range(len(shapes[0]))]
            padded = []
            for v in vals:
                pw = []
                for d in range(len(max_shape) - 1, -1, -1):
                    pw.extend([0, max_shape[d] - v.shape[d]])
                padded.append(torch.nn.functional.pad(v, pw))
            result[k] = torch.stack(padded)
    return result

# ============================================================================
# 6. LOAD MODEL (QLoRA — same as train_ddp.py)
# ============================================================================
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

print(f'Loading {MODEL_NAME} with 4-bit quantization...')

if MODEL_KEY == 'instructblip':
    from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
    processor = InstructBlipProcessor.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    model = InstructBlipForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16,
        quantization_config=bnb_config,
        device_map={'': 0},
        cache_dir=CACHE_DIR, use_safetensors=False)

elif MODEL_KEY == 'llava_med':
    from transformers import LlavaForConditionalGeneration, AutoProcessor
    processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16,
        quantization_config=bnb_config,
        device_map={'': 0},
        cache_dir=CACHE_DIR)

if hasattr(processor, 'tokenizer') and processor.tokenizer.pad_token is None:
    processor.tokenizer.pad_token = processor.tokenizer.eos_token

print(f'  Loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB')

# ---- Apply LoRA ----
from peft import (LoraConfig, get_peft_model, TaskType,
                  prepare_model_for_kbit_training,
                  get_peft_model_state_dict, set_peft_model_state_dict)

model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=config['grad_ckpt'])

lora_cfg = LoraConfig(
    r=LORA_R, lora_alpha=LORA_ALPHA,
    target_modules=config['lora_targets'],
    lora_dropout=LORA_DROPOUT, bias='none',
    task_type=getattr(TaskType, config['lora_task']))

model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

# ============================================================================
# 7. INFERENCE & EVAL FUNCTIONS
# ============================================================================
def generate_pred(model, processor, image, question):
    """Generate a prediction for a single sample."""
    try:
        dev = next(model.parameters()).device
        if MODEL_KEY == 'instructblip':
            prompt = f'Answer this medical question concisely. Question: {question} Answer:'
            inputs = processor(images=image, text=prompt, return_tensors='pt').to(dev, torch.float16)
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
            return processor.decode(out[0], skip_special_tokens=True).strip()
        elif MODEL_KEY == 'llava_med':
            prompt = f'USER: <image>\nAnswer concisely: {question}\nASSISTANT:'
            inputs = processor(text=prompt, images=image, return_tensors='pt').to(dev, torch.float16)
            stop_ids = [processor.tokenizer.eos_token_id]
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                 eos_token_id=stop_ids)
            decoded = processor.decode(out[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
            decoded = re.sub(r'[^\x20-\x7E]+', '', decoded)
            decoded = re.sub(r'(.)\1{3,}', r'\1', decoded)
            return decoded.strip()
    except Exception as e:
        print(f'  [ERROR] {e}')
        return ''

def run_eval(model, processor, df, show_samples=5):
    """Evaluate model on dataframe."""
    results = []
    model.eval()
    with torch.no_grad():
        for idx, (_, row) in enumerate(df.iterrows()):
            img = Image.open(os.path.join(IMAGE_DIR, f"{row['img_id']}.jpg")).convert('RGB')
            pred = generate_pred(model, processor, img, row['question'])
            r = evaluate_one(row, pred)
            results.append(r)
            if idx < show_samples:
                mark = 'Y' if r['exact_match'] else '~' if r['word_f1']>=0.5 else 'X'
                print(f"  [{idx+1}] {mark} F1:{r['word_f1']:.2f} | {r['question'][:50]}")
    return results

# ============================================================================
# 8. FEDERATED AVERAGING (FedAvg)
# ============================================================================
def cosine_annealing(current_round, total_rounds, lr_max, lr_min=0.0):
    """Cosine annealing LR schedule across federated rounds."""
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * current_round / total_rounds))


def fedavg(client_states, client_sizes):
    """Weighted average of client LoRA state dicts (FedAvg).

    Weights are proportional to each client's dataset size.
    """
    total = sum(client_sizes)
    weights = [s / total for s in client_sizes]

    avg_state = {}
    for key in client_states[0]:
        avg_state[key] = sum(
            w * client_states[i][key].float()
            for i, w in enumerate(weights)
        ).to(client_states[0][key].dtype)

    return avg_state


def train_one_client(model, dataloader, lr, local_epochs, grad_accum):
    """Train model on one client's data for local_epochs.

    Returns updated LoRA state dict and average loss.
    """
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01
    )
    model.train()
    total_loss, total_steps = 0.0, 0

    for epoch in range(local_epochs):
        optimizer.zero_grad()
        for step, batch in enumerate(dataloader):
            batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / grad_accum
            loss.backward()
            total_loss += out.loss.item()
            total_steps += 1

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

        # Handle remaining steps
        if total_steps % grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

    avg_loss = total_loss / max(total_steps, 1)

    # Get updated LoRA weights (detached copies)
    state = {k: v.detach().clone().cpu()
             for k, v in get_peft_model_state_dict(model).items()}

    return state, avg_loss

# ============================================================================
# 9. CREATE CLIENT DATALOADERS
# ============================================================================
print('\nCreating client dataloaders...')
client_loaders = []
for i, cdf in enumerate(client_dfs):
    ds = VQADataset(cdf, processor, IMAGE_DIR, MODEL_KEY, MAX_SEQ_LEN, MAX_ANSWER_LEN)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)
    client_loaders.append(dl)
    print(f'  Client {i}: {len(ds)} samples, {len(dl)} batches')

# ============================================================================
# 10. FEDERATED TRAINING LOOP
# ============================================================================
print(f'\n{"="*60}')
print(f'  Starting FedAvg: {args.num_rounds} rounds, {args.num_clients} clients')
print(f'{"="*60}\n')

# Save initial global LoRA state
global_state = {k: v.detach().clone().cpu()
                for k, v in get_peft_model_state_dict(model).items()}

clients_per_round = max(1, int(args.num_clients * args.fraction_fit))
round_losses = []   # avg loss per round
round_lrs = []      # LR per round

t0 = time.time()

for rnd in range(1, args.num_rounds + 1):
    # 1. Select clients for this round
    selected = sorted(random.sample(range(args.num_clients), k=clients_per_round))

    # 2. Compute LR for this round
    lr = cosine_annealing(rnd, args.num_rounds, LR_MAX, LR_MIN)
    round_lrs.append(lr)

    # 3. Train each selected client
    client_states = []
    client_sizes = []
    client_losses = []

    for cid in selected:
        # Load global weights into model
        set_peft_model_state_dict(model, global_state)

        # Train locally
        state, loss = train_one_client(
            model, client_loaders[cid], lr, args.local_epochs, GRAD_ACCUM)

        client_states.append(state)
        client_sizes.append(len(client_dfs[cid]))
        client_losses.append(loss)

    # 4. FedAvg: aggregate client LoRA weights
    global_state = fedavg(client_states, client_sizes)

    # 5. Log
    avg_loss = np.mean(client_losses)
    round_losses.append(avg_loss)
    elapsed = (time.time() - t0) / 60

    print(f'  Round {rnd:>3}/{args.num_rounds} | '
          f'Clients: {selected} | '
          f'Loss: {avg_loss:.4f} | '
          f'LR: {lr:.2e} | '
          f'{elapsed:.1f} min')

    # 6. Save checkpoint every 5 rounds
    if rnd % 5 == 0 or rnd == args.num_rounds:
        set_peft_model_state_dict(model, global_state)
        model.save_pretrained(os.path.join(CKPT_DIR, f'round_{rnd}'))
        print(f'    Checkpoint saved: round_{rnd}')

    # Clean up
    del client_states, client_sizes, client_losses
    gc.collect()
    torch.cuda.empty_cache()

total_time = (time.time() - t0) / 60
print(f'\nFederated training done in {total_time:.1f} min')

# Load final global weights
set_peft_model_state_dict(model, global_state)

# ============================================================================
# 11. EVALUATION (Zero-Shot + Federated Fine-Tuned)
# ============================================================================
model.eval()

# 11a. Zero-Shot (disable LoRA adapter)
print(f'\n=== ZERO-SHOT: {MODEL_NAME} ===')
model.disable_adapter_layers()
zs_results = run_eval(model, processor, eval_df)
zs_summary = summarize(zs_results, f'{MODEL_NAME} (Zero-Shot)')
print_results(zs_summary)
pd.DataFrame(zs_results).to_csv(os.path.join(RESULTS_DIR, f'{TAG}_zs.csv'), index=False)

# 11b. Federated Fine-Tuned (re-enable LoRA adapter)
print(f'\n=== FEDERATED FT: {MODEL_NAME} ({args.partition.upper()}) ===')
model.enable_adapter_layers()
ft_results = run_eval(model, processor, eval_df)
ft_summary = summarize(ft_results, f'{MODEL_NAME} (FL-{args.partition.upper()})')
print_results(ft_summary)
pd.DataFrame(ft_results).to_csv(os.path.join(RESULTS_DIR, f'{TAG}_ft.csv'), index=False)

# Save summary JSON
summary = {
    'config': {
        'model': MODEL_NAME,
        'model_key': MODEL_KEY,
        'partition': args.partition,
        'num_clients': args.num_clients,
        'num_rounds': args.num_rounds,
        'fraction_fit': args.fraction_fit,
        'clients_per_round': clients_per_round,
        'local_epochs': args.local_epochs,
        'lr_max': LR_MAX,
        'lora_r': LORA_R,
        'lora_alpha': LORA_ALPHA,
        'batch_size': BATCH_SIZE,
        'grad_accum': GRAD_ACCUM,
        'total_train_samples': len(train_df),
        'eval_samples': len(eval_df),
        'total_time_min': round(total_time, 1),
    },
    'zs': zs_summary,
    'ft': ft_summary,
    'round_losses': [round(l, 4) for l in round_losses],
}
with open(os.path.join(RESULTS_DIR, f'{TAG}_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)
print(f'\nSummary saved: {TAG}_summary.json')

# ============================================================================
# 12. VISUALIZATION
# ============================================================================

# 12a. Federated Loss Curve
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(range(1, len(round_losses)+1), round_losses, 'o-', color='#3498db',
         markersize=4, label='Round Loss')
if len(round_losses) > 3:
    smooth = pd.Series(round_losses).rolling(3, min_periods=1).mean()
    ax1.plot(range(1, len(round_losses)+1), smooth, color='#e74c3c',
             linewidth=2, label='Smoothed')
ax1.set_xlabel('Round'); ax1.set_ylabel('Loss')
ax1.set_title(f'Federated Loss ({args.partition.upper()})', fontweight='bold')
ax1.legend(); ax1.grid(alpha=0.3)

# 12b. Metrics Comparison Bar Chart
metrics_to_plot = [
    ('F1', 'avg_word_f1'), ('BLEU-1', 'avg_bleu_1'), ('BLEU-4', 'avg_bleu_4'),
    ('ROUGE-L', 'avg_rouge_l'), ('METEOR', 'avg_meteor'),
]
labels = [m[0] for m in metrics_to_plot]
zs_vals = [zs_summary.get(m[1], 0) for m in metrics_to_plot]
ft_vals = [ft_summary.get(m[1], 0) for m in metrics_to_plot]

x = np.arange(len(labels))
w = 0.35
bars_zs = ax2.bar(x - w/2, zs_vals, w, label='Zero-Shot', color='#95a5a6')
bars_ft = ax2.bar(x + w/2, ft_vals, w, label=f'FL-{args.partition.upper()}', color='#27ae60')

for bar, val in zip(bars_zs, zs_vals):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
             f'{val:.1f}', ha='center', fontsize=9, color='#636e72')
for bar, val in zip(bars_ft, ft_vals):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
             f'{val:.1f}', ha='center', fontsize=9, fontweight='bold', color='#2d3436')

ax2.set_xticks(x); ax2.set_xticklabels(labels)
ax2.set_ylabel('Score (%)'); ax2.set_title('ZS vs Federated FT', fontweight='bold')
ax2.legend(); ax2.grid(axis='y', alpha=0.3)

plt.suptitle(f'{MODEL_NAME} — Federated ({args.partition.upper()}, {args.num_clients}C, {args.num_rounds}R)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, f'{TAG}_loss_metrics.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f'Plot saved: {TAG}_loss_metrics.png')

# 12c. Data Distribution Plot (Non-IID visualization)
if args.partition == 'noniid':
    fig, ax = plt.subplots(figsize=(10, 5))
    classes = sorted(train_df['question_class'].unique())
    class_counts = np.zeros((args.num_clients, len(classes)))
    for i, cdf in enumerate(client_dfs):
        for j, cls in enumerate(classes):
            class_counts[i, j] = len(cdf[cdf['question_class'] == cls])

    bottom = np.zeros(args.num_clients)
    colors = plt.cm.Set3(np.linspace(0, 1, len(classes)))
    for j, cls in enumerate(classes):
        ax.bar(range(args.num_clients), class_counts[:, j], bottom=bottom,
               label=cls[:20], color=colors[j])
        bottom += class_counts[:, j]

    ax.set_xlabel('Client ID'); ax.set_ylabel('Number of Samples')
    ax.set_title(f'Non-IID Data Distribution (Dirichlet α=0.5)', fontweight='bold')
    ax.set_xticks(range(args.num_clients))
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'{TAG}_data_dist.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Data distribution plot saved: {TAG}_data_dist.png')

# ============================================================================
# 13. CLEANUP
# ============================================================================
del model, processor
if torch.cuda.is_available(): torch.cuda.empty_cache()
gc.collect()
print(f'\nDone. Total time: {total_time:.1f} min')
