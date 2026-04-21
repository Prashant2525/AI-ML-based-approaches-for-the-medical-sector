#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unified VLM Fine-Tuning & Evaluation on Kvasir-VQA-x1
=====================================================
DDP (DistributedDataParallel) version for multi-GPU training.

Models: InstructBLIP (3.5B) | SmolVLM2 (2.2B)
System: NVIDIA DGX V100-SXM2-32GB

Usage:
    torchrun --nproc_per_node=4 train_ddp.py --model instructblip
    torchrun --nproc_per_node=4 train_ddp.py --model smolvlm2
"""

#%% 1. Imports & DDP Bootstrap
import os, json, gc, math, re, time, warnings, random, argparse
warnings.filterwarnings('ignore')
import logging
logging.getLogger('transformers').setLevel(logging.ERROR)

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import pandas as pd
import numpy as np
from PIL import Image
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
matplotlib.use('Agg')  # Non-interactive backend for scripts
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.rcParams['figure.dpi'] = 120

# ---- DDP setup ----
def setup_ddp():
    import datetime
    dist.init_process_group(backend='nccl', timeout=datetime.timedelta(hours=2))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    return local_rank

def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0

def log(msg):
    if is_main():
        print(msg, flush=True)

local_rank = setup_ddp()
world_size = dist.get_world_size()
log(f'DDP initialized: {world_size} GPUs')

#%% 2. Configuration
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='instructblip',
                    choices=['instructblip', 'smolvlm2'])
args, _ = parser.parse_known_args()
MODEL_KEY = args.model

MODEL_REGISTRY = {
    'instructblip': {
        'model_id':     'Salesforce/instructblip-flan-t5-xl',
        'model_name':   'InstructBLIP (Flan-T5-XL)',
        'model_n': 'instructblip',
        'model_type':   'encoder_decoder',
        'learning_rate': 1e-5,
        'lora_targets': ['q', 'k', 'v', 'o'],
        'lora_task':    'SEQ_2_SEQ_LM',
        'lora_r':       32,
        'lora_alpha':   64,
        'grad_ckpt':    False,
        'batch_size':   4,
        'grad_accum':   8,
    },

    'smolvlm2': {
        'model_id':     'HuggingFaceTB/SmolVLM2-2.2B-Instruct',
        'model_name':   'SmolVLM2 (2.2B)',
        'model_n':      'smolvlm2',
        'model_type':   'causal',
        'learning_rate': 1e-5,
        'lora_targets': ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        'lora_task':    'CAUSAL_LM',
        'lora_r':       32,
        'lora_alpha':   64,
        'grad_ckpt':    True,
        'batch_size':   4,
        'grad_accum':   8,
    },
}

config = MODEL_REGISTRY[MODEL_KEY]
MODEL_ID   = config['model_id']
MODEL_NAME = config['model_name']
MODEL_TYPE = config['model_type']
LORA_R     = config['lora_r']
LORA_ALPHA = config['lora_alpha']
BATCH_SIZE = config['batch_size']
# DDP: each GPU processes full batch, so reduce accum by world_size
GRAD_ACCUM = max(1, config['grad_accum'] // world_size)

MAX_TRAIN_SAMPLES = None  
NUM_EPOCHS        = 3
LEARNING_RATE     = config['learning_rate']
LORA_DROPOUT      = 0.1      
MAX_SEQ_LEN       = 256      
MAX_ANSWER_LEN    = 64       
MAX_NEW_TOKENS    = 64      
NUM_EVAL_SAMPLES  = 5000
SEED              = 42       

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

PROJECT_DIR = os.getcwd()
DATA_DIR    = os.path.join(PROJECT_DIR, 'data')
IMAGE_DIR   = os.path.join(DATA_DIR, 'images')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results', 'predictions')
CKPT_DIR = os.path.join(
    PROJECT_DIR,
    'checkpoints',
    f'{MODEL_KEY}_lora{MAX_TRAIN_SAMPLES}_ep{NUM_EPOCHS}_lr{LEARNING_RATE}_lora_r{LORA_R}_lora_alpha{LORA_ALPHA}_bs{BATCH_SIZE}_ga{GRAD_ACCUM}_eval{NUM_EVAL_SAMPLES}'
)
CACHE_DIR   = os.path.join(PROJECT_DIR, 'hf_cache')

if is_main():
    for d in [DATA_DIR, IMAGE_DIR, RESULTS_DIR, CKPT_DIR, CACHE_DIR]:
        os.makedirs(d, exist_ok=True)
dist.barrier()

os.environ['HF_HOME'] = CACHE_DIR
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

log(f'Model:    {MODEL_NAME}')
log(f'GPUs:     {world_size}')
log(f'Batch/GPU: {BATCH_SIZE}  Accum: {GRAD_ACCUM}  Eff: {BATCH_SIZE * world_size * GRAD_ACCUM}')

#%% 3. Evaluation Metrics
# 10 metrics with Porter stemming: Accuracy, F1, BLEU-1/2/3/4, ROUGE-1/2/L, ECE
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

def compute_ece(confs, accs, bins=10):
    if not confs: return 0.0
    edges = np.linspace(0, 1, bins+1)
    total, ece_val = len(confs), 0.0
    for i in range(bins):
        mask = [(edges[i] <= c < edges[i+1]) for c in confs]
        cnt = sum(mask)
        if cnt:
            ece_val += (cnt/total) * abs(
                np.mean([a for a,m in zip(accs,mask) if m]) -
                np.mean([c for c,m in zip(confs,mask) if m]))
    return ece_val

def compute_meteor(pred, gt):
    ref_tokens = normalize_text(gt)
    hyp_tokens = normalize_text(pred)
    if not ref_tokens or not hyp_tokens: return 0.0
    try: return _meteor([ref_tokens], hyp_tokens)
    except: return 0.0

def compute_risk_coverage_auc(results):
    if not results: return 0.0
    sorted_r = sorted(results, key=lambda r: r['word_f1'], reverse=True)
    n = len(sorted_r)
    risks, coverages = [], []
    cum_risk = 0.0
    for i, r in enumerate(sorted_r):
        cum_risk += (0.0 if r['exact_match'] else 1.0)
        coverages.append((i + 1) / n)
        risks.append(cum_risk / (i + 1))
    _trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    return round(_trapz(risks, coverages), 4)

def compute_auroc(results):
    if not results: return 0.0
    labels = [1 if r['exact_match'] else 0 for r in results]
    scores = [r['word_f1'] for r in results]
    if sum(labels) == 0 or sum(labels) == len(labels): return 0.0
    try:
        from sklearn.metrics import roc_auc_score
        return round(roc_auc_score(labels, scores), 4)
    except:
        pos = [s for s, l in zip(scores, labels) if l == 1]
        neg = [s for s, l in zip(scores, labels) if l == 0]
        conc = sum(1 for p in pos for n in neg if p > n)
        ties = sum(1 for p in pos for n in neg if p == n)
        total = len(pos) * len(neg)
        return round((conc + 0.5 * ties) / total, 4) if total else 0.0

def diverse_sample(df, n, seed=42):
    parts = []
    for c in sorted(df['complexity'].unique()):
        sub = df[df['complexity']==c]
        parts.append(sub.sample(n=min(max(1, n//3), len(sub)), random_state=seed))
    return pd.concat(parts).head(n)

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
    s['auroc'] = round(compute_auroc(results) * 100, 1)
    s['risk_coverage_auc'] = round(compute_risk_coverage_auc(results) * 100, 1)
    return s

def print_results(s):
    print(f"\n{'='*55}")
    print(f"  {s['model']}")
    print(f"{'='*55}")
    for label, key in [('Accuracy','accuracy'),('F1','avg_word_f1'),
        ('BLEU-1','avg_bleu_1'),('BLEU-2','avg_bleu_2'),('BLEU-3','avg_bleu_3'),
        ('BLEU-4','avg_bleu_4'),('ROUGE-1','avg_rouge_1'),('ROUGE-2','avg_rouge_2'),
        ('ROUGE-L','avg_rouge_l'),('METEOR','avg_meteor'),('ECE','ece'),
        ('AUROC','auroc'),('RC-AUC','risk_coverage_auc')]:
        val = s.get(key, None)
        if val is not None:
            print(f"  {label:<12} {val:>6.1f}%")
    print(f"{'='*55}")

log('Metrics ready.')

#%% 4. Load Data
train_df = pd.read_csv(os.path.join(DATA_DIR, 'kvasir_vqa_x1_train.csv'))
test_df  = pd.read_csv(os.path.join(DATA_DIR, 'kvasir_vqa_x1_test.csv'))

train_df = train_df[train_df['img_id'].apply(
    lambda x: os.path.exists(os.path.join(IMAGE_DIR, f'{x}.jpg')))].reset_index(drop=True)
test_df = test_df[test_df['img_id'].apply(
    lambda x: os.path.exists(os.path.join(IMAGE_DIR, f'{x}.jpg')))].reset_index(drop=True)

if MAX_TRAIN_SAMPLES is not None and len(train_df) > MAX_TRAIN_SAMPLES:
    train_df = train_df.sample(n=MAX_TRAIN_SAMPLES, random_state=SEED).reset_index(drop=True)
    log(f'Subsampled to {MAX_TRAIN_SAMPLES} training samples')
else:
    log(f'Using all {len(train_df)} training samples')

eval_df = diverse_sample(test_df, NUM_EVAL_SAMPLES)
log(f'Train: {len(train_df)} | Test: {len(test_df)} | Eval: {len(eval_df)}')

#%% 5. Load Model
# QLoRA: 4-bit quantization via bitsandbytes (saves ~60-75% VRAM)
from transformers import BitsAndBytesConfig

USE_QUANTIZATION = True  # Set False to use fp16 (original behavior)

if USE_QUANTIZATION:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,  # nested quantization for extra savings
    )
    log(f'Loading {MODEL_NAME} with 4-bit quantization (QLoRA)...')
else:
    bnb_config = None
    log(f'Loading {MODEL_NAME} in fp16...')

if MODEL_KEY == 'instructblip':
    from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
    processor = InstructBlipProcessor.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    model = InstructBlipForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16,
        quantization_config=bnb_config,
        device_map={'': local_rank} if bnb_config else None,
        cache_dir=CACHE_DIR, use_safetensors=False)

elif MODEL_KEY == 'smolvlm2':
    from transformers import AutoModelForImageTextToText, AutoProcessor
    processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    # Disable image splitting: single 384x384 patch per image (faster, less VRAM)
    processor.image_processor.do_image_splitting = False
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16,
        quantization_config=bnb_config,
        device_map={'': local_rank} if bnb_config else None,
        cache_dir=CACHE_DIR)

# Ensure pad token exists
if hasattr(processor, 'tokenizer') and processor.tokenizer.pad_token is None:
    processor.tokenizer.pad_token = processor.tokenizer.eos_token

# Move to local GPU (only needed for fp16; quantized models are already on device)
if not USE_QUANTIZATION:
    model = model.to(f'cuda:{local_rank}')
model.eval()
log(f'Loaded. GPU {local_rank}: {torch.cuda.memory_allocated(local_rank)/1e9:.1f} GB')

#%% 6. Inference Function
def generate_pred(model, processor, image, question):
    """Run on unwrapped model (single GPU, rank 0 only)."""
    try:
        dev = next(model.parameters()).device
        if MODEL_KEY == 'instructblip':
            prompt = f'Answer this medical question concisely. Question: {question} Answer:'
            inputs = processor(images=image, text=prompt, return_tensors='pt').to(dev, torch.float16)
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
            return processor.decode(out[0], skip_special_tokens=True).strip()

        elif MODEL_KEY == 'smolvlm2':
            prompt = f'<|im_start|>user\n<image>Answer concisely: {question}<end_of_utterance>\n<|im_start|>assistant\n'
            inputs = processor(text=prompt, images=[image], return_tensors='pt',
                               padding=False, truncation=False).to(dev, torch.float16)
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
            decoded = processor.decode(out[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
            decoded = re.sub(r'[^\x20-\x7E]+', '', decoded)
            return decoded.strip()
    except Exception as e:
        print(f'[ERROR] {e}')
        return ''

def run_eval(model, processor, df):
    results = []
    for idx, (_, row) in enumerate(df.iterrows()):
        img = Image.open(os.path.join(IMAGE_DIR, f"{row['img_id']}.jpg")).convert('RGB')
        pred = generate_pred(model, processor, img, row['question'])
        r = evaluate_one(row, pred)
        results.append(r)
        if idx < 5 or r['exact_match']:
            mark = 'Y' if r['exact_match'] else '~' if r['word_f1']>=0.5 else 'X'
            print(f"  [{idx+1}] {mark} F1:{r['word_f1']:.2f} B1:{r['bleu_1']:.2f} | {r['question'][:45]}")
    return results

log('Inference ready.')

# NOTE: Zero-shot eval moved to AFTER training (after DDP destroy).
# With LoRA, we disable the adapter to get zero-shot predictions.

#%% 8. Training Dataset
# InstructBLIP (encoder-decoder): input=question, labels=answer
# SmolVLM2 (causal): input=prompt+answer, labels=answer only (prompt masked)

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

        elif self.mk == 'smolvlm2':
            prompt = f'<|im_start|>user\n<image>Answer concisely: {q}<end_of_utterance>\n<|im_start|>assistant\n'
            full   = f'{prompt}{a}<end_of_utterance>'
            inputs = self.proc(text=full, images=[img], return_tensors='pt',
                               padding=False, truncation=False)
            # Mask from the END: answer tokens + eos are the only trainable part
            ans_ids = self.proc.tokenizer.encode(a, add_special_tokens=False)
            eos_id  = self.proc.tokenizer.convert_tokens_to_ids('<end_of_utterance>')
            answer_len = len(ans_ids) + 1  # +1 for <end_of_utterance>
            labels = inputs['input_ids'].clone().squeeze(0)
            labels[:-answer_len] = -100
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
            result[k] = vals[0]
            continue
        if len(vals) == 1:
            result[k] = vals[0].unsqueeze(0)
            continue
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


train_ds = VQADataset(train_df, processor, IMAGE_DIR, MODEL_KEY, MAX_SEQ_LEN, MAX_ANSWER_LEN)
val_df_sub = train_df.sample(n=min(100, len(train_df)), random_state=99)
val_ds = VQADataset(val_df_sub, processor, IMAGE_DIR, MODEL_KEY, MAX_SEQ_LEN, MAX_ANSWER_LEN)

# DDP: DistributedSampler splits data across GPUs
train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=dist.get_rank(),
                                   shuffle=True, seed=SEED)
val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=dist.get_rank(),
                                 shuffle=False)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=train_sampler,
                      collate_fn=collate_fn, num_workers=0)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, sampler=val_sampler,
                    collate_fn=collate_fn, num_workers=0)
log(f'Train batches/GPU: {len(train_dl)} | Val batches/GPU: {len(val_dl)}')
log(f'Batch/GPU: {BATCH_SIZE} | Accum: {GRAD_ACCUM} | Eff: {BATCH_SIZE * world_size * GRAD_ACCUM}')

#%% 9. Apply LoRA
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

# QLoRA: prepare quantized model for training (freezes quantized layers, casts LoRA to fp32)
if USE_QUANTIZATION:
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=config['grad_ckpt'])
    if config['grad_ckpt']:
        log('Gradient checkpointing: ON (via prepare_model_for_kbit_training)')
elif config['grad_ckpt']:
    model.gradient_checkpointing_enable()
    log('Gradient checkpointing: ON')
else:
    log('Gradient checkpointing: OFF (not supported by this model)')

lora_cfg = LoraConfig(
    r=LORA_R, lora_alpha=LORA_ALPHA,
    target_modules=config['lora_targets'],
    lora_dropout=LORA_DROPOUT, bias='none',
    task_type=getattr(TaskType, config['lora_task']))

model = get_peft_model(model, lora_cfg)
if is_main():
    model.print_trainable_parameters()
log(f'LoRA: r={LORA_R}, alpha={LORA_ALPHA}, targets={config["lora_targets"]}')

#%% 10. Wrap with DDP
# DDP wraps model AFTER LoRA is applied
model = DDP(model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True, static_graph=True)
log(f'DDP wrapped on {world_size} GPUs')

#%% 11. Training Loop
# Cosine LR, gradient clipping, early stopping (patience=3).
from transformers import get_scheduler

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
total_steps = (len(train_dl) * NUM_EPOCHS) // GRAD_ACCUM
warmup = int(total_steps * 0.1)
scheduler = get_scheduler('cosine', optimizer, num_warmup_steps=warmup, num_training_steps=total_steps)

step_losses, ep_train, ep_val = [], [], []
best_vl = float('inf'); patience = 3; no_imp = 0

log(f'\n{"="*60}')
log(f'  Epochs: {NUM_EPOCHS}  Batch/GPU: {BATCH_SIZE}  Accum: {GRAD_ACCUM}  '
    f'EffBatch: {BATCH_SIZE * world_size * GRAD_ACCUM}')
log(f'  LR: {LEARNING_RATE}  Warmup: {warmup}  Steps: {total_steps}  Scheduler: cosine')
log(f'  LoRA r={LORA_R} alpha={LORA_ALPHA} targets={config["lora_targets"]}')
log(f'{"="*60}\n')

model.train()
t0 = time.time()

for epoch in range(NUM_EPOCHS):
    train_sampler.set_epoch(epoch)  # DDP: shuffle differently each epoch
    el, nb_steps = 0.0, 0
    pbar = tqdm(train_dl, desc=f'Epoch {epoch+1}', disable=not is_main())
    for step, batch in enumerate(pbar):
        batch = {k: v.to(local_rank) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        out = model(**batch)
        loss = out.loss

        # NaN guard: skip bad batches instead of poisoning the whole run
        if torch.isnan(loss) or torch.isinf(loss):
            if is_main():
                print(f'  [WARN] NaN/Inf loss at step {step}, skipping batch')
            optimizer.zero_grad()
            continue

        # DDP averages gradients automatically; we still divide by GRAD_ACCUM for accumulation
        (loss / GRAD_ACCUM).backward()
        el += loss.item(); nb_steps += 1

        if (step + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
            step_losses.append(loss.item())
            if is_main():
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{scheduler.get_last_lr()[0]:.2e}'})

    ep_train.append(el / max(nb_steps, 1))

    # Validation
    model.eval()
    vl, vb = 0.0, 0
    with torch.no_grad():
        for batch in val_dl:
            batch = {k: v.to(local_rank) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            vl += model(**batch).loss.item(); vb += 1

    # Average val loss across GPUs
    vl_tensor = torch.tensor([vl, float(vb)], device=f'cuda:{local_rank}')
    dist.all_reduce(vl_tensor, op=dist.ReduceOp.SUM)
    avg_vl = vl_tensor[0].item() / max(vl_tensor[1].item(), 1)
    ep_val.append(avg_vl)

    cur_lr = scheduler.get_last_lr()[0]
    log(f'  Epoch {epoch+1}: Train={ep_train[-1]:.4f} | Val={avg_vl:.4f} | LR={cur_lr:.2e}')
    if avg_vl < best_vl:
        best_vl = avg_vl; no_imp = 0
        if is_main():
            model.module.save_pretrained(CKPT_DIR)  # .module unwraps DDP
            log(f'  Best model saved (val={best_vl:.4f})')
    else:
        no_imp += 1
        if no_imp >= patience:
            log(f'  Early stopping at epoch {epoch+1}'); break
    model.train()

log(f'\nDone in {(time.time()-t0)/60:.1f} min | Best val: {best_vl:.4f}')

# --- DDP is no longer needed. Destroy it so non-rank-0 processes exit cleanly. ---
eval_model = model.module  # unwrap DDP before destroying process group
_my_rank = dist.get_rank()  # save rank before destroying
dist.barrier()  # final sync — ensure all ranks finished training
dist.destroy_process_group()

if _my_rank != 0:
    # Non-rank-0: cleanup and exit
    del model, processor, eval_model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    gc.collect()
    log('Done. GPU cleared.')
    import sys; sys.exit(0)

# === From here on, only rank 0 continues (no DDP, no barriers, no timeouts) ===

eval_model.eval()

#%% 12a. Zero-Shot Evaluation (disable LoRA adapter)
log(f'=== ZERO-SHOT: {MODEL_NAME} ===')
eval_model.disable_adapter_layers()
zs_results = run_eval(eval_model, processor, eval_df)
zs_summary = summarize(zs_results, f'{MODEL_NAME} (Zero-Shot)')
print_results(zs_summary)
pd.DataFrame(zs_results).to_csv(os.path.join(RESULTS_DIR, f'{MODEL_KEY}_zs.csv'), index=False)

#%% 12b. Fine-Tuned Evaluation (re-enable LoRA adapter)
log(f'=== FINE-TUNED: {MODEL_NAME} ===')
eval_model.enable_adapter_layers()
ft_results = run_eval(eval_model, processor, eval_df)
ft_summary = summarize(ft_results, f'{MODEL_NAME} (Fine-Tuned)')
print_results(ft_summary)
pd.DataFrame(ft_results).to_csv(os.path.join(RESULTS_DIR, f'{MODEL_KEY}_ft.csv'), index=False)
with open(os.path.join(RESULTS_DIR, f'{MODEL_KEY}_summary.json'), 'w') as f:
    json.dump({'zs': zs_summary, 'ft': ft_summary}, f, indent=2)

#%% 13. BERTScore (rank 0 only — heavy model)
if is_main():
    try:
        from bert_score import score as bert_score_fn
        log('Computing BERTScore...')
        zs_preds = [r['prediction'] for r in zs_results]
        zs_gts   = [r['ground_truth'] for r in zs_results]
        ft_preds = [r['prediction'] for r in ft_results]
        ft_gts   = [r['ground_truth'] for r in ft_results]
        _, _, zs_bf1 = bert_score_fn(zs_preds, zs_gts, model_type='distilbert-base-uncased',
                                      batch_size=32, verbose=False, device=f'cuda:{local_rank}')
        _, _, ft_bf1 = bert_score_fn(ft_preds, ft_gts, model_type='distilbert-base-uncased',
                                      batch_size=32, verbose=False, device=f'cuda:{local_rank}')
        for r, bs in zip(zs_results, zs_bf1.tolist()): r['bertscore_f1'] = round(bs, 3)
        for r, bs in zip(ft_results, ft_bf1.tolist()): r['bertscore_f1'] = round(bs, 3)
        zs_summary['avg_bertscore'] = round(zs_bf1.mean().item() * 100, 1)
        ft_summary['avg_bertscore'] = round(ft_bf1.mean().item() * 100, 1)
        log(f'  ZS BERTScore: {zs_summary["avg_bertscore"]}% | FT: {ft_summary["avg_bertscore"]}%')
    except Exception as e:
        log(f'BERTScore skipped: {e}')

#%% 14. Visualization & Export (rank 0 only)
if is_main():
    # Loss curves
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    a1.plot(step_losses, alpha=0.4, color='#3498db')
    if len(step_losses) > 10:
        w = max(5, len(step_losses)//20)
        a1.plot(pd.Series(step_losses).rolling(w, min_periods=1).mean(),
                color='#e74c3c', linewidth=2, label='Smoothed')
    a1.set_xlabel('Step'); a1.set_ylabel('Loss')
    a1.set_title('Training Loss', fontweight='bold'); a1.legend(); a1.grid(alpha=0.3)
    ep_x = range(1, len(ep_train)+1)
    a2.plot(ep_x, ep_train, 'o-', color='#3498db', label='Train')
    a2.plot(ep_x, ep_val, 's-', color='#e74c3c', label='Val')
    a2.set_xlabel('Epoch'); a2.set_ylabel('Loss')
    a2.set_title('Train vs Val', fontweight='bold'); a2.legend(); a2.grid(alpha=0.3)
    plt.suptitle(f'{MODEL_NAME} Loss', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'{MODEL_KEY}_loss.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Metrics comparison
    labels = ['Acc','F1','B1','B2','B3','B4','R1','R2','RL','MET','BERT','ECE','AUROC','RC-AUC']
    keys = ['accuracy','avg_word_f1','avg_bleu_1','avg_bleu_2','avg_bleu_3','avg_bleu_4',
            'avg_rouge_1','avg_rouge_2','avg_rouge_l','avg_meteor','avg_bertscore','ece','auroc','risk_coverage_auc']
    zv = [zs_summary.get(k,0) for k in keys]
    fv = [ft_summary.get(k,0) for k in keys]
    x = np.arange(len(labels)); w = 0.35
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.bar(x-w/2, zv, w, label='Zero-Shot', color='#95a5a6')
    ax.bar(x+w/2, fv, w, label='Fine-Tuned', color='#27ae60')
    for i,(z,f) in enumerate(zip(zv,fv)):
        ax.text(i-w/2, z+0.2, f'{z:.1f}', ha='center', fontsize=6, rotation=45)
        ax.text(i+w/2, f+0.2, f'{f:.1f}', ha='center', fontsize=6, fontweight='bold', rotation=45)
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel('Score (%)')
    ax.set_title(f'{MODEL_NAME}: ZS vs FT (DDP {world_size} GPUs)', fontweight='bold', fontsize=14)
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'{MODEL_KEY}_metrics.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Prediction grids
    def prediction_grid(results, title, fname, n=6):
        show = results[:n]
        if not show: return
        cols = min(3, len(show)); rows = math.ceil(len(show)/cols)
        fig, axes = plt.subplots(rows, cols, figsize=(7*cols, 6*rows))
        if rows==1 and cols==1: axes = np.array([axes])
        axes = np.atleast_2d(axes)
        fig.suptitle(title, fontsize=16, fontweight='bold', y=1.01)
        for i, r in enumerate(show):
            ri, ci = divmod(i, cols); ax = axes[ri][ci]
            p = os.path.join(IMAGE_DIR, f"{r['img_id']}.jpg")
            if os.path.exists(p): ax.imshow(Image.open(p).convert('RGB'))
            ax.set_xticks([]); ax.set_yticks([])
            if r['exact_match']: st, cl = 'EXACT', '#27ae60'
            elif r['word_f1'] >= 0.5: st, cl = 'PARTIAL', '#f39c12'
            else: st, cl = 'WRONG', '#e74c3c'
            txt = f"{st} | F1:{r['word_f1']:.2f} B1:{r['bleu_1']:.2f}\nQ:{r['question'][:85]}\nGT:{r['ground_truth'][:65]}\nP:{r['prediction'][:65]}"
            ax.text(0.02, 0.98, txt, transform=ax.transAxes, fontsize=7, va='top', color=cl,
                    fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.75))
        for i in range(len(show), rows*cols):
            axes[divmod(i,cols)[0]][divmod(i,cols)[1]].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, fname), dpi=150, bbox_inches='tight')
        plt.close()

    prediction_grid(zs_results, f'{MODEL_NAME} - Zero-Shot', f'{MODEL_KEY}_zs_grid.png')
    prediction_grid(ft_results, f'{MODEL_NAME} - Fine-Tuned', f'{MODEL_KEY}_ft_grid.png')
    log('Prediction grids saved.')

    # BLEU & ROUGE Breakdown
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    bn = ['BLEU-1','BLEU-2','BLEU-3','BLEU-4']
    bz = [zs_summary.get(f'avg_bleu_{i}',0) for i in range(1,5)]
    bf = [ft_summary.get(f'avg_bleu_{i}',0) for i in range(1,5)]
    bx = np.arange(4); bw = 0.35
    a1.bar(bx-bw/2, bz, bw, label='ZS', color='#bdc3c7')
    a1.bar(bx+bw/2, bf, bw, label='FT', color='#2ecc71')
    for j,(z,f) in enumerate(zip(bz,bf)):
        a1.text(j-bw/2, z+0.3, f'{z:.1f}', ha='center', fontsize=8)
        a1.text(j+bw/2, f+0.3, f'{f:.1f}', ha='center', fontsize=8, fontweight='bold')
    a1.set_xticks(bx); a1.set_xticklabels(bn)
    a1.set_title('BLEU', fontweight='bold'); a1.legend(); a1.grid(axis='y', alpha=0.3)
    rn = ['ROUGE-1','ROUGE-2','ROUGE-L']
    rz = [zs_summary.get(k,0) for k in ['avg_rouge_1','avg_rouge_2','avg_rouge_l']]
    rf = [ft_summary.get(k,0) for k in ['avg_rouge_1','avg_rouge_2','avg_rouge_l']]
    rx = np.arange(3)
    a2.bar(rx-bw/2, rz, bw, label='ZS', color='#bdc3c7')
    a2.bar(rx+bw/2, rf, bw, label='FT', color='#e67e22')
    for j,(z,f) in enumerate(zip(rz,rf)):
        a2.text(j-bw/2, z+0.3, f'{z:.1f}', ha='center', fontsize=8)
        a2.text(j+bw/2, f+0.3, f'{f:.1f}', ha='center', fontsize=8, fontweight='bold')
    a2.set_xticks(rx); a2.set_xticklabels(rn)
    a2.set_title('ROUGE', fontweight='bold'); a2.legend(); a2.grid(axis='y', alpha=0.3)
    plt.suptitle(f'{MODEL_NAME} - Text Metrics', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'{MODEL_KEY}_bleu_rouge.png'), dpi=150, bbox_inches='tight')
    plt.close()
    log('BLEU/ROUGE breakdown saved.')

    # Per-Complexity Breakdown
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (res, nm, cl) in zip(axes, [(zs_results,'Zero-Shot','#95a5a6'),(ft_results,'Fine-Tuned','#27ae60')]):
        rdf = pd.DataFrame(res)
        lvls = sorted(rdf['complexity'].unique())
        f1 = [rdf[rdf['complexity']==l]['word_f1'].mean()*100 for l in lvls]
        em = [rdf[rdf['complexity']==l]['exact_match'].mean()*100 for l in lvls]
        xx = np.arange(len(lvls)); w = 0.35
        ax.bar(xx-w/2, em, w, label='Accuracy', color=cl, alpha=0.7)
        ax.bar(xx+w/2, f1, w, label='F1', color=cl)
        ax.set_xlabel('Complexity'); ax.set_ylabel('Score (%)')
        ax.set_title(nm, fontweight='bold')
        ax.set_xticks(xx); ax.set_xticklabels([f'L{l}' for l in lvls])
        ax.legend(); ax.set_ylim(0, 100)
    plt.suptitle(f'{MODEL_NAME} - Performance by Complexity', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'{MODEL_KEY}_complexity.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Heatmap
    ft_df_res = pd.DataFrame(ft_results)
    if ft_df_res['question_class'].nunique() > 1:
        pv = ft_df_res.pivot_table(values='word_f1', index='question_class', columns='complexity', aggfunc='mean') * 100
        fig, ax = plt.subplots(figsize=(8, max(4, len(pv)*0.5+1)))
        sns.heatmap(pv, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax, linewidths=0.5)
        ax.set_title(f'{MODEL_NAME} - F1 by Class x Complexity', fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f'{MODEL_KEY}_heatmap.png'), dpi=150, bbox_inches='tight')
        plt.close()
    log('Complexity & heatmap saved.')

    # Calibration Plot
    def cal_plot(results, title, ax):
        co = [r['word_f1'] for r in results]
        ac = [1.0 if r['exact_match'] else 0.0 for r in results]
        edges = np.linspace(0,1,11); bc, ba = [], []
        for i in range(10):
            m = [(edges[i]<=c<edges[i+1]) for c in co]; cnt = sum(m)
            bc.append(np.mean([c for c,x in zip(co,m) if x]) if cnt else (edges[i]+edges[i+1])/2)
            ba.append(np.mean([a for a,x in zip(ac,m) if x]) if cnt else 0)
        ax.bar(range(10), ba, color='#3498db', alpha=0.7, label='Accuracy')
        ax.plot(range(10), bc, 'r--o', markersize=4, label='Confidence')
        ax.set_title(title, fontweight='bold'); ax.legend(fontsize=8); ax.set_ylim(0,1.1)

    fig, (a1,a2) = plt.subplots(1, 2, figsize=(14, 5))
    cal_plot(zs_results, f'Zero-Shot (ECE={zs_summary.get("ece",0):.2f}%)', a1)
    cal_plot(ft_results, f'Fine-Tuned (ECE={ft_summary.get("ece",0):.2f}%)', a2)
    plt.suptitle(f'{MODEL_NAME} Calibration', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'{MODEL_KEY}_cal.png'), dpi=150, bbox_inches='tight')
    plt.close()
    log('Calibration plot saved.')

    # Print comparison table
    log(f"\n{'Metric':<8} {'ZS':>8} {'FT':>8} {'Delta':>8}")
    log('-'*35)
    for n,z,f in zip(labels,zv,fv):
        d = f - z
        if n in ('ECE','RC-AUC'):
            tag = 'BETTER' if d < 0 else 'WORSE'
        else:
            tag = 'BETTER' if d > 0 else 'WORSE'
        log(f"  {n:<8} {z:>7.1f}% {f:>7.1f}% {d:>+7.1f}   {tag}")

    # Final report
    imp = sum(1 for n,z,f in zip(labels,zv,fv)
              if (f < z if n in ('ECE','RC-AUC') else f > z))
    log(f'\n  {imp}/{len(labels)} metrics improved.')

    # Save full comparison JSON
    comparison = {
        'model': MODEL_NAME, 'model_key': MODEL_KEY,
        'zero_shot': zs_summary, 'fine_tuned': ft_summary,
        'config': {'lr': LEARNING_RATE, 'epochs_run': len(ep_train),
                   'lora_r': LORA_R, 'lora_targets': config['lora_targets'],
                   'train_samples': len(train_df), 'best_val_loss': best_vl,
                   'batch_size': BATCH_SIZE, 'grad_accum': GRAD_ACCUM,
                   'world_size': world_size,
                   'effective_batch': BATCH_SIZE * world_size * GRAD_ACCUM}
    }
    with open(os.path.join(RESULTS_DIR, f'{MODEL_KEY}_comparison.json'), 'w') as f:
        json.dump(comparison, f, indent=2)

    # Update CSVs with BERTScore
    pd.DataFrame(zs_results).to_csv(os.path.join(RESULTS_DIR, f'{MODEL_KEY}_zs.csv'), index=False)
    pd.DataFrame(ft_results).to_csv(os.path.join(RESULTS_DIR, f'{MODEL_KEY}_ft.csv'), index=False)

    log(f'\nAll results saved to: {RESULTS_DIR}')

#%% 15. Extended Evaluation Metrics: CHRF++, BLEURT, BERT-F1
# ============================================================================
# EXTENDED METRICS: CHRF++ | BLEURT | BERT-F1
# Computed from existing CSV files — no re-training needed
# ============================================================================

# ---- Configuration ----
MODEL_KEY = config['model_id']
MODEL_NAME = config['model_name']
MODEL_N = config['model_n']
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results', 'predictions')
DEVICE = 'cuda:4'  # GPU for BERT-F1 / BLEURT

# ---- Load existing CSV files ----
zs_path = os.path.join(RESULTS_DIR, f'{MODEL_N}_zs.csv')
ft_path = os.path.join(RESULTS_DIR, f'{MODEL_N}_ft.csv')

assert os.path.exists(zs_path), f"ZS CSV not found: {zs_path}"
assert os.path.exists(ft_path), f"FT CSV not found: {ft_path}"

zs_df = pd.read_csv(zs_path)
ft_df = pd.read_csv(ft_path)

# Fill NaN predictions with empty string (edge case safety)
zs_df['prediction'] = zs_df['prediction'].fillna('')
zs_df['ground_truth'] = zs_df['ground_truth'].fillna('')
ft_df['prediction'] = ft_df['prediction'].fillna('')
ft_df['ground_truth'] = ft_df['ground_truth'].fillna('')

zs_preds = zs_df['prediction'].tolist()
zs_refs  = zs_df['ground_truth'].tolist()
ft_preds = ft_df['prediction'].tolist()
ft_refs  = ft_df['ground_truth'].tolist()

print(f"Loaded {len(zs_preds)} ZS samples, {len(ft_preds)} FT samples")

# ---- Storage for results ----
extended_metrics = {
    'model': MODEL_NAME,
    'model_key': MODEL_KEY,
    'n_zs': len(zs_preds),
    'n_ft': len(ft_preds),
    'zero_shot': {},
    'fine_tuned': {},
}

# ====================================================================
# 1. CHRF++ (Character F-score with word bigrams)
# ====================================================================
print("\n" + "="*60)
print("  Computing CHRF++...")
print("="*60)

try:
    from sacrebleu.metrics import CHRF
    chrf = CHRF(word_order=2)  # word_order=2 makes it chrF++

    # Per-sample computation
    zs_chrf_scores = []
    ft_chrf_scores = []

    for pred, ref in zip(zs_preds, zs_refs):
        score = chrf.sentence_score(pred, [ref]).score
        zs_chrf_scores.append(round(score, 2))

    for pred, ref in zip(ft_preds, ft_refs):
        score = chrf.sentence_score(pred, [ref]).score
        ft_chrf_scores.append(round(score, 2))

    # Add per-sample scores to DataFrames
    zs_df['chrf_pp'] = zs_chrf_scores
    ft_df['chrf_pp'] = ft_chrf_scores

    # Aggregate
    zs_chrf_avg = round(np.mean(zs_chrf_scores), 1)
    ft_chrf_avg = round(np.mean(ft_chrf_scores), 1)

    extended_metrics['zero_shot']['chrf_pp'] = zs_chrf_avg
    extended_metrics['fine_tuned']['chrf_pp'] = ft_chrf_avg

    print(f"  ZS CHRF++: {zs_chrf_avg}%")
    print(f"  FT CHRF++: {ft_chrf_avg}%")
    print(f"  Delta:     {ft_chrf_avg - zs_chrf_avg:+.1f}")

except ImportError:
    print("  [!] sacrebleu not installed. Run: python -m pip install sacrebleu")
    extended_metrics['zero_shot']['chrf_pp'] = None
    extended_metrics['fine_tuned']['chrf_pp'] = None

# ====================================================================
# 2. BERT-F1 (BERTScore F1 using distilbert)
# ====================================================================
print("\n" + "="*60)
print("  Computing BERT-F1...")
print("="*60)

try:
    from bert_score import score as bert_score_fn

    # Compute BERTScore F1
    _, _, zs_bf1 = bert_score_fn(
        zs_preds, zs_refs,
        model_type='distilbert-base-uncased',
        batch_size=64,
        verbose=True,
        device=DEVICE
    )
    _, _, ft_bf1 = bert_score_fn(
        ft_preds, ft_refs,
        model_type='distilbert-base-uncased',
        batch_size=64,
        verbose=True,
        device=DEVICE
    )

    # Per-sample scores
    zs_bert_scores = [round(s, 3) for s in zs_bf1.tolist()]
    ft_bert_scores = [round(s, 3) for s in ft_bf1.tolist()]

    zs_df['bert_f1'] = zs_bert_scores
    ft_df['bert_f1'] = ft_bert_scores

    # Aggregate (as percentage)
    zs_bert_avg = round(np.mean(zs_bert_scores) * 100, 1)
    ft_bert_avg = round(np.mean(ft_bert_scores) * 100, 1)

    extended_metrics['zero_shot']['bert_f1'] = zs_bert_avg
    extended_metrics['fine_tuned']['bert_f1'] = ft_bert_avg

    print(f"  ZS BERT-F1: {zs_bert_avg}%")
    print(f"  FT BERT-F1: {ft_bert_avg}%")
    print(f"  Delta:      {ft_bert_avg - zs_bert_avg:+.1f}")

except Exception as e:
    print(f"  [!] BERTScore failed: {e}")
    print("  Run: pip install bert-score")
    extended_metrics['zero_shot']['bert_f1'] = None
    extended_metrics['fine_tuned']['bert_f1'] = None

# ====================================================================
# 3. BLEURT (Learned evaluation metric)
# ====================================================================
print("\n" + "="*60)
print("  Computing BLEURT...")
print("="*60)

bleurt_computed = False
try:
    from bleurt import score as bleurt_score

    # Use BLEURT-20 (lighter checkpoint)
    checkpoint = "/raid/home/dgxuser40/Prashant/BLEURT-20"
    scorer = bleurt_score.BleurtScorer(checkpoint)

    zs_bleurt_scores = scorer.score(references=zs_refs, candidates=zs_preds)
    ft_bleurt_scores = scorer.score(references=ft_refs, candidates=ft_preds)

    zs_bleurt_scores = [round(s, 3) for s in zs_bleurt_scores]
    ft_bleurt_scores = [round(s, 3) for s in ft_bleurt_scores]

    zs_df['bleurt'] = zs_bleurt_scores
    ft_df['bleurt'] = ft_bleurt_scores

    zs_bleurt_avg = round(np.mean(zs_bleurt_scores), 3)
    ft_bleurt_avg = round(np.mean(ft_bleurt_scores), 3)

    extended_metrics['zero_shot']['bleurt'] = zs_bleurt_avg
    extended_metrics['fine_tuned']['bleurt'] = ft_bleurt_avg
    bleurt_computed = True

    print(f"  ZS BLEURT: {zs_bleurt_avg}")
    print(f"  FT BLEURT: {ft_bleurt_avg}")
    print(f"  Delta:     {ft_bleurt_avg - zs_bleurt_avg:+.3f}")

except ImportError:
    print("  [!] BLEURT not installed. This is optional.")
    print("  To install:python -m pip install git+https://github.com/google-research/bleurt.git")
    print("  Note: Requires ~2GB download. Skipping gracefully.")
    extended_metrics['zero_shot']['bleurt'] = None
    extended_metrics['fine_tuned']['bleurt'] = None
except Exception as e:
    print(f"  [!] BLEURT failed: {e}")
    print("  Skipping BLEURT. Other metrics are unaffected.")
    extended_metrics['zero_shot']['bleurt'] = None
    extended_metrics['fine_tuned']['bleurt'] = None

# ====================================================================
# 4. Save updated CSVs with new per-sample columns
# ====================================================================
zs_df.to_csv(zs_path, index=False)
ft_df.to_csv(ft_path, index=False)
print(f"\nUpdated CSVs saved with new metric columns.")

# ====================================================================
# 5. Comparison Table
# ====================================================================
print("\n" + "="*60)
print(f"  Extended Metrics: {MODEL_NAME}")
print("="*60)
print(f"  {'Metric':<12} {'ZS':>8} {'FT':>8} {'Delta':>8}")
print(f"  {'-'*40}")

zs_m = extended_metrics['zero_shot']
ft_m = extended_metrics['fine_tuned']

for label, key, fmt in [
    ('CHRF++', 'chrf_pp', '.1f'),
    ('BERT-F1', 'bert_f1', '.1f'),
    ('BLEURT', 'bleurt', '.3f'),
]:
    zv = zs_m.get(key)
    fv = ft_m.get(key)
    if zv is not None and fv is not None:
        delta = fv - zv
        better = "BETTER" if delta > 0 else "WORSE" if delta < 0 else "SAME"
        print(f"  {label:<12} {zv:>8{fmt}} {fv:>8{fmt}} {delta:>+8{fmt}}   {better}")
    else:
        print(f"  {label:<12} {'N/A':>8} {'N/A':>8} {'N/A':>8}   SKIPPED")

print(f"  {'-'*40}")

# ====================================================================
# 6. Bar Chart: ZS vs FT for Extended Metrics
# ====================================================================
plot_labels = []
plot_zs = []
plot_ft = []

for label, key in [('CHRF++', 'chrf_pp'), ('BERT-F1', 'bert_f1'), ('BLEURT', 'bleurt')]:
    zv = zs_m.get(key)
    fv = ft_m.get(key)
    if zv is not None and fv is not None:
        plot_labels.append(label)
        # Normalize BLEURT to percentage-like scale for visual comparison
        if key == 'bleurt':
            plot_zs.append(zv * 100)  # BLEURT is -1 to 1, scale to %
            plot_ft.append(fv * 100)
        else:
            plot_zs.append(zv)
            plot_ft.append(fv)

if plot_labels:
    x = np.arange(len(plot_labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars_zs = ax.bar(x - w/2, plot_zs, w, label='Zero-Shot', color='#95a5a6', edgecolor='white')
    bars_ft = ax.bar(x + w/2, plot_ft, w, label='Fine-Tuned', color='#27ae60', edgecolor='white')

    # Add value labels on bars
    for bar, val in zip(bars_zs, plot_zs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10, color='#636e72')
    for bar, val in zip(bars_ft, plot_ft):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold', color='#2d3436')

    ax.set_xticks(x)
    ax.set_xticklabels(plot_labels, fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'{MODEL_NAME}: Extended Metrics — ZS vs FT', fontweight='bold', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(max(plot_zs), max(plot_ft)) * 1.15)

    plt.tight_layout()
    chart_path = os.path.join(RESULTS_DIR, f'{MODEL_N}_extended_metrics.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nBar chart saved: {chart_path}")
else:
    print("\nNo metrics computed — skipping chart.")

# ====================================================================
# 7. Save Extended Metrics JSON (separate file, does NOT touch existing JSON)
# ====================================================================
json_path = os.path.join(RESULTS_DIR, f'{MODEL_N}_extended_metrics.json')
with open(json_path, 'w') as f:
    json.dump(extended_metrics, f, indent=2)
print(f"Extended metrics JSON saved: {json_path}")

# ====================================================================
# 8. Also update the main comparison JSON if it exists (non-destructive)
# ====================================================================
main_json_path = os.path.join(RESULTS_DIR, f'{MODEL_N}_comparison.json')
if os.path.exists(main_json_path):
    try:
        with open(main_json_path, 'r') as f:
            comparison = json.load(f)

        # Add extended metrics under new keys (never overwrites existing keys)
        for period, data in [('zero_shot', zs_m), ('fine_tuned', ft_m)]:
            if period in comparison:
                for key in ['chrf_pp', 'bert_f1', 'bleurt']:
                    if key not in comparison[period] and data.get(key) is not None:
                        comparison[period][key] = data[key]

        with open(main_json_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        print(f"Main comparison JSON updated (non-destructive): {main_json_path}")
    except Exception as e:
        print(f"Could not update main JSON (no harm done): {e}")

main_json_path_1 = os.path.join(RESULTS_DIR, f'{MODEL_N}_summary.json')
if os.path.exists(main_json_path_1):
    try:
        with open(main_json_path_1, 'r') as f:
            comparison = json.load(f)

        # Add extended metrics under new keys (never overwrites existing keys)
        for period, data in [('zs', zs_m), ('ft', ft_m)]:
            if period in comparison:
                for key in ['chrf_pp', 'bert_f1', 'bleurt']:
                    if key not in comparison[period] and data.get(key) is not None:
                        comparison[period][key] = data[key]

        with open(main_json_path_1, 'w') as f:
            json.dump(comparison, f, indent=2)
        print(f"Main summary JSON updated (non-destructive): {main_json_path_1}")
    except Exception as e:
        print(f"Could not update main JSON (no harm done): {e}")

print("\nExtended metrics computation complete.")


#%% 16. Cleanup (rank 0 only — DDP already destroyed)
del model, processor, eval_model
if torch.cuda.is_available(): torch.cuda.empty_cache()
gc.collect()
log('Done. GPU cleared.')