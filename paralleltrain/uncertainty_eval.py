#!/usr/bin/env python
"""
Uncertainty-Aware Evaluation for DDP-Trained VLMs
=================================================
Standalone script: loads a LoRA checkpoint, runs entropy + MC Dropout,
computes abstention threshold, and generates safety plots.

Usage:
    python paralleltrain/uncertainty_eval.py --model smolvlm2 --eval_samples 500 --gpu 0
"""
import os, json, re, time, random, argparse, warnings, gc
warnings.filterwarnings('ignore')
import torch
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from collections import Counter
from tqdm.auto import tqdm
import nltk
nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score as _meteor
from nltk.stem import PorterStemmer
from rouge_score import rouge_scorer
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── CLI ──
parser = argparse.ArgumentParser(description='Uncertainty eval for DDP-trained VLMs')
parser.add_argument('--model', type=str, default='smolvlm2', choices=['instructblip', 'smolvlm2'])
parser.add_argument('--checkpoint_dir', type=str, default=None, help='Path to LoRA adapter dir')
parser.add_argument('--eval_samples', type=int, default=500)
parser.add_argument('--mc_passes', type=int, default=5)
parser.add_argument('--target_coverage', type=float, default=0.80)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--max_new_tokens', type=int, default=64)
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

MODEL_KEY = args.model
DEVICE = f'cuda:{args.gpu}'
SEED = args.seed
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

MODEL_REGISTRY = {
    'instructblip': {
        'model_id': 'Salesforce/instructblip-flan-t5-xl',
        'model_name': 'InstructBLIP (Flan-T5-XL)',
        'model_type': 'encoder_decoder',
    },
    'smolvlm2': {
        'model_id': 'HuggingFaceTB/SmolVLM2-2.2B-Instruct',
        'model_name': 'SmolVLM2 (2.2B)',
        'model_type': 'causal',
    },
}
config = MODEL_REGISTRY[MODEL_KEY]
MODEL_ID = config['model_id']
MODEL_NAME = config['model_name']
MODEL_TYPE = config['model_type']

PROJECT_DIR = os.getcwd()
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
IMAGE_DIR = os.path.join(DATA_DIR, 'images')
CACHE_DIR = os.path.join(PROJECT_DIR, 'hf_cache')

# Auto-detect checkpoint
if args.checkpoint_dir:
    CKPT_DIR = args.checkpoint_dir
else:
    mr = Path(PROJECT_DIR) / 'paralleltrain' / 'checkpoints'
    candidates = sorted(mr.glob(f'{MODEL_KEY}_*'), key=lambda p: p.stat().st_mtime, reverse=True)
    CKPT_DIR = str(candidates[0]) if candidates else None
    if CKPT_DIR is None:
        raise FileNotFoundError(f'No checkpoint found for {MODEL_KEY} in {mr}')

OUTPUT_DIR = os.path.join(CKPT_DIR, 'results', 'uncertainty')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f'Model:      {MODEL_NAME}')
print(f'Checkpoint: {CKPT_DIR}')
print(f'Eval:       {args.eval_samples} samples, {args.mc_passes} MC passes')
print(f'Device:     {DEVICE}')
print(f'Output:     {OUTPUT_DIR}')

# ── Metrics (from train_ddp.py) ──
bleu_smoother = SmoothingFunction().method1
rouge_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
stemmer = PorterStemmer()

def normalize_text(text):
    text = str(text).strip().lower()
    text = re.sub(r'[^\w\s]', '', text)
    return [stemmer.stem(w) for w in text.split()]

def normalize_simple(text):
    text = str(text).strip().lower()
    return re.sub(r'[^\w\s]', '', re.sub(r'\s+', ' ', text))

def compute_word_f1(pred, gt):
    p, g = set(normalize_text(pred)), set(normalize_text(gt))
    if not p or not g: return 0.0
    c = p & g
    return 2 * len(c) / (len(p) + len(g)) if c else 0.0

def compute_bleu(pred, gt, n):
    ref, hyp = normalize_text(gt), normalize_text(pred)
    if not ref or not hyp: return 0.0
    w = tuple([1.0/n]*n + [0.0]*(4-n))
    try: return sentence_bleu([ref], hyp, weights=w, smoothing_function=bleu_smoother)
    except: return 0.0

def compute_rouge(pred, gt):
    if not pred.strip() or not gt.strip():
        return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    s = rouge_obj.score(gt.strip().lower(), pred.strip().lower())
    return {k: s[k].fmeasure for k in ['rouge1', 'rouge2', 'rougeL']}

def compute_meteor(pred, gt):
    ref_t, hyp_t = normalize_text(gt), normalize_text(pred)
    if not ref_t or not hyp_t: return 0.0
    try: return _meteor([ref_t], hyp_t)
    except: return 0.0

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

def diverse_sample(df, n, seed=42):
    parts = []
    for c in sorted(df['complexity'].unique()):
        sub = df[df['complexity']==c]
        parts.append(sub.sample(n=min(max(1, n//3), len(sub)), random_state=seed))
    return pd.concat(parts).head(n)

# ── Load Data ──
train_csv = os.path.join(DATA_DIR, 'kvasir_vqa_x1_train.csv')
test_csv = os.path.join(DATA_DIR, 'kvasir_vqa_x1_test.csv')
test_df = pd.read_csv(test_csv)
test_df = test_df[test_df['img_id'].apply(
    lambda x: os.path.exists(os.path.join(IMAGE_DIR, f'{x}.jpg')))].reset_index(drop=True)
eval_df = diverse_sample(test_df, args.eval_samples, SEED)
print(f'Eval subset: {len(eval_df)} samples (stratified)')

# ── Load Model + LoRA ──
print(f'\nLoading {MODEL_NAME} with 4-bit quantization...')
from transformers import BitsAndBytesConfig
from peft import PeftModel

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)

if MODEL_KEY == 'instructblip':
    from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
    processor = InstructBlipProcessor.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    base_model = InstructBlipForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, quantization_config=bnb_config,
        device_map={'': args.gpu}, cache_dir=CACHE_DIR, use_safetensors=False)
elif MODEL_KEY == 'smolvlm2':
    from transformers import AutoModelForImageTextToText, AutoProcessor
    processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    processor.image_processor.do_image_splitting = False
    base_model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, quantization_config=bnb_config,
        device_map={'': args.gpu}, cache_dir=CACHE_DIR)

if hasattr(processor, 'tokenizer') and processor.tokenizer.pad_token is None:
    processor.tokenizer.pad_token = processor.tokenizer.eos_token

model = PeftModel.from_pretrained(base_model, CKPT_DIR)
model.eval()
print(f'LoRA loaded from {CKPT_DIR}')
print(f'GPU mem: {torch.cuda.memory_allocated(args.gpu)/1e9:.1f} GB')

# ── Prompt Helpers ──
def make_prompt(question):
    if MODEL_KEY == 'instructblip':
        return f'Answer this medical question concisely. Question: {question} Answer:'
    else:
        return f'<|im_start|>user\n<image>Answer concisely: {question}<end_of_utterance>\n<|im_start|>assistant\n'

def prepare_inputs(image, question):
    prompt = make_prompt(question)
    if MODEL_KEY == 'instructblip':
        return processor(images=image, text=prompt, return_tensors='pt').to(DEVICE, torch.float16)
    else:
        return processor(text=prompt, images=[image], return_tensors='pt',
                         padding=False, truncation=False).to(DEVICE, torch.float16)

def decode_output(output_ids, input_ids):
    if MODEL_KEY == 'instructblip':
        return processor.decode(output_ids, skip_special_tokens=True).strip()
    else:
        new_tokens = output_ids[input_ids.shape[-1]:]
        decoded = processor.decode(new_tokens, skip_special_tokens=True)
        return re.sub(r'[^\x20-\x7E]+', '', decoded).strip()

# ── Uncertainty Functions ──
def get_entropy_and_confidence(model, image, question):
    inputs = prepare_inputs(image, question)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                             do_sample=False, output_scores=True, return_dict_in_generate=True)
    prediction = decode_output(out.sequences[0], inputs.get('input_ids', torch.tensor([[]])))

    entropies, log_probs = [], []
    gen_ids = out.sequences[0]
    if MODEL_TYPE == 'causal' and 'input_ids' in inputs:
        gen_ids = gen_ids[inputs['input_ids'].shape[-1]:]

    for step, score in enumerate(out.scores):
        probs = torch.softmax(score[0], dim=-1)
        log_p = torch.log(probs.clamp(min=1e-10))
        entropies.append(-(probs * log_p).sum().item())
        if step < len(gen_ids):
            log_probs.append(log_p[gen_ids[step]].item())

    entropy_mean = float(np.mean(entropies)) if entropies else 0.0
    confidence = float(np.exp(np.mean(log_probs))) if log_probs else 0.0
    return prediction, entropy_mean, confidence

def enable_dropout(m):
    for mod in m.modules():
        if isinstance(mod, torch.nn.Dropout): mod.train()

def disable_dropout(m):
    for mod in m.modules():
        if isinstance(mod, torch.nn.Dropout): mod.eval()

def mc_dropout_inference(model, image, question, n_passes):
    inputs = prepare_inputs(image, question)
    input_ids = inputs.get('input_ids', torch.tensor([[]]))
    enable_dropout(model)
    answers = []
    for _ in range(n_passes):
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        answers.append(decode_output(gen[0], input_ids))
    disable_dropout(model)

    pw = [compute_word_f1(answers[i], answers[j])
          for i in range(len(answers)) for j in range(i+1, len(answers))]
    mc_unc = 1.0 - (np.mean(pw) if pw else 1.0)
    normed = [normalize_simple(a) for a in answers]
    majority = Counter(normed).most_common(1)[0][0]
    pred = answers[normed.index(majority)]
    return pred, answers, float(mc_unc), len(set(normed))/len(normed)

# ── Main Evaluation Loop ──
print(f'\n{"="*60}')
print(f'  UNCERTAINTY EVALUATION: {MODEL_NAME}')
print(f'  {args.eval_samples} samples x (1 greedy + {args.mc_passes} MC) = ~{args.eval_samples*(1+args.mc_passes)} generations')
print(f'{"="*60}\n')

results = []
start = time.time()
for idx, (_, row) in enumerate(tqdm(eval_df.iterrows(), total=len(eval_df), desc='Uncertainty eval')):
    img_path = os.path.join(IMAGE_DIR, f"{row['img_id']}.jpg")
    if not os.path.exists(img_path): continue
    image = Image.open(img_path).convert('RGB')
    question = str(row['question'])
    gt = str(row['answer'])
    comp = int(row.get('complexity', 1))

    try:
        pred_g, entropy, confidence = get_entropy_and_confidence(model, image, question)
        pred_mc, mc_ans, mc_unc, unique_ratio = mc_dropout_inference(model, image, question, args.mc_passes)
    except Exception as e:
        print(f'  [SKIP] {e}')
        continue

    prediction = pred_mc
    entropy_norm = min(entropy / 10.0, 1.0)
    combined = 0.4 * entropy_norm + 0.3 * mc_unc + 0.3 * (1.0 - confidence)

    r = compute_rouge(prediction, gt)
    results.append({
        'img_id': row['img_id'], 'question': question, 'ground_truth': gt,
        'prediction': prediction, 'complexity': comp,
        'exact_match': ' '.join(normalize_text(prediction)) == ' '.join(normalize_text(gt)),
        'word_f1': round(compute_word_f1(prediction, gt), 3),
        'bleu_1': round(compute_bleu(prediction, gt, 1), 3),
        'bleu_2': round(compute_bleu(prediction, gt, 2), 3),
        'bleu_3': round(compute_bleu(prediction, gt, 3), 3),
        'bleu_4': round(compute_bleu(prediction, gt, 4), 3),
        'rouge_1': round(r['rouge1'], 3), 'rouge_2': round(r['rouge2'], 3), 'rouge_l': round(r['rougeL'], 3),
        'meteor': round(compute_meteor(prediction, gt), 3),
        'entropy': round(entropy, 4), 'confidence': round(confidence, 4),
        'mc_uncertainty': round(mc_unc, 4), 'mc_unique_ratio': round(unique_ratio, 3),
        'combined_uncertainty': round(combined, 4),
    })

    if idx < 3 or idx % 50 == 0:
        tag = 'Y' if results[-1]['exact_match'] else '~' if results[-1]['word_f1']>=0.5 else 'X'
        print(f"  [{idx+1}] {tag} F1:{results[-1]['word_f1']:.2f} Unc:{combined:.3f} | {question[:50]}")

elapsed = time.time() - start
print(f'\nDone: {len(results)} samples in {elapsed/60:.1f}min ({elapsed/max(1,len(results)):.1f}s/sample)')

if len(results) == 0:
    print('No results! Exiting.'); exit(1)

# ── BERTScore & CHRF++ (post-loop, matches train_ddp.py) ──
print('\nComputing BERTScore...')
try:
    from bert_score import score as bert_score_fn
    preds_list = [r['prediction'] for r in results]
    gts_list = [r['ground_truth'] for r in results]
    _, _, bf1 = bert_score_fn(preds_list, gts_list, model_type='distilbert-base-uncased',
                               batch_size=32, verbose=False, device=DEVICE)
    for r, bs in zip(results, bf1.tolist()): r['bertscore_f1'] = round(bs, 3)
    avg_bertscore = round(bf1.mean().item() * 100, 1)
    print(f'  BERTScore F1: {avg_bertscore}%')
except Exception as e:
    print(f'  BERTScore skipped: {e}')
    for r in results: r['bertscore_f1'] = 0.0
    avg_bertscore = 0.0

print('Computing CHRF++...')
try:
    from sacrebleu.metrics import CHRF
    chrf = CHRF(word_order=2)
    chrf_scores = [round(chrf.sentence_score(r['prediction'], [r['ground_truth']]).score, 2) for r in results]
    for r, cs in zip(results, chrf_scores): r['chrf_pp'] = cs
    avg_chrf = round(np.mean(chrf_scores), 1)
    print(f'  CHRF++: {avg_chrf}')
except Exception as e:
    print(f'  CHRF++ skipped: {e}')
    for r in results: r['chrf_pp'] = 0.0
    avg_chrf = 0.0

print('Computing BLEURT...')
try:
    from bleurt import score as bleurt_score
    checkpoint = os.path.join(os.path.expanduser('~'), 'BLEURT-20')
    if not os.path.exists(checkpoint):
        checkpoint = '/raid/home/dgxuser40/Prashant/BLEURT-20'
    scorer = bleurt_score.BleurtScorer(checkpoint)
    bleurt_scores = [round(s, 3) for s in scorer.score(
        references=[r['ground_truth'] for r in results],
        candidates=[r['prediction'] for r in results])]
    for r, bs in zip(results, bleurt_scores): r['bleurt'] = bs
    avg_bleurt = round(np.mean(bleurt_scores), 3)
    print(f'  BLEURT: {avg_bleurt}')
except Exception as e:
    print(f'  BLEURT skipped: {e}')
    avg_bleurt = None

# ── Abstention & Safety Metrics ──
f1_arr = np.array([r['word_f1'] for r in results])
ent_arr = np.array([r['entropy'] for r in results])
conf_arr = np.array([r['confidence'] for r in results])
mc_arr = np.array([r['mc_uncertainty'] for r in results])
unc_arr = np.array([r['combined_uncertainty'] for r in results])
comps = [r['complexity'] for r in results]

print('\nCorrelation (uncertainty vs F1):')
for name, vals in [('Entropy', ent_arr), ('MC Dropout', mc_arr),
                   ('1-Confidence', 1-conf_arr), ('Combined', unc_arr)]:
    r_val = np.corrcoef(vals, f1_arr)[0,1]
    print(f'  {name:<15} r = {r_val:+.3f}')

# Threshold tuning
best_t, best_acc, best_cov = unc_arr.max(), 0.0, 1.0
for t in np.linspace(unc_arr.min(), unc_arr.max(), 100):
    mask = unc_arr <= t
    cov = mask.sum() / len(unc_arr)
    sel = f1_arr[mask].mean() if mask.sum() > 0 else 0
    if cov >= args.target_coverage and sel > best_acc:
        best_t, best_acc, best_cov = t, sel, cov

answered = [i for i,u in enumerate(unc_arr) if u <= best_t]
abstained = [i for i,u in enumerate(unc_arr) if u > best_t]
sel_f1 = best_acc * 100
overall_f1 = f1_arr.mean() * 100

print(f'\n{"="*60}')
print(f'  ABSTENTION RESULTS')
print(f'{"="*60}')
print(f'  Threshold:    {best_t:.4f}')
print(f'  Coverage:     {best_cov*100:.1f}% ({len(answered)}/{len(results)})')
print(f'  Selective F1: {sel_f1:.1f}%')
print(f'  Overall F1:   {overall_f1:.1f}%')
print(f'  Gain:         +{sel_f1 - overall_f1:.1f}%')

# AUROC
binary = (f1_arr >= 0.5).astype(int)
inc_idx, cor_idx = np.where(binary==0)[0], np.where(binary==1)[0]
if len(inc_idx) > 0 and len(cor_idx) > 0:
    conc = sum(1 if unc_arr[i]>unc_arr[j] else 0.5 if unc_arr[i]==unc_arr[j] else 0
               for i in inc_idx for j in cor_idx)
    auroc = conc / (len(inc_idx)*len(cor_idx))
else:
    auroc = 0.5

# Risk-Coverage
si = np.argsort(unc_arr)
sf = f1_arr[si]
coverages = [(n+1)/len(sf) for n in range(len(sf))]
sel_accs = [sf[:n+1].mean() for n in range(len(sf))]
_trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
auc_risk = float(_trapz([1-a for a in sel_accs], coverages))

# ECE
ece = compute_ece(conf_arr.tolist(), f1_arr.tolist())
ece_data = []
bins_e = np.linspace(0,1,11)
for i in range(10):
    mask = (conf_arr>=bins_e[i]) & (conf_arr<bins_e[i+1]) if i<9 else (conf_arr>=bins_e[i]) & (conf_arr<=bins_e[i+1])
    nb = mask.sum()
    if nb==0: ece_data.append((0,0,0)); continue
    ece_data.append((float(conf_arr[mask].mean()), float(f1_arr[mask].mean()), int(nb)))

# Selective accuracy table
sel_table = {}
for t_cov in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    n = max(1, int(t_cov * len(sf)))
    sel_table[t_cov] = float(sf[:n].mean())

print(f'\n{"="*60}')
print(f'  SAFETY METRICS')
print(f'{"="*60}')
print(f'  AUROC:    {auroc:.3f}')
print(f'  AUC-Risk: {auc_risk:.3f}')
print(f'  ECE:      {ece:.3f}')
print(f'\n  Selective F1 at coverage:')
for k,v in sel_table.items():
    mark = ' <- target' if abs(k-args.target_coverage)<0.01 else ''
    print(f'    {k*100:>5.0f}% -> {v*100:.1f}%{mark}')

# ── Safety Plots ──
fig, axes = plt.subplots(2, 2, figsize=(14, 11))

ax = axes[0,0]
ax.plot(coverages, [a*100 for a in sel_accs], 'b-', lw=2, label='Selective F1')
ax.axhline(y=overall_f1, color='r', ls='--', alpha=.7, label=f'Overall ({overall_f1:.1f}%)')
ax.axvline(x=args.target_coverage, color='g', ls=':', alpha=.7, label=f'Target ({args.target_coverage*100:.0f}%)')
ax.set_xlabel('Coverage'); ax.set_ylabel('Selective Word F1 (%)'); ax.set_title('Risk-Coverage')
ax.legend(fontsize=9); ax.grid(True, alpha=.3)

ax = axes[0,1]
colors = ['#2ecc71' if f>=0.5 else '#e74c3c' for f in f1_arr]
ax.scatter(unc_arr, f1_arr*100, c=colors, alpha=.7, s=50, edgecolors='w', lw=.5)
ax.axvline(x=best_t, color='orange', ls='--', lw=2, label=f'tau={best_t:.3f}')
ax.set_xlabel('Combined Uncertainty'); ax.set_ylabel('Word F1 (%)'); ax.set_title('Uncertainty vs Quality')
ax.legend(fontsize=9); ax.grid(True, alpha=.3)

ax = axes[1,0]
bc = [d[0] for d in ece_data if d[2]>0]
ba = [d[1] for d in ece_data if d[2]>0]
if bc: ax.bar(bc, ba, width=.08, alpha=.7, color='#3498db', label='Actual F1')
ax.plot([0,1],[0,1], 'r--', alpha=.7, label='Perfect'); ax.set_xlim(-.05,1.05); ax.set_ylim(-.05,1.05)
ax.set_xlabel('Confidence'); ax.set_ylabel('Accuracy'); ax.set_title(f'Reliability (ECE={ece:.3f})')
ax.legend(fontsize=9); ax.grid(True, alpha=.3)

ax = axes[1,1]
for lvl in sorted(set(comps)):
    lvl_u = [unc_arr[i] for i,c in enumerate(comps) if c==lvl]
    ax.hist(lvl_u, bins=15, alpha=.5, label=f'L{lvl} (n={len(lvl_u)})')
ax.axvline(x=best_t, color='orange', ls='--', lw=2, label=f'tau={best_t:.3f}')
ax.set_xlabel('Combined Uncertainty'); ax.set_ylabel('Count'); ax.set_title('Uncertainty by Complexity')
ax.legend(fontsize=9); ax.grid(True, alpha=.3)

plt.suptitle(f'{MODEL_NAME} — Uncertainty-Aware Safety Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'safety_plots.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f'\nPlot saved: {OUTPUT_DIR}/safety_plots.png')

# ── Save Results ──
unc_summary = {
    'model': MODEL_NAME, 'model_key': MODEL_KEY,
    'method': 'LoRA + uncertainty abstention',
    'checkpoint': os.path.basename(CKPT_DIR),
    'mc_dropout_passes': args.mc_passes,
    'target_coverage': args.target_coverage,
    'eval_samples': len(results),
    'elapsed_min': round(elapsed/60, 1),
    'vqa_metrics': {
        'accuracy': round(sum(r['exact_match'] for r in results)/len(results)*100, 2),
        'word_f1': round(overall_f1, 2),
        'bleu_1': round(np.mean([r['bleu_1'] for r in results])*100, 2),
        'bleu_2': round(np.mean([r['bleu_2'] for r in results])*100, 2),
        'bleu_3': round(np.mean([r['bleu_3'] for r in results])*100, 2),
        'bleu_4': round(np.mean([r['bleu_4'] for r in results])*100, 2),
        'rouge_1': round(np.mean([r['rouge_1'] for r in results])*100, 2),
        'rouge_2': round(np.mean([r['rouge_2'] for r in results])*100, 2),
        'rouge_l': round(np.mean([r['rouge_l'] for r in results])*100, 2),
        'meteor': round(np.mean([r['meteor'] for r in results])*100, 2),
        'bertscore': avg_bertscore,
        'chrf_pp': avg_chrf,
        'bleurt': avg_bleurt,
    },
    'safety_metrics': {'auroc': round(auroc,4), 'auc_risk': round(auc_risk,4), 'ece': round(ece,4), 'risk_coverage_auc': round(auc_risk*100,1)},
    'abstention': {
        'threshold': round(best_t,4), 'coverage': round(best_cov,4),
        'selective_f1': round(sel_f1,2), 'overall_f1': round(overall_f1,2),
        'n_answered': len(answered), 'n_abstained': len(abstained),
    },
    'selective_accuracy': {f'{int(k*100)}pct': round(v*100,2) for k,v in sel_table.items()},
}
with open(os.path.join(OUTPUT_DIR, 'uncertainty_summary.json'), 'w') as f:
    json.dump(unc_summary, f, indent=2)

unc_df = pd.DataFrame(results)
unc_df['abstained'] = unc_arr > best_t
unc_df.to_csv(os.path.join(OUTPUT_DIR, 'uncertainty_predictions.csv'), index=False)

# Update comparison JSON (non-destructive)
comp_path = os.path.join(CKPT_DIR, 'results', 'predictions', f'{MODEL_KEY}_comparison.json')
if os.path.exists(comp_path):
    try:
        with open(comp_path) as f: comp = json.load(f)
        comp['uncertainty'] = unc_summary
        with open(comp_path, 'w') as f: json.dump(comp, f, indent=2)
        print(f'Updated: {comp_path}')
    except: pass

print(f'\nAll results saved to: {OUTPUT_DIR}')
print(f'\n{"="*60}')
print(f'  FINAL SUMMARY: {MODEL_NAME}')
print(f'{"="*60}')
print(f'  Overall F1:   {overall_f1:.1f}%')
print(f'  Selective F1:  {sel_f1:.1f}%  (+{sel_f1-overall_f1:.1f}%)')
print(f'  AUROC:         {auroc:.3f}')
print(f'  Coverage:      {best_cov*100:.1f}%  ({len(answered)} answered, {len(abstained)} abstained)')
print(f'{"="*60}')

del model, base_model, processor
if torch.cuda.is_available(): torch.cuda.empty_cache()
gc.collect()
print('GPU cleared. Done!')
