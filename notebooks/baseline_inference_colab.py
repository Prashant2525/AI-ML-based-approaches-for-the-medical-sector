"""
Kvasir-VQA-x1 Baseline Inference — Google Colab Version
========================================================
Run this in Google Colab with GPU runtime enabled.

Steps:
1. Go to Runtime > Change runtime type > select GPU (T4 is fine)
2. Mount Google Drive (if using Option A)
3. Run all cells

Option A: Upload project to Google Drive first, then mount.
Option B: Download everything fresh in Colab.
"""

# ============================================================
# CELL 1: Setup — Install dependencies
# ============================================================
# !pip install -q datasets transformers accelerate torch torchvision pandas pyyaml tqdm Pillow

# ============================================================
# CELL 2: Choose your data source
# ============================================================
# Set USE_DRIVE = True if you uploaded project to Google Drive
# Set USE_DRIVE = False to download fresh in Colab

USE_DRIVE = True

import os

if USE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    
    # CHANGE THIS to your actual Google Drive path
    PROJECT_DIR = "/content/drive/MyDrive/AI-ML-based-approaches-for-the-medical-sector"
    DATA_DIR = os.path.join(PROJECT_DIR, "data")
    IMAGE_DIR = os.path.join(DATA_DIR, "images")
    RESULTS_DIR = os.path.join(PROJECT_DIR, "results", "predictions")
else:
    PROJECT_DIR = "/content/medical-vqa"
    DATA_DIR = os.path.join(PROJECT_DIR, "data")
    IMAGE_DIR = os.path.join(DATA_DIR, "images")
    RESULTS_DIR = os.path.join(PROJECT_DIR, "results", "predictions")

os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# CELL 3: Download dataset (Only if USE_DRIVE = False)
# ============================================================
if not USE_DRIVE:
    import json
    from pathlib import Path
    from datasets import load_dataset
    from tqdm.auto import tqdm
    import pandas as pd

    os.makedirs(IMAGE_DIR, exist_ok=True)

    # Download images
    print("[INFO] Downloading images from SimulaMet-HOST/Kvasir-VQA...")
    ds_host = load_dataset("SimulaMet-HOST/Kvasir-VQA", split="raw")
    seen = set()
    for row in tqdm(ds_host, desc="Saving images"):
        if row["img_id"] not in seen:
            row["image"].save(os.path.join(IMAGE_DIR, f"{row['img_id']}.jpg"))
            seen.add(row["img_id"])
    print(f"[INFO] Saved {len(seen)} images")

    # Download QA pairs
    for split in ["train", "test"]:
        print(f"[INFO] Downloading {split} split...")
        ds = load_dataset("SimulaMet/Kvasir-VQA-x1", split=split)
        records = []
        for row in tqdm(ds, desc=f"Processing {split}"):
            records.append({
                "img_id": row["img_id"],
                "complexity": row["complexity"],
                "question": row["question"],
                "answer": row["answer"],
                "question_class": row["question_class"],
                "original": json.dumps(row["original"]) if row.get("original") else ""
            })
        df = pd.DataFrame(records)
        df.to_csv(os.path.join(DATA_DIR, f"kvasir_vqa_x1_{split}.csv"), index=False)
        print(f"[INFO] Saved {len(df)} {split} samples")

# ============================================================
# CELL 4: Run Baseline Inference
# ============================================================
import json
import torch
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# --- Configuration ---
MODEL_NAME = "Salesforce/blip2-opt-2.7b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 64
NUM_SAMPLES = 20

print(f"[INFO] Device: {DEVICE}")
print(f"[INFO] CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

# --- Load Model ---
print(f"\n[INFO] Loading model: {MODEL_NAME}")
processor = Blip2Processor.from_pretrained(MODEL_NAME)
model = Blip2ForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
)
model = model.to(DEVICE)
model.eval()
print("[INFO] Model loaded successfully!")

# --- Load Test Data ---
test_csv = os.path.join(DATA_DIR, "kvasir_vqa_x1_test.csv")
test_df = pd.read_csv(test_csv)
print(f"\n[INFO] Test set: {len(test_df)} samples")

# --- Select Diverse Samples ---
samples = []
for complexity in sorted(test_df["complexity"].unique()):
    subset = test_df[test_df["complexity"] == complexity]
    n = max(1, NUM_SAMPLES // 3)
    samples.append(subset.sample(n=min(n, len(subset)), random_state=42))
sample_df = pd.concat(samples).head(NUM_SAMPLES)

# --- Run Inference ---
results = []
correct = 0
total = 0

print(f"\n[INFO] Running inference on {len(sample_df)} samples...\n")
print("=" * 80)

for idx, (_, row) in enumerate(sample_df.iterrows()):
    img_path = os.path.join(IMAGE_DIR, f"{row['img_id']}.jpg")
    
    if not os.path.exists(img_path):
        print(f"[SKIP] Image not found: {row['img_id']}")
        continue
    
    image = Image.open(img_path).convert("RGB")
    question = row["question"]
    ground_truth = str(row["answer"])
    
    # Prepare prompt
    prompt = f"Question: {question} Answer:"
    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt",
    ).to(DEVICE, torch.float16 if DEVICE == "cuda" else torch.float32)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            num_beams=3,
        )
    
    prediction = processor.decode(outputs[0], skip_special_tokens=True).strip()
    
    # Exact match check
    is_correct = prediction.strip().lower() == ground_truth.strip().lower()
    if is_correct:
        correct += 1
    total += 1
    
    result = {
        "img_id": row["img_id"],
        "complexity": int(row["complexity"]),
        "question_class": row["question_class"],
        "question": question,
        "ground_truth": ground_truth,
        "prediction": prediction,
        "exact_match": is_correct,
    }
    results.append(result)
    
    # Print
    status = "CORRECT" if is_correct else "WRONG"
    print(f"[{idx+1}/{len(sample_df)}] {status}")
    print(f"  Image:      {row['img_id']}")
    print(f"  Complexity: {row['complexity']}")
    print(f"  Question:   {question[:100]}")
    print(f"  GT Answer:  {ground_truth[:80]}")
    print(f"  Predicted:  {prediction[:80]}")
    print("-" * 80)

# ============================================================
# CELL 5: Results Summary
# ============================================================
accuracy = (correct / total * 100) if total > 0 else 0

print(f"\n{'=' * 80}")
print(f"BASELINE INFERENCE SUMMARY")
print(f"{'=' * 80}")
print(f"  Model:           {MODEL_NAME}")
print(f"  Total samples:   {total}")
print(f"  Exact matches:   {correct}")
print(f"  Accuracy:        {accuracy:.1f}%")

# Per-complexity breakdown
results_df = pd.DataFrame(results)
print(f"\n  Per-Complexity Accuracy:")
for c in sorted(results_df["complexity"].unique()):
    c_df = results_df[results_df["complexity"] == c]
    c_acc = c_df["exact_match"].mean() * 100
    print(f"    Level {c}: {c_df['exact_match'].sum()}/{len(c_df)} ({c_acc:.1f}%)")
print(f"{'=' * 80}")

# Save results
results_path = os.path.join(RESULTS_DIR, "baseline_predictions.csv")
results_df.to_csv(results_path, index=False)
print(f"\n[INFO] Predictions saved to {results_path}")

# Save summary JSON
summary = {
    "model": MODEL_NAME,
    "num_samples": total,
    "exact_match_accuracy": accuracy,
    "correct": int(correct),
    "total": int(total),
    "per_complexity": {}
}
for c in sorted(results_df["complexity"].unique()):
    c_df = results_df[results_df["complexity"] == c]
    summary["per_complexity"][f"level_{c}"] = {
        "correct": int(c_df["exact_match"].sum()),
        "total": int(len(c_df)),
        "accuracy": round(c_df["exact_match"].mean() * 100, 1)
    }

summary_path = os.path.join(RESULTS_DIR, "baseline_summary.json")
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"[INFO] Summary saved to {summary_path}")

print("\n[DONE] Baseline inference complete!")
