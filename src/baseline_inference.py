"""
Baseline zero-shot VQA inference using a pre-trained model (BLIP-2).

Runs inference on sample images from Kvasir-VQA-x1 WITHOUT any fine-tuning
to establish baseline performance and demonstrate hallucination behavior.

Usage:
    python src/baseline_inference.py
"""

import os
import json
import yaml
import torch
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from transformers import Blip2Processor, Blip2ForConditionalGeneration


def load_config(config_path="configs/config.yaml"):
    """Load project configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_model(config):
    """Load pre-trained BLIP-2 model and processor."""
    model_name = config["model"]["name"]
    device = config["model"]["device"]

    print(f"[INFO] Loading model: {model_name}")
    print(f"[INFO] Device: {device}")

    processor = Blip2Processor.from_pretrained(model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    model = model.to(device)
    model.eval()

    print(f"[INFO] Model loaded successfully.")
    return model, processor


def run_inference(model, processor, image, question, config):
    """
    Run VQA inference on a single image-question pair.

    Returns:
        dict: {prediction, confidence_logits}
    """
    device = config["model"]["device"]
    max_tokens = config["model"]["max_new_tokens"]

    # Prepare prompt
    prompt = f"Question: {question} Answer:"

    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt",
    ).to(device, torch.float16 if device == "cuda" else torch.float32)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            num_beams=3,
        )

    # Only decode the NEW tokens (skip the echoed prompt)
    generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    prediction = processor.decode(generated_ids, skip_special_tokens=True).strip()

    return {"prediction": prediction}


def compute_word_f1(prediction, ground_truth):
    """
    Compute word-level F1 score between prediction and ground truth.
    This is the standard VQA evaluation metric that handles paraphrasing.
    """
    pred_tokens = set(prediction.strip().lower().split())
    gt_tokens = set(ground_truth.strip().lower().split())

    if not pred_tokens or not gt_tokens:
        return 0.0

    common = pred_tokens & gt_tokens
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def main():
    config = load_config()
    data_dir = Path(config["paths"]["data_dir"])
    image_dir = Path(config["paths"]["image_dir"])
    output_dir = Path(config["paths"]["predictions_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    n_samples = config["model"]["num_inference_samples"]

    # Load test data
    test_csv = data_dir / "kvasir_vqa_x1_test.csv"
    if not test_csv.exists():
        print("[ERROR] Test data not found. Run download_dataset.py first.")
        return

    test_df = pd.read_csv(test_csv)
    sample_df = select_diverse_samples(test_df, n_samples)

    # Load model
    model, processor = load_model(config)

    # Run inference
    results = []
    exact_matches = 0
    total_f1 = 0.0
    total = 0

    print(f"\n[INFO] Running inference on {len(sample_df)} samples...\n")
    print("=" * 80)

    for idx, (_, row) in enumerate(sample_df.iterrows()):
        img_path = image_dir / f"{row['img_id']}.jpg"

        if not img_path.exists():
            print(f"[SKIP] Image not found: {row['img_id']}")
            continue

        image = Image.open(img_path).convert("RGB")
        question = row["question"]
        ground_truth = str(row["answer"])

        output = run_inference(model, processor, image, question, config)
        prediction = output["prediction"]

        # Evaluation metrics
        is_exact = prediction.strip().lower() == ground_truth.strip().lower()
        f1_score = compute_word_f1(prediction, ground_truth)

        if is_exact:
            exact_matches += 1
        total_f1 += f1_score
        total += 1

        result = {
            "img_id": row["img_id"],
            "complexity": int(row["complexity"]),
            "question_class": row["question_class"],
            "question": question,
            "ground_truth": ground_truth,
            "prediction": prediction,
            "exact_match": is_exact,
            "word_f1": round(f1_score, 3),
        }
        results.append(result)

        # Print result with F1 score
        if is_exact:
            status = "✓ EXACT"
        elif f1_score >= 0.5:
            status = "~ PARTIAL"
        else:
            status = "✗ WRONG"

        print(f"[{idx + 1}/{len(sample_df)}] {status} (F1: {f1_score:.2f})")
        print(f"  Complexity: {row['complexity']}")
        print(f"  Question:   {question[:100]}")
        print(f"  GT Answer:  {ground_truth[:80]}")
        print(f"  Predicted:  {prediction[:80]}")
        print("-" * 80)

    # Summary
    exact_acc = (exact_matches / total * 100) if total > 0 else 0
    avg_f1 = (total_f1 / total * 100) if total > 0 else 0
    partial_matches = sum(1 for r in results if r["word_f1"] >= 0.5)

    print(f"\n{'=' * 80}")
    print(f"BASELINE INFERENCE SUMMARY")
    print(f"{'=' * 80}")
    print(f"  Model:                {config['model']['name']}")
    print(f"  Total samples:        {total}")
    print(f"  Exact matches:        {exact_matches}/{total} ({exact_acc:.1f}%)")
    print(f"  Partial matches (F1≥0.5): {partial_matches}/{total}")
    print(f"  Average Word F1:      {avg_f1:.1f}%")

    # Per-complexity breakdown
    results_df = pd.DataFrame(results)
    print(f"\n  Per-Complexity:")
    for c in sorted(results_df["complexity"].unique()):
        c_df = results_df[results_df["complexity"] == c]
        c_exact = c_df["exact_match"].sum()
        c_f1 = c_df["word_f1"].mean() * 100
        print(f"    Level {c}: Exact {c_exact}/{len(c_df)}, Avg F1: {c_f1:.1f}%")

    print(f"{'=' * 80}")

    # Save results
    results_path = output_dir / "baseline_predictions.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n[INFO] Predictions saved to {results_path}")

    # Save summary
    summary = {
        "model": config["model"]["name"],
        "num_samples": total,
        "exact_match_accuracy": exact_acc,
        "average_word_f1": avg_f1,
        "exact_matches": int(exact_matches),
        "partial_matches_f1_gte_50": int(partial_matches),
        "total": total,
        "per_complexity": {},
    }
    for c in sorted(results_df["complexity"].unique()):
        c_df = results_df[results_df["complexity"] == c]
        summary["per_complexity"][f"level_{c}"] = {
            "exact_matches": int(c_df["exact_match"].sum()),
            "total": int(len(c_df)),
            "exact_accuracy": round(c_df["exact_match"].mean() * 100, 1),
            "avg_word_f1": round(c_df["word_f1"].mean() * 100, 1),
        }

    summary_path = output_dir / "baseline_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[INFO] Summary saved to {summary_path}")


if __name__ == "__main__":
    main()

