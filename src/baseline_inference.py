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

    prediction = processor.decode(outputs[0], skip_special_tokens=True).strip()

    return {"prediction": prediction}


def select_diverse_samples(df, n_samples, seed=42):
    """Select diverse samples across complexity levels and question classes."""
    samples = []

    for complexity in sorted(df["complexity"].unique()):
        subset = df[df["complexity"] == complexity]
        n_per_complexity = max(1, n_samples // 3)
        sample = subset.sample(n=min(n_per_complexity, len(subset)), random_state=seed)
        samples.append(sample)

    result = pd.concat(samples).head(n_samples)
    return result


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
    correct = 0
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

        # Simple exact match check
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

        # Print result
        status = "✓" if is_correct else "✗"
        print(f"[{idx + 1}/{len(sample_df)}] {status}")
        print(f"  Image:      {row['img_id']}")
        print(f"  Complexity: {row['complexity']}")
        print(f"  Question:   {question[:100]}")
        print(f"  GT Answer:  {ground_truth[:80]}")
        print(f"  Predicted:  {prediction[:80]}")
        print("-" * 80)

    # Summary
    accuracy = (correct / total * 100) if total > 0 else 0
    print(f"\n{'=' * 80}")
    print(f"BASELINE INFERENCE SUMMARY")
    print(f"{'=' * 80}")
    print(f"  Total samples:   {total}")
    print(f"  Exact matches:   {correct}")
    print(f"  Accuracy:        {accuracy:.1f}%")
    print(f"{'=' * 80}")

    # Save results
    results_df = pd.DataFrame(results)
    results_path = output_dir / "baseline_predictions.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n[INFO] Predictions saved to {results_path}")

    # Save summary
    summary = {
        "model": config["model"]["name"],
        "num_samples": total,
        "exact_match_accuracy": accuracy,
        "correct": correct,
        "total": total,
        "per_complexity": {},
    }
    for c in sorted(results_df["complexity"].unique()):
        c_df = results_df[results_df["complexity"] == c]
        c_correct = c_df["exact_match"].sum()
        c_total = len(c_df)
        summary["per_complexity"][f"level_{c}"] = {
            "correct": int(c_correct),
            "total": int(c_total),
            "accuracy": round(c_correct / c_total * 100, 1) if c_total > 0 else 0,
        }

    summary_path = output_dir / "baseline_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[INFO] Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
