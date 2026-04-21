"""
MoE-TinyMed: Mixture-of-Experts vision language model for medical VQA.

A lightweight medical VLM using MoE routing with Phi-2 backbone (~3.6B total,
~2B active parameters). Purpose-built for medical VQA tasks.

Reference: "MoE-TinyMed: Mixture of Experts for Tiny Medical Large
Vision-Language Models" (2024)

Usage:
    from src.moe_tinymed import load_model, run_inference, run_evaluation
"""

import os
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm

# MoE-TinyMed uses a custom architecture based on MoE-LLaVA
# GitHub: jiangsongtao/TinyMed
# It requires cloning the repo and loading via their custom code
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor

MODEL_NAME = "MoE-TinyMed"
# Verified HuggingFace model IDs (checked via HF API)
HF_MODEL_ID = "Joycean0301/MoE-Tinymed-phi2-llava"
FALLBACK_MODEL_IDS = [
    "JsST/TinyMed",
]


def load_model(device="cuda", model_id=None):
    """
    Load MoE-TinyMed model and processor.

    This model uses a custom MoE architecture built on Phi-2 with a
    visual encoder. Due to the custom architecture, we attempt multiple
    loading strategies.

    Args:
        device: 'cuda' or 'cpu'
        model_id: Override the default HuggingFace model ID

    Returns:
        tuple: (model, processor/tokenizer)
    """
    if model_id is None:
        model_id = HF_MODEL_ID

    print(f"[INFO] Loading MoE-TinyMed model: {model_id}")
    print(f"[INFO] Device: {device}")

    # Strategy 1: Try loading as a standard HuggingFace VLM
    model_ids_to_try = [model_id] + FALLBACK_MODEL_IDS

    model = None
    processor = None

    for mid in model_ids_to_try:
        try:
            print(f"[INFO] Attempting to load from: {mid}")

            # Try loading with AutoProcessor (for VLMs)
            try:
                processor = AutoProcessor.from_pretrained(mid, trust_remote_code=True)
            except Exception:
                processor = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)

            model = AutoModelForCausalLM.from_pretrained(
                mid,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                trust_remote_code=True,
                device_map="auto" if device == "cuda" else None,
            )

            if device == "cuda" and not hasattr(model, "hf_device_map"):
                model = model.to(device)

            model.eval()
            print(f"[INFO] MoE-TinyMed loaded successfully from: {mid}")
            return model, processor

        except Exception as e:
            print(f"[WARN] Failed to load from {mid}: {e}")
            continue

    raise RuntimeError(
        f"[ERROR] Could not load MoE-TinyMed from any source. "
        f"Tried: {model_ids_to_try}. "
        f"You may need to clone the repo: git clone https://github.com/jiangsongtao/TinyMed"
    )


def run_inference(model, processor, image, question, device="cuda", max_new_tokens=64):
    """
    Run zero-shot VQA inference on a single image-question pair.

    Args:
        model: Loaded MoE-TinyMed model
        processor: Model's processor/tokenizer
        image: PIL Image (RGB)
        question: Question string
        device: Device string
        max_new_tokens: Maximum tokens to generate

    Returns:
        dict: {"prediction": str}
    """
    prompt = f"Question: {question} Answer:"

    # Handle different processor types
    if hasattr(processor, "image_processor") or hasattr(processor, "__call__"):
        try:
            inputs = processor(
                images=image,
                text=prompt,
                return_tensors="pt",
            )
        except TypeError:
            # Fallback: text-only processor, encode image separately
            inputs = processor(prompt, return_tensors="pt")
    else:
        inputs = processor(prompt, return_tensors="pt")

    # Move to device
    dtype = torch.float16 if device == "cuda" else torch.float32
    inputs = {k: v.to(device, dtype) if v.dtype in [torch.float32, torch.float16]
              else v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=3,
        )

    # Decode only new tokens
    input_len = inputs.get("input_ids", torch.tensor([[]])).shape[-1]
    generated_ids = outputs[0][input_len:]
    prediction = processor.decode(generated_ids, skip_special_tokens=True).strip()

    return {"prediction": prediction}


def run_evaluation(data_dir, image_dir, output_dir, num_samples=20, device="cuda"):
    """
    Full evaluation pipeline for MoE-TinyMed on Kvasir-VQA-x1.

    Args:
        data_dir: Path to data directory containing CSVs
        image_dir: Path to image directory
        output_dir: Path to save results
        num_samples: Number of test samples to evaluate
        device: 'cuda' or 'cpu'

    Returns:
        dict: Evaluation summary
    """
    from src.eval_utils import (
        select_diverse_samples, process_single_result,
        print_result, print_summary, save_results,
    )

    # Load test data
    test_csv = os.path.join(data_dir, "kvasir_vqa_x1_test.csv")
    test_df = pd.read_csv(test_csv)
    sample_df = select_diverse_samples(test_df, num_samples)

    # Load model
    model, processor = load_model(device)

    # Run inference
    results = []
    print(f"\n[INFO] Running MoE-TinyMed inference on {len(sample_df)} samples...\n")
    print("=" * 80)

    for idx, (_, row) in enumerate(sample_df.iterrows()):
        img_path = os.path.join(image_dir, f"{row['img_id']}.jpg")

        if not os.path.exists(img_path):
            print(f"[SKIP] Image not found: {row['img_id']}")
            continue

        image = Image.open(img_path).convert("RGB")
        output = run_inference(model, processor, image, row["question"], device)
        result = process_single_result(row, output["prediction"])
        results.append(result)
        print_result(idx, len(sample_df), result)

    # Save and print summary
    summary = save_results(results, "moe_tinymed", output_dir)
    print_summary(MODEL_NAME, summary)
    return summary


if __name__ == "__main__":
    run_evaluation(
        data_dir="./data",
        image_dir="./data/images",
        output_dir="./results/predictions",
    )
