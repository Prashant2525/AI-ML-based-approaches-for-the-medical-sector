"""
BiomedGPT: Lightweight generalist biomedical vision-language model.

A biomedical LLM (LLaMA-based) designed for diverse biomedical tasks.
We use it as a text-based medical QA model, providing the question in
text form to test biomedical knowledge without visual grounding.

Reference: "BiomedGPT: A Unified and Generalist Biomedical Generative
Pre-trained Transformer for Vision, Language, and Multimodal Tasks" (2023/2024)

Usage:
    from src.biomedgpt import load_model, run_inference, run_evaluation
"""

import os
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "BiomedGPT"
# Verified HuggingFace model IDs (checked via HF API)
# PharMolix/BioMedGPT-LM-7B: 491 downloads, LLaMA-based, text-generation
HF_MODEL_ID = "PharMolix/BioMedGPT-LM-7B"
FALLBACK_MODEL_IDS = []


def load_model(device="cuda", model_id=None):
    """
    Load BiomedGPT model and tokenizer.

    This is a text-only LLaMA-based biomedical LM. Since it lacks a visual
    encoder, it answers questions based on biomedical text knowledge only.

    Args:
        device: 'cuda' or 'cpu'
        model_id: Override default HuggingFace model ID

    Returns:
        tuple: (model, tokenizer)
    """
    if model_id is None:
        model_id = HF_MODEL_ID

    print(f"[INFO] Loading BiomedGPT model: {model_id}")
    print(f"[INFO] Device: {device}")

    model_ids_to_try = [model_id] + FALLBACK_MODEL_IDS

    for mid in model_ids_to_try:
        try:
            print(f"[INFO] Attempting to load from: {mid}")

            tokenizer = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)

            # FIX: Set pad_token to eos_token if not set
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                print("[INFO] Set pad_token = eos_token")

            model = AutoModelForCausalLM.from_pretrained(
                mid,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                trust_remote_code=True,
                device_map="auto" if device == "cuda" else None,
            )

            if device == "cuda" and not hasattr(model, "hf_device_map"):
                model = model.to(device)

            model.eval()
            print(f"[INFO] BiomedGPT loaded successfully from: {mid}")
            return model, tokenizer

        except Exception as e:
            print(f"[WARN] Failed to load from {mid}: {e}")
            continue

    raise RuntimeError(
        f"[ERROR] Could not load BiomedGPT from any source. "
        f"Tried: {model_ids_to_try}. "
        f"Install: pip install transformers>=4.36.0"
    )


def run_inference(model, processor, image, question, device="cuda", max_new_tokens=64):
    """
    Run zero-shot VQA inference on a single image-question pair.

    BiomedGPT is text-only, so the image is not used directly.
    The model answers based on biomedical text knowledge.

    Args:
        model: Loaded BiomedGPT model
        processor: Model's tokenizer
        image: PIL Image (RGB) — not used by this text-only model
        question: Question string
        device: Device string
        max_new_tokens: Maximum tokens to generate

    Returns:
        dict: {"prediction": str}
    """
    prompt = (
        f"Based on the medical image, answer the following question. "
        f"Question: {question} Answer:"
    )

    # Tokenize — no padding needed for single-sample inference
    inputs = processor(prompt, return_tensors="pt", truncation=True, max_length=512)

    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=3,
        )

    # Strip prompt tokens from output
    input_len = inputs["input_ids"].shape[-1]
    generated_ids = outputs[0][input_len:]
    prediction = processor.decode(generated_ids, skip_special_tokens=True).strip()

    return {"prediction": prediction}


def run_evaluation(data_dir, image_dir, output_dir, num_samples=20, device="cuda"):
    """
    Full evaluation pipeline for BiomedGPT on Kvasir-VQA-x1.
    """
    from src.eval_utils import (
        select_diverse_samples, process_single_result,
        print_result, print_summary, save_results,
    )

    test_csv = os.path.join(data_dir, "kvasir_vqa_x1_test.csv")
    test_df = pd.read_csv(test_csv)
    sample_df = select_diverse_samples(test_df, num_samples)

    model, processor = load_model(device)

    results = []
    print(f"\n[INFO] Running BiomedGPT inference on {len(sample_df)} samples...\n")
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

    summary = save_results(results, "biomedgpt", output_dir)
    print_summary(MODEL_NAME, summary)
    return summary


if __name__ == "__main__":
    run_evaluation(
        data_dir="./data",
        image_dir="./data/images",
        output_dir="./results/predictions",
    )

