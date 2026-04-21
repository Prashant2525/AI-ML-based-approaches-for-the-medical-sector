"""
MiniGPT-4: Large-scale vision-language model for enhanced multimodal understanding.

A VLM using Vicuna/LLaMA as its LLM backbone (~7B params) with a BLIP-2
visual encoder. Though larger than other "small" models, it provides a
useful comparison point for understanding the impact of model scale.

Reference: "MiniGPT-4: Enhancing Vision-Language Understanding with
Advanced Large Language Models" (2023)

Usage:
    from src.minigpt4 import load_model, run_inference, run_evaluation
"""

import os
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoProcessor,
    Blip2Processor,
    Blip2ForConditionalGeneration,
    InstructBlipProcessor,
    InstructBlipForConditionalGeneration,
)

MODEL_NAME = "MiniGPT-4"
# MiniGPT-4 is not directly on HuggingFace as a single model card.
# It requires Vicuna weights + MiniGPT-4 projection layer.
# We use InstructBLIP (from same lab, similar architecture, readily available)
# as a practical stand-in that captures the same architectural paradigm.
HF_MODEL_ID = "Salesforce/instructblip-vicuna-7b"
FALLBACK_MODEL_IDS = [
    "Salesforce/instructblip-flan-t5-xl",
    "Vision-CAIR/MiniGPT-4",
]


def load_model(device="cuda", model_id=None):
    """
    Load MiniGPT-4 (or compatible InstructBLIP-Vicuna) model.

    MiniGPT-4 uses the same BLIP-2 visual encoder + Vicuna LLM architecture.
    InstructBLIP-Vicuna is architecturally equivalent and readily available
    on HuggingFace, making it the preferred loading path.

    Args:
        device: 'cuda' or 'cpu'
        model_id: Override default HuggingFace model ID

    Returns:
        tuple: (model, processor)
    """
    if model_id is None:
        model_id = HF_MODEL_ID

    print(f"[INFO] Loading MiniGPT-4 model: {model_id}")
    print(f"[INFO] Device: {device}")

    model_ids_to_try = [model_id] + FALLBACK_MODEL_IDS

    for mid in model_ids_to_try:
        try:
            print(f"[INFO] Attempting to load from: {mid}")

            # Try InstructBLIP first (most compatible)
            if "instructblip" in mid.lower():
                processor = InstructBlipProcessor.from_pretrained(mid)
                model = InstructBlipForConditionalGeneration.from_pretrained(
                    mid,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    device_map="auto" if device == "cuda" else None,
                )
            # Try BLIP-2 style loading
            elif "blip2" in mid.lower():
                processor = Blip2Processor.from_pretrained(mid)
                model = Blip2ForConditionalGeneration.from_pretrained(
                    mid,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    device_map="auto" if device == "cuda" else None,
                )
            else:
                # Generic loading for MiniGPT-4 or other models
                try:
                    processor = AutoProcessor.from_pretrained(
                        mid, trust_remote_code=True
                    )
                except Exception:
                    processor = AutoTokenizer.from_pretrained(
                        mid, trust_remote_code=True
                    )

                model = AutoModelForCausalLM.from_pretrained(
                    mid,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    trust_remote_code=True,
                    device_map="auto" if device == "cuda" else None,
                )

            if device == "cuda" and not hasattr(model, "hf_device_map"):
                model = model.to(device)

            model.eval()
            print(f"[INFO] MiniGPT-4 loaded successfully from: {mid}")
            return model, processor

        except Exception as e:
            print(f"[WARN] Failed to load from {mid}: {e}")
            continue

    raise RuntimeError(
        f"[ERROR] Could not load MiniGPT-4 from any source. "
        f"Tried: {model_ids_to_try}. "
        f"See: https://github.com/Vision-CAIR/MiniGPT-4"
    )


def run_inference(model, processor, image, question, device="cuda", max_new_tokens=64):
    """
    Run zero-shot VQA inference on a single image-question pair.

    Handles both InstructBLIP and generic model architectures.

    Args:
        model: Loaded model
        processor: Model's processor
        image: PIL Image (RGB)
        question: Question string
        device: Device string
        max_new_tokens: Maximum tokens to generate

    Returns:
        dict: {"prediction": str}
    """
    prompt = f"Question: {question} Answer:"

    # Prepare inputs
    dtype = torch.float16 if device == "cuda" else torch.float32

    if isinstance(processor, (InstructBlipProcessor, Blip2Processor)):
        inputs = processor(
            images=image,
            text=prompt,
            return_tensors="pt",
        ).to(device, dtype)
    else:
        try:
            inputs = processor(
                images=image,
                text=prompt,
                return_tensors="pt",
            )
            inputs = {k: v.to(device, dtype) if v.dtype in [torch.float32, torch.float16]
                      else v.to(device) for k, v in inputs.items()}
        except TypeError:
            inputs = processor(prompt, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=3,
        )

    # Decode — handle different model architectures
    if isinstance(model, (InstructBlipForConditionalGeneration,
                          Blip2ForConditionalGeneration)):
        # These models return only generated tokens
        prediction = processor.decode(outputs[0], skip_special_tokens=True).strip()
    else:
        # Causal LM — strip prompt tokens
        input_len = inputs.get("input_ids", torch.tensor([[]])).shape[-1]
        generated_ids = outputs[0][input_len:]
        prediction = processor.decode(generated_ids, skip_special_tokens=True).strip()

    return {"prediction": prediction}


def run_evaluation(data_dir, image_dir, output_dir, num_samples=20, device="cuda"):
    """
    Full evaluation pipeline for MiniGPT-4 on Kvasir-VQA-x1.

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
    print(f"\n[INFO] Running MiniGPT-4 inference on {len(sample_df)} samples...\n")
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
    summary = save_results(results, "minigpt4", output_dir)
    print_summary(MODEL_NAME, summary)
    return summary


if __name__ == "__main__":
    run_evaluation(
        data_dir="./data",
        image_dir="./data/images",
        output_dir="./results/predictions",
    )
