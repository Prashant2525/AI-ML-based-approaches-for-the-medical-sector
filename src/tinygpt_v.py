"""
TinyGPT-V / Florence-2: Small general-purpose vision-language model.

Since TinyGPT-V (Tyrannosaurus/TinyGPT-V) requires a custom repo setup
that does not work with standard HuggingFace `from_pretrained` loading,
we use Microsoft Florence-2-base (0.23B params) as a drop-in replacement.

Florence-2 is a lightweight, general-purpose VLM that supports VQA tasks
out of the box and loads seamlessly via HuggingFace Transformers.

Reference: "Florence-2: Advancing a Unified Representation for a Variety
of Vision Tasks" (Microsoft, 2024)

Usage:
    from src.tinygpt_v import load_model, run_inference, run_evaluation
"""

import os
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForCausalLM

MODEL_NAME = "Florence-2 (TinyGPT-V replacement)"
# Verified HuggingFace model IDs
HF_MODEL_ID = "microsoft/Florence-2-base"
FALLBACK_MODEL_IDS = [
    "microsoft/Florence-2-large",
]


def load_model(device="cuda", model_id=None):
    """
    Load Florence-2 model and processor.

    Florence-2 is a lightweight (0.23B) multimodal model that supports
    VQA, captioning, and other vision-language tasks.

    Args:
        device: 'cuda' or 'cpu'
        model_id: Override default HuggingFace model ID

    Returns:
        tuple: (model, processor)
    """
    if model_id is None:
        model_id = HF_MODEL_ID

    print(f"[INFO] Loading Florence-2 model: {model_id}")
    print(f"[INFO] Device: {device}")

    model_ids_to_try = [model_id] + FALLBACK_MODEL_IDS

    for mid in model_ids_to_try:
        try:
            print(f"[INFO] Attempting to load from: {mid}")

            processor = AutoProcessor.from_pretrained(mid, trust_remote_code=True)

            model = AutoModelForCausalLM.from_pretrained(
                mid,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                trust_remote_code=True,
            )

            if device == "cuda":
                model = model.to(device)

            model.eval()
            print(f"[INFO] Florence-2 loaded successfully from: {mid}")
            return model, processor

        except Exception as e:
            print(f"[WARN] Failed to load from {mid}: {e}")
            continue

    raise RuntimeError(
        f"[ERROR] Could not load Florence-2 from any source. "
        f"Tried: {model_ids_to_try}."
    )


def run_inference(model, processor, image, question, device="cuda", max_new_tokens=64):
    """
    Run zero-shot VQA inference on a single image-question pair.

    Florence-2 uses a task prefix format: <VQA> for visual question answering.

    Args:
        model: Loaded Florence-2 model
        processor: Model's processor
        image: PIL Image (RGB)
        question: Question string
        device: Device string
        max_new_tokens: Maximum tokens to generate

    Returns:
        dict: {"prediction": str}
    """
    # Florence-2 uses task-prefix prompting
    prompt = f"<VQA> {question}"

    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt",
    )

    # Move to device
    dtype = torch.float16 if device == "cuda" else torch.float32
    inputs = {k: v.to(device, dtype) if v.is_floating_point()
              else v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=3,
        )

    prediction = processor.decode(outputs[0], skip_special_tokens=True).strip()
    # Florence-2 may include the task prefix in output — remove it
    if prediction.startswith("<VQA>"):
        prediction = prediction[5:].strip()

    return {"prediction": prediction}


def run_evaluation(data_dir, image_dir, output_dir, num_samples=20, device="cuda"):
    """
    Full evaluation pipeline for Florence-2 on Kvasir-VQA-x1.
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
    print(f"\n[INFO] Running Florence-2 inference on {len(sample_df)} samples...\n")
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

    summary = save_results(results, "florence2", output_dir)
    print_summary(MODEL_NAME, summary)
    return summary


if __name__ == "__main__":
    run_evaluation(
        data_dir="./data",
        image_dir="./data/images",
        output_dir="./results/predictions",
    )
