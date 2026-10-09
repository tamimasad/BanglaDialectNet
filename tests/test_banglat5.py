"""
Run all 3 BanglaDialectNet tasks using the trained BanglaT5 model.

Usage:
python scripts/infer_banglat5.py --text "কই যাইবার চাও?"

"""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Paths

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "banglat5-baseline"


TASK_PREFIXES = {
    "Normalization": "normalize:",
    "English Translation": "translate to english:",
    "Dialect Identification": "identify dialect:",
}


def load_model():
    """Load our fine-tuned BanglaT5 model."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model directory not found:\n{MODEL_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading trained model: {MODEL_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        use_fast=False,
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

    model.to(device)
    model.eval()

    print(f"Device: {device}")

    return tokenizer, model, device


def generate(model_input, tokenizer, model, device):
    """Generate one prediction."""

    encoded = tokenizer(
        model_input,
        return_tensors="pt",
        max_length=64,
        truncation=True,
    )

    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.no_grad():
        generated_ids = model.generate(
            **encoded,
            max_new_tokens=64,
            num_beams=4,
            do_sample=False,
        )

    return tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run all BanglaDialectNet BanglaT5 tasks."
    )

    parser.add_argument(
        "--text",
        required=True,
        help="Dialect Bangla input text.",
    )

    args = parser.parse_args()

    tokenizer, model, device = load_model()

    print("\n" + "=" * 60)
    print("BanglaDialectNet — BanglaT5 Multi-task Inference")
    print("=" * 60)

    print(f"\nOriginal input:\n{args.text}")

    # Run the same input through all three learned tasks.
    for task_name, prefix in TASK_PREFIXES.items():

        model_input = f"{prefix} {args.text}"

        output = generate(
            model_input,
            tokenizer,
            model,
            device,
        )

        print("\n" + "-" * 60)
        print(f"Task: {task_name}")
        print(f"Model input: {model_input}")
        print(f"Output: {output}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
