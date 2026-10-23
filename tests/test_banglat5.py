"""
Run all 3 BanglaDialectNet tasks using the published BanglaDINT-T5 model.

By default this pulls the model straight from the Hugging Face Hub
(tamimaungan/BanglaDINT-T5)
Usage:
    python tests/test_banglat5.py --text "কই যাইবার চাও?"
Or:
    python tests/test_banglat5.py --text "koi jaibar chao?"
Or, against a local checkpoint:
    python tests/test_banglat5.py --text "..." --model models/BanglaDINT-T5-production
"""

import argparse

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

DEFAULT_MODEL_ID = "tamimaungan/BanglaDINT-T5"

TASK_PREFIXES = {
    "Normalization": "normalize:",
    "English Translation": "translate to english:",
    "Dialect Identification": "identify dialect:",
}


def load_model(model_id):
    """Load the tokenizer + model, from the Hub by default or a local path."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
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

    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(
        description="Run all BanglaDialectNet BanglaT5 tasks."
    )
    parser.add_argument("--text", required=True, help="Dialect Bangla input text.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_ID,
        help=f"Hub model ID or local checkpoint path (default: {DEFAULT_MODEL_ID})",
    )
    args = parser.parse_args()

    tokenizer, model, device = load_model(args.model)

    print("\n" + "=" * 60)
    print("BanglaDialectNet — BanglaT5 Multi-task Inference")
    print("=" * 60)
    print(f"\nOriginal input:\n{args.text}")

    # Run the same input through all three learned tasks.
    for task_name, prefix in TASK_PREFIXES.items():
        model_input = f"{prefix} {args.text}"
        output = generate(model_input, tokenizer, model, device)

        print("\n" + "-" * 60)
        print(f"Task: {task_name}")
        print(f"Model input: {model_input}")
        print(f"Output: {output}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
