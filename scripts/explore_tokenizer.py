"""
Tokenizer exploration for the banglat5 baseline (Phase 2).
Measures REAL subword token lengths for 3 multi-task formats
(normalize, translate-to-english, identify-dialect) using banglat5's
own tokenizer -- word/character counts from EDA don't map directly
to subword token counts, need this to size max_length correctly.

"""

import sys

sys.path.insert(0, "src")

import numpy as np
from transformers import AutoTokenizer
from normalizer import normalize

from data.load_processed import load_processed_splits

MODEL_NAME = "csebuetnlp/banglat5"


def build_task_examples(df):
    """Each row into 3 (input, target) pairs: normalize, translate, identify."""
    inputs, targets = [], []
    for _, row in df.iterrows():
        dialect_text = normalize(str(row["dialect_bangla"]))
        standard_text = normalize(str(row["standard_bangla"]))
        english_text = str(row["english"])
        region = str(row["region"])

        inputs.append(f"normalize: {dialect_text}")
        targets.append(standard_text)

        inputs.append(f"translate to english: {dialect_text}")
        targets.append(english_text)

        inputs.append(f"identify dialect: {dialect_text}")
        targets.append(region)

    return inputs, targets


def token_length_stats(tokenizer, texts, label):
    encoded = tokenizer(texts)
    lengths = np.array([len(ids) for ids in encoded["input_ids"]])
    print(f"\n{label} (n={len(lengths)}):")
    print(f"  mean={lengths.mean():.1f}  median={np.median(lengths):.0f}")
    print(
        f"  p90={np.percentile(lengths, 90):.0f}  "
        f"p95={np.percentile(lengths, 95):.0f}  "
        f"p99={np.percentile(lengths, 99):.0f}  max={lengths.max()}"
    )
    return lengths


def recommend_max_length(lengths, round_to=(32, 64, 128, 256, 512)):
    """Smallest bucket from round_to that covers the 99th percentile."""
    p99 = np.percentile(lengths, 99)
    for candidate in round_to:
        if candidate >= p99:
            return candidate
    return round_to[-1]


def main():
    print(
        f"Loading tokenizer: {MODEL_NAME} (use_fast=False -- required, see model card)"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

    train_df = load_processed_splits()["train"]
    print(
        f"\nBuilding multi-task examples from {len(train_df)} rows "
        f"-> {len(train_df) * 3} task examples..."
    )
    inputs, targets = build_task_examples(train_df)

    input_lengths = token_length_stats(
        tokenizer, inputs, "INPUT token lengths (all 3 tasks)"
    )
    target_lengths = token_length_stats(
        tokenizer, targets, "TARGET token lengths (all 3 tasks)"
    )

    max_input_len = recommend_max_length(input_lengths)
    max_target_len = recommend_max_length(target_lengths)
    print(f"\nRecommended max_length (covers 99th percentile):")
    print(f"  Input:  {max_input_len}")
    print(f"  Target: {max_target_len}")

    print("\nSample examples after normalization (visual sanity check):")
    for i in range(0, 6, 2):
        print(f"  [{inputs[i][:50]}] -> [{targets[i][:50]}]")


if __name__ == "__main__":
    main()
