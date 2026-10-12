"""
Tokenizer exploration for the banglat5 baseline (Phase 2).
Measures REAL subword token lengths for 3 multi-task formats
(normalize, normalize-banglish, translate-to-english, identify-dialect) using banglat5's
own tokenizer -- word/character counts from EDA don't map directly
to subword token counts, need this to size max_length correctly.

"""

import sys

sys.path.insert(0, "src")

import numpy as np
from transformers import AutoTokenizer

from data_loader.load_processed import load_processed_splits
from data_loader.multitask import (
    build_task_examples,
    TASK_NORMALIZE,
    TASK_NORMALIZE_BANGLISH,
    TASK_TRANSLATE,
    TASK_TRANSLATE_BANGLISH,
    TASK_IDENTIFY,
    TASK_IDENTIFY_BANGLISH,
)

MODEL_NAME = "csebuetnlp/banglat5"


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


def recommend_max_length(true_max, round_to=(32, 64, 128, 256, 512)):
    for candidate in round_to:
        if candidate >= true_max:
            return candidate
    return round_to[-1]


def main():
    print(f"Loading tokenizer: {MODEL_NAME} (use_fast=False)")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

    train_df = load_processed_splits()["train"]
    inputs, targets, task_names = build_task_examples(train_df)
    print(f"\nTotal task examples: {len(inputs)} (from {len(train_df)} rows x 6 tasks)")

    bangla_tasks = {TASK_NORMALIZE, TASK_TRANSLATE, TASK_IDENTIFY}
    banglish_tasks = {
        TASK_NORMALIZE_BANGLISH,
        TASK_TRANSLATE_BANGLISH,
        TASK_IDENTIFY_BANGLISH,
    }

    bangla_inputs = [inp for inp, t in zip(inputs, task_names) if t in bangla_tasks]
    banglish_inputs = [inp for inp, t in zip(inputs, task_names) if t in banglish_tasks]

    bangla_lengths = token_length_stats(
        tokenizer, bangla_inputs, "INPUT lengths -- Bangla-script tasks"
    )
    banglish_lengths = token_length_stats(
        tokenizer, banglish_inputs, "INPUT lengths -- Banglish tasks"
    )
    target_lengths = token_length_stats(
        tokenizer, targets, "TARGET lengths (all tasks)"
    )

    overall_input_max = max(bangla_lengths.max(), banglish_lengths.max())
    print(f"\nRecommended max_length (covers true max observed, not just p99):")
    print(
        f"  Input:  {recommend_max_length(overall_input_max)}  "
        f"(Bangla max={bangla_lengths.max()}, Banglish max={banglish_lengths.max()})"
    )
    print(f"  Target: {recommend_max_length(target_lengths.max())}")


if __name__ == "__main__":
    main()
