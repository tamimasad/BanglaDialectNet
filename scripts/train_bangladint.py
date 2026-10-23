"""Final training run for BanglaDialectNet.

Training and validation data are combined, with a fresh stratified 10%
holdout for checkpoint monitoring. Training uses per-example loss averaging and balanced checkpoint selection.

The model predicts the five known dialects. Unknown-dialect detection is handled at inference time; normalization and translation remain separate
tasks.
"""

import json
import os
import sys

sys.path.insert(0, "src")

import numpy as np
import pandas as pd
from datasets import Dataset
from sacrebleu.metrics import CHRF
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

from data_loader.load_processed import load_processed_splits
from data_loader.multitask import ALL_TASKS, build_task_examples

# ----------------------------- Settings ------------------------------------
SMOKE = "--smoke" in sys.argv

MODEL_NAME = "csebuetnlp/banglat5"
OUTPUT_DIR = "models/_smoke_test" if SMOKE else "models/BanglaDINT-T5"
REPORTS_DIR = "reports"
SEED = 42
MAX_INPUT_LEN = 64
MAX_TARGET_LEN = 64
NUM_EPOCHS = 1 if SMOKE else 8
USE_BF16 = True
MONITOR_FRACTION = 0.10

DIALECTS = ["Noakhali", "Barishal", "Chittagong", "Sylhet", "Mymensingh"]


OVERSAMPLE = {}

REQUIRED_COLUMNS = (
    "dialect_bangla",
    "dialect_banglish",
    "standard_bangla",
    "english",
    "region",
)


# ----------------------------- Data checks ---------------------------------
def check_dataframe(df, label):
    df = df.copy()
    df.columns = [str(c).lstrip("\ufeff").strip() for c in df.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{label}: missing columns {missing}. Found: {list(df.columns)}"
        )

    for column in REQUIRED_COLUMNS:
        blank = df[column].isna() | df[column].astype(str).str.strip().eq("")
        if blank.any():
            raise ValueError(f"{label}: {int(blank.sum())} blank values in '{column}'")

    df["region"] = df["region"].astype(str).str.strip()
    unexpected = set(df["region"]) - set(DIALECTS)
    if unexpected:
        raise ValueError(f"{label}: unexpected dialect labels {unexpected}")
    return df


def make_final_train_and_monitor_split(
    train_df, val_df, fraction=MONITOR_FRACTION, seed=SEED
):
    """Create a small, stratified monitor split from train+val."""
    combined = pd.concat([train_df, val_df], ignore_index=True)
    monitor_parts, train_parts = [], []
    for region, group in combined.groupby("region"):
        group = group.sample(frac=1.0, random_state=seed)
        n_monitor = max(1, int(round(len(group) * fraction)))
        monitor_parts.append(group.iloc[:n_monitor])
        train_parts.append(group.iloc[n_monitor:])
    monitor_df = pd.concat(monitor_parts, ignore_index=True)
    train_df_final = pd.concat(train_parts, ignore_index=True)
    return train_df_final, monitor_df


def build_examples(df, label, oversample=None):
    df = check_dataframe(df, label)
    inputs, targets, task_names = build_task_examples(
        df, tasks=ALL_TASKS, oversample=oversample
    )

    multipliers_by_task = oversample or {}
    region_counts = df["region"].value_counts().to_dict()
    for task in ALL_TASKS:
        multipliers = multipliers_by_task.get(task, {})
        expected = sum(
            count * multipliers.get(region, 1)
            for region, count in region_counts.items()
        )
        observed = task_names.count(task)
        if observed != expected:
            raise ValueError(
                f"{label}: task '{task}' has {observed}, expected {expected}"
            )
        print(f"    {task}: {observed}")
    return inputs, targets


def assert_no_truncation(tokenizer, inputs, targets, label):
    longest_input = max(len(ids) for ids in tokenizer(inputs)["input_ids"])
    longest_target = max(
        len(ids) for ids in tokenizer(text_target=targets)["input_ids"]
    )
    print(
        f"  {label}: longest input = {longest_input} tokens, longest target = {longest_target} tokens"
    )
    if longest_input > MAX_INPUT_LEN or longest_target > MAX_TARGET_LEN:
        raise ValueError(
            f"{label}: examples longer than the limits (input {MAX_INPUT_LEN}, "
            f"target {MAX_TARGET_LEN}). Raise MAX_INPUT_LEN / MAX_TARGET_LEN above."
        )


def tokenize_batch(batch, tokenizer):
    model_inputs = tokenizer(batch["input"], max_length=MAX_INPUT_LEN, truncation=True)
    labels = tokenizer(
        text_target=batch["target"], max_length=MAX_TARGET_LEN, truncation=True
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# ----------------------------- Balanced loss -------------------------------
class PerExampleLossTrainer(Seq2SeqTrainer):
    """Averages loss per example, not per token."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs.logits.float()
        token_loss = CrossEntropyLoss(ignore_index=-100, reduction="none")(
            logits.reshape(-1, logits.size(-1)), labels.reshape(-1)
        ).reshape(labels.shape)
        mask = (labels != -100).float()
        per_example = (token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        loss = per_example.mean()
        return (loss, outputs) if return_outputs else loss


# ----------------------------- Balanced metrics ----------------------------
def has_bangla_script(text):
    return any("\u0980" <= ch <= "\u09ff" for ch in text)


def make_compute_metrics(tokenizer):
    chrf = CHRF()

    def compute_metrics(eval_pred):
        predictions = eval_pred.predictions
        labels = eval_pred.label_ids
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        pred_text = [
            t.strip()
            for t in tokenizer.batch_decode(predictions, skip_special_tokens=True)
        ]
        true_text = [
            t.strip() for t in tokenizer.batch_decode(labels, skip_special_tokens=True)
        ]

        metrics = {}
        accuracies = []
        for dialect in DIALECTS:
            rows = [i for i, t in enumerate(true_text) if t == dialect]
            correct = sum(pred_text[i] == dialect for i in rows)
            accuracy = 100.0 * correct / max(len(rows), 1)
            metrics[f"acc_{dialect}"] = float(accuracy)
            accuracies.append(accuracy)
        metrics["dialect_macro_acc"] = float(np.mean(accuracies))
        metrics["dialect_min_acc"] = float(min(accuracies))

        dialect_set = set(DIALECTS)
        groups = {
            "normalize": [
                i
                for i, t in enumerate(true_text)
                if t not in dialect_set and has_bangla_script(t)
            ],
            "translate": [
                i
                for i, t in enumerate(true_text)
                if t not in dialect_set and not has_bangla_script(t)
            ],
        }
        for name, rows in groups.items():
            if rows:
                hyp = [pred_text[i] for i in rows]
                ref = [true_text[i] for i in rows]
                metrics[f"{name}_chrf"] = float(chrf.corpus_score(hyp, [ref]).score)
            else:
                metrics[f"{name}_chrf"] = 0.0

        metrics["balanced_score"] = float(
            np.mean(
                [
                    metrics["dialect_macro_acc"],
                    metrics["normalize_chrf"],
                    metrics["translate_chrf"],
                ]
            )
        )
        return metrics

    return compute_metrics


def print_final_report(metrics):
    print("\n" + "=" * 60)
    print("BEST MODEL -- MONITOR-SLICE RESULTS (test set never touched by this script)")
    print("=" * 60)
    print(f"  balanced score   : {metrics['eval_balanced_score']:.2f}")
    print(f"  normalize chrF   : {metrics['eval_normalize_chrf']:.2f}")
    print(f"  translate chrF   : {metrics['eval_translate_chrf']:.2f}")
    print(
        f"  dialect accuracy : {metrics['eval_dialect_macro_acc']:.1f}% "
        f"(weakest {metrics['eval_dialect_min_acc']:.1f}%)"
    )
    for dialect in DIALECTS:
        print(f"    {dialect:<11}: {metrics[f'eval_acc_{dialect}']:.1f}%")


# ----------------------------- Main ----------------------------------------
def main():
    set_seed(SEED)
    if SMOKE:
        print("SMOKE TEST: tiny subset, 1 epoch. Only checks that the code runs.\n")

    print(f"Loading tokenizer and model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    splits = load_processed_splits()
    train_raw = check_dataframe(splits["train"], "train")
    val_raw = check_dataframe(splits["validation"], "validation")

    print(
        f"Combining train ({len(train_raw)}) + validation ({len(val_raw)}), "
        f"carving {MONITOR_FRACTION:.0%} for monitoring..."
    )
    train_final_df, monitor_df = make_final_train_and_monitor_split(train_raw, val_raw)
    print(f"  Final training rows : {len(train_final_df)}")
    print(f"  Monitor rows        : {len(monitor_df)}")

    if SMOKE:
        train_final_df = train_final_df.sample(200, random_state=SEED)
        monitor_df = monitor_df.sample(min(100, len(monitor_df)), random_state=SEED)

    print("Building multi-task examples (count per task):")
    print("  train:")
    train_inputs, train_targets = build_examples(train_final_df, "train", OVERSAMPLE)
    print("  monitor:")
    monitor_inputs, monitor_targets = build_examples(monitor_df, "monitor")
    print(f"  Train examples: {len(train_inputs)}")
    print(f"  Monitor examples: {len(monitor_inputs)}")

    print("Checking that nothing gets truncated...")
    assert_no_truncation(tokenizer, train_inputs, train_targets, "train")
    assert_no_truncation(tokenizer, monitor_inputs, monitor_targets, "monitor")

    print("Tokenizing...")
    train_ds = Dataset.from_dict({"input": train_inputs, "target": train_targets}).map(
        lambda b: tokenize_batch(b, tokenizer),
        batched=True,
        remove_columns=["input", "target"],
    )
    monitor_ds = Dataset.from_dict(
        {"input": monitor_inputs, "target": monitor_targets}
    ).map(
        lambda b: tokenize_batch(b, tokenizer),
        batched=True,
        remove_columns=["input", "target"],
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=2,
        learning_rate=3e-4,
        num_train_epochs=NUM_EPOCHS,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="balanced_score",
        greater_is_better=True,
        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LEN,
        generation_num_beams=1,  # greedy during training for speed; real scoring uses beam search separately
        bf16=USE_BF16,
        seed=SEED,
        logging_steps=50,
        report_to="tensorboard",
    )

    trainer = PerExampleLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=monitor_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        compute_metrics=make_compute_metrics(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print("\nStarting training.")
    print("Optional: in a second terminal run `nvidia-smi -l 2` to watch the GPU.\n")
    trainer.train()

    print(f"\nSaving best model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    final_metrics = trainer.evaluate()
    print_final_report(final_metrics)
    if not SMOKE:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        path = os.path.join(REPORTS_DIR, "BanglaDINT_production_monitor_metrics.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(final_metrics, f, indent=2)
        print(f"\nSaved: {path}")
        print(f"View training curves with: tensorboard --logdir {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
