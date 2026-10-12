"""
Fine-tune banglat5 as the Phase 2 baseline for BanglaDialectNet.

Trains ONE banglat5 model jointly on all 3 tasks (normalize,
translate-to-english, identify-dialect), using both Bangla-script and
romanized Banglish dialect inputs (6 task-input pairs per row).

"""

import sys

sys.path.insert(0, "src")

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
)

from data_loader.load_processed import load_processed_splits
from data_loader.multitask import ALL_TASKS, build_task_examples

MODEL_NAME = "csebuetnlp/banglat5"
OUTPUT_DIR = "models/banglat5-baseline"
MAX_INPUT_LEN = 64
MAX_TARGET_LEN = 64


def prepare_multitask_dataframe(df, split_name):

    df = df.copy()
    df.columns = [str(column).lstrip("\ufeff").strip() for column in df.columns]
    if not df.columns.is_unique:
        raise ValueError(f"{split_name}: duplicate column names after stripping spaces")

    raw_to_processed = {
        "standard_bangla": "bangla_speech",
        "english": "english_speech",
        "region": "region_name",
    }
    for processed_name, raw_name in raw_to_processed.items():
        if processed_name not in df.columns and raw_name in df.columns:
            df[processed_name] = df[raw_name]

    for processed_name, suffix in (
        ("dialect_bangla", "_bangla_speech"),
        ("dialect_banglish", "_banglish_speech"),
    ):
        if processed_name in df.columns:
            continue
        regional_columns = [
            column for column in df.columns if column.lower().endswith(suffix)
        ]
        if len(regional_columns) == 1:
            df[processed_name] = df[regional_columns[0]]
        elif len(regional_columns) > 1 and "region" in df.columns:

            region_keys = (
                df["region"]
                .astype(str)
                .str.strip()
                .str.lower()
                .str.replace(r"[\s-]+", "_", regex=True)
            )
            df[processed_name] = None
            for column in regional_columns:
                column_region = column[: -len(suffix)].lower()
                matching_rows = region_keys == column_region
                df.loc[matching_rows, processed_name] = df.loc[matching_rows, column]

    required_columns = (
        "dialect_bangla",
        "dialect_banglish",
        "standard_bangla",
        "english",
        "region",
    )
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(
            f"{split_name}: missing required columns {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    for column in required_columns:
        blank = df[column].isna() | df[column].astype(str).str.strip().eq("")
        if blank.any():
            raise ValueError(
                f"{split_name}: {int(blank.sum())} rows have missing/blank "
                f"'{column}' values (for example, index {df.index[blank][0]})."
            )

    df["region"] = df["region"].astype(str).str.strip()
    return df


def build_verified_task_examples(df, split_name):
    """Explicitly expand and check all six Bangla/Banglish training tasks."""
    df = prepare_multitask_dataframe(df, split_name)
    inputs, targets, task_names = build_task_examples(df, tasks=ALL_TASKS)
    expected_per_task = len(df)
    for task in ALL_TASKS:
        observed = task_names.count(task)
        if observed != expected_per_task:
            raise ValueError(
                f"{split_name}: task '{task}' has {observed} examples; "
                f"expected {expected_per_task}."
            )
        print(f"    {task}: {observed}")
    if not inputs:
        raise ValueError(f"{split_name}: no examples produced")
    return inputs, targets


def to_hf_dataset(inputs, targets):
    return Dataset.from_dict({"input": inputs, "target": targets})


def tokenize_batch(batch, tokenizer):
    model_inputs = tokenizer(batch["input"], max_length=MAX_INPUT_LEN, truncation=True)
    labels = tokenizer(
        text_target=batch["target"], max_length=MAX_TARGET_LEN, truncation=True
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def main():
    print(f"Loading tokenizer and model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    print("Loading processed dataset and building multi-task examples...")
    splits = load_processed_splits()
    train_inputs, train_targets = build_verified_task_examples(splits["train"], "train")
    val_inputs, val_targets = build_verified_task_examples(
        splits["validation"], "validation"
    )
    print(f"  Train examples: {len(train_inputs)}")
    print(f"  Validation examples: {len(val_inputs)}")

    train_ds = to_hf_dataset(train_inputs, train_targets)
    val_ds = to_hf_dataset(val_inputs, val_targets)

    print("Tokenizing...")
    train_ds = train_ds.map(
        lambda b: tokenize_batch(b, tokenizer),
        batched=True,
        remove_columns=["input", "target"],
    )
    val_ds = val_ds.map(
        lambda b: tokenize_batch(b, tokenizer),
        batched=True,
        remove_columns=["input", "target"],
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        learning_rate=3e-4,
        num_train_epochs=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,  # efficient disk usage -- only 2 checkpoints
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=False,
        logging_steps=50,
        report_to="tensorboard",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("\nStarting training.")
    print(
        "Optional: open a second terminal and run `nvidia-smi -l 2` to watch GPU usage.\n"
    )
    trainer.train()

    print(f"\nSaving best model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("Done. View training curves with: tensorboard --logdir logs")


if __name__ == "__main__":
    main()
