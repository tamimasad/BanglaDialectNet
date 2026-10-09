"""
Fine-tune banglat5 as the Phase 2 baseline for BanglaDialectNet.

Trains ONE banglat5 model jointly on all 3 tasks (normalize,
translate-to-english, identify-dialect).

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
from data_loader.multitask import build_task_examples

MODEL_NAME = "csebuetnlp/banglat5"
OUTPUT_DIR = "models/banglat5-baseline"
MAX_INPUT_LEN = 64
MAX_TARGET_LEN = 64


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
    train_inputs, train_targets, _ = build_task_examples(splits["train"])
    val_inputs, val_targets, _ = build_task_examples(splits["validation"])
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
        gradient_accumulation_steps=2,  # effective batch size 32
        learning_rate=3e-4,
        num_train_epochs=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,  # cap disk usage -- keep only 2 checkpoints
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,  # mixed precision -- fits comfortably on 12GB
        logging_dir="logs",
        logging_steps=50,
        report_to="tensorboard",  # free, local, no account needed
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
