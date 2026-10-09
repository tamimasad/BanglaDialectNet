"""
Fine-tune BanglaT5 as the Phase 2 baseline for BanglaDialectNet.

Trains ONE BanglaT5 model jointly on all 3 tasks:
1. normalize
2. translate-to-english
3. identify-dialect

"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

# Make imports work reliably when this script is launched from the project root
# with: python scripts/train_banglat5.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from data_loader.load_processed import load_processed_splits
from data_loader.multitask import build_task_examples


MODEL_NAME = "csebuetnlp/banglat5"
OUTPUT_DIR = PROJECT_ROOT / "models" / "banglat5-baseline"
LOGGING_DIR = OUTPUT_DIR / "logs"

MAX_INPUT_LEN = 64
MAX_TARGET_LEN = 64
SEED = 42


def to_hf_dataset(inputs: list[str], targets: list[str]) -> Dataset:
    """Convert parallel input/target lists into a Hugging Face Dataset."""
    return Dataset.from_dict({"input": inputs, "target": targets})


def tokenize_batch(batch, tokenizer):
    """Tokenize model inputs and sequence-to-sequence targets."""
    model_inputs = tokenizer(
        batch["input"],
        max_length=MAX_INPUT_LEN,
        truncation=True,
    )

    labels = tokenizer(
        text_target=batch["target"],
        max_length=MAX_TARGET_LEN,
        truncation=True,
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def build_training_args() -> Seq2SeqTrainingArguments:
    """
    Build Trainer arguments while handling small API differences between
    Transformers versions.

    In particular:
    - Newer versions use `eval_strategy`.
    - Older versions may use `evaluation_strategy`.
    - Some versions accept `logging_dir`, while others do not.
    """
    supported_args = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters

    # Keep the original training goal/hyperparameters intact.
    kwargs = {
        "output_dir": str(OUTPUT_DIR),
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 16,
        "gradient_accumulation_steps": 2,  # effective batch size = 32
        "learning_rate": 3e-4,
        "num_train_epochs": 5,
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        # Use mixed precision only when CUDA is available.
        "fp16": torch.cuda.is_available(),
        "logging_steps": 50,
        "report_to": "tensorboard",
        "seed": SEED,
        "data_seed": SEED,
    }

    # Transformers API compatibility: evaluation strategy name changed.
    if "eval_strategy" in supported_args:
        kwargs["eval_strategy"] = "epoch"
    elif "evaluation_strategy" in supported_args:
        kwargs["evaluation_strategy"] = "epoch"
    else:
        raise RuntimeError(
            "This Transformers version exposes neither 'eval_strategy' nor "
            "'evaluation_strategy'. Please check the installed Transformers version."
        )

    # Only pass logging_dir when the installed Transformers version supports it.
    # This directly avoids the TypeError seen in the user's environment.
    if "logging_dir" in supported_args:
        kwargs["logging_dir"] = str(LOGGING_DIR)

    return Seq2SeqTrainingArguments(**kwargs)


def build_trainer(
    model,
    tokenizer,
    training_args,
    train_ds,
    val_ds,
    data_collator,
) -> Seq2SeqTrainer:
    """Create Seq2SeqTrainer with tokenizer/processor API compatibility."""
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_ds,
        "eval_dataset": val_ds,
        "data_collator": data_collator,
        "callbacks": [EarlyStoppingCallback(early_stopping_patience=2)],
    }

    supported_args = inspect.signature(Seq2SeqTrainer.__init__).parameters

    # Current Transformers uses processing_class; older releases used tokenizer.
    if "processing_class" in supported_args:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in supported_args:
        trainer_kwargs["tokenizer"] = tokenizer

    return Seq2SeqTrainer(**trainer_kwargs)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer and model: {MODEL_NAME}")
    print(f"Training device: {'CUDA GPU' if torch.cuda.is_available() else 'CPU'}")

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
        lambda batch: tokenize_batch(batch, tokenizer),
        batched=True,
        remove_columns=["input", "target"],
    )
    val_ds = val_ds.map(
        lambda batch: tokenize_batch(batch, tokenizer),
        batched=True,
        remove_columns=["input", "target"],
    )

    # Dynamic padding is more memory-efficient than padding every example to
    # MAX_INPUT_LEN/MAX_TARGET_LEN before batching.
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
    )

    training_args = build_training_args()

    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        training_args=training_args,
        train_ds=train_ds,
        val_ds=val_ds,
        data_collator=data_collator,
    )

    print("\nStarting training.")
    print(
        "Optional: open a second terminal and run `nvidia-smi -l 2` "
        "to watch GPU usage.\n"
    )

    trainer.train()

    print(f"\nSaving best model to {OUTPUT_DIR}")
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    print("Done.")
    print(f"View TensorBoard curves with: tensorboard --logdir \"{OUTPUT_DIR}\"")


if __name__ == "__main__":
    main()
