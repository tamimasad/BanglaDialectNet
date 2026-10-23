"""
Evaluate a fine-tuned banglat5 model, covering all 6 tasks
(3 outputs x 2 input scripts: Bangla, Banglish).
"""

import os
import sys

sys.path.insert(0, "src")

import pandas as pd
import torch
from sacrebleu.metrics import BLEU, CHRF
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

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

MODEL_DIR = "models/BanglaDINT-T5"
SPLIT = "test"

MAX_TARGET_LEN = 64
GEN_BATCH_SIZE = 32
NUM_BEAMS = 4
REPORTS_DIR = "reports"
DOCS_DIR = "docs"

TASK_GROUPS = [
    ("normalize", TASK_NORMALIZE, TASK_NORMALIZE_BANGLISH, False),
    ("translate_to_english", TASK_TRANSLATE, TASK_TRANSLATE_BANGLISH, False),
    ("identify_dialect", TASK_IDENTIFY, TASK_IDENTIFY_BANGLISH, True),
]


def generate_predictions(model, tokenizer, inputs, device):
    predictions = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(inputs), GEN_BATCH_SIZE):
            batch = inputs[i : i + GEN_BATCH_SIZE]
            encoded = tokenizer(
                batch, padding=True, truncation=True, max_length=64, return_tensors="pt"
            ).to(device)
            output_ids = model.generate(
                **encoded, max_length=MAX_TARGET_LEN, num_beams=NUM_BEAMS
            )
            decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            predictions.extend(decoded)
            print(
                f"  generated {min(i + GEN_BATCH_SIZE, len(inputs))}/{len(inputs)}",
                end="\r",
            )
    print()
    return predictions


def exact_match_accuracy(predictions, targets):
    correct = sum(p.strip() == t.strip() for p, t in zip(predictions, targets))
    return 100.0 * correct / len(targets)


def bleu_and_chrf(predictions, targets):
    bleu_score = BLEU().corpus_score(predictions, [targets]).score
    chrf_score = CHRF().corpus_score(predictions, [targets]).score
    return bleu_score, chrf_score


def evaluate_one_task(model, tokenizer, device, df, task, all_rows):
    inputs, targets, _ = build_task_examples(df, tasks=(task,))
    predictions = generate_predictions(model, tokenizer, inputs, device)
    for inp, tgt, pred, region in zip(inputs, targets, predictions, df["region"]):
        all_rows.append(
            {
                "task": task,
                "region": region,
                "input": inp,
                "target": tgt,
                "prediction": pred,
            }
        )
    return inputs, targets, predictions


def main():
    if SPLIT not in ("train", "validation", "test"):
        raise ValueError(f"Unexpected SPLIT: {SPLIT!r}")
    if SPLIT == "test":
        print(
            "WARNING: evaluating on the TEST set. This should happen only once, "
            "as the final step after all tuning is done. Ctrl+C now to cancel.\n"
        )

    model_label = os.path.basename(MODEL_DIR.rstrip("/\\"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading fine-tuned model from {MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR).to(device)

    df = load_processed_splits()[SPLIT]
    print(f"{SPLIT} set: {len(df)} rows\n")

    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(DOCS_DIR, exist_ok=True)

    report_lines = [
        f"# BanglaDialectNet -- {model_label} Results ({SPLIT} set)",
        "## Bangla-script vs Banglish input, side by side\n",
    ]
    all_rows = []

    for output_name, bangla_task, banglish_task, is_classification in TASK_GROUPS:
        print(f"Evaluating: {output_name}")
        report_lines.append(f"## Task: {output_name}")

        for variant_label, task in (
            ("Bangla script", bangla_task),
            ("Banglish", banglish_task),
        ):
            print(f"  variant: {variant_label}")
            inputs, targets, predictions = evaluate_one_task(
                model, tokenizer, device, df, task, all_rows
            )

            if is_classification:
                accuracy = exact_match_accuracy(predictions, targets)
                report_lines.append(f"- {variant_label}: accuracy={accuracy:.1f}%")

                per_dialect = {}
                for pred, tgt, region in zip(predictions, targets, df["region"]):
                    per_dialect.setdefault(region, [0, 0])
                    per_dialect[region][1] += 1
                    if pred.strip() == tgt.strip():
                        per_dialect[region][0] += 1
                for region, (correct, total) in sorted(per_dialect.items()):
                    report_lines.append(
                        f"    - {region}: {100*correct/total:.1f}% ({correct}/{total})"
                    )
            else:
                bleu, chrf = bleu_and_chrf(predictions, targets)
                exact = exact_match_accuracy(predictions, targets)
                report_lines.append(
                    f"- {variant_label}: BLEU={bleu:.2f}  chrF={chrf:.2f}  exact-match={exact:.1f}%"
                )

                # Per-dialect BLEU/chrF
                per_dialect_data = {}
                for pred, tgt, region in zip(predictions, targets, df["region"]):
                    per_dialect_data.setdefault(region, {"preds": [], "tgts": []})
                    per_dialect_data[region]["preds"].append(pred)
                    per_dialect_data[region]["tgts"].append(tgt)
                for region in sorted(per_dialect_data):
                    d = per_dialect_data[region]
                    d_bleu, d_chrf = bleu_and_chrf(d["preds"], d["tgts"])
                    report_lines.append(
                        f"    - {region}: BLEU={d_bleu:.2f}  chrF={d_chrf:.2f}"
                    )

                if variant_label == "Bangla script" and output_name == "normalize":
                    raw_dialect_text = [
                        inp.replace(f"{bangla_task}: ", "", 1) for inp in inputs
                    ]
                    copy_bleu, copy_chrf = bleu_and_chrf(raw_dialect_text, targets)
                    report_lines.append(
                        f"    (copy baseline, no model: BLEU={copy_bleu:.2f}  chrF={copy_chrf:.2f})"
                    )
        report_lines.append("")

    results_path = os.path.join(DOCS_DIR, f"results_{model_label}_{SPLIT}.md")
    predictions_path = os.path.join(
        REPORTS_DIR, f"predictions_{model_label}_{SPLIT}.csv"
    )

    with open(results_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    pd.DataFrame(all_rows).to_csv(predictions_path, index=False, encoding="utf-8-sig")

    print("\n".join(report_lines))
    print(f"\nFull results: {results_path}")
    print(f"Per-row predictions: {predictions_path}")


if __name__ == "__main__":
    main()
