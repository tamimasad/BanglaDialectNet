"""
Data Analysis for BanglaDialectNet (Phase 1, Step 4).

Runs on the VALIDATED raw data before any processing decisions.

What this checks:
  1. Sentence length distribution (characters & words) per text column,
     per dialect -- informs tokenizer max_length choices.
  2. Character-set sanity check -- flags any unexpected non-Bangla,
     non-Latin, non-punctuation characters.
  3. Train/validation/test leakage check.

Outputs:
  reports/eda_summary.md      -- human-readable summary
  reports/figures/*.png       -- length distribution plots

"""

import os
import sys
import unicodedata
from collections import Counter

import matplotlib

matplotlib.use("Agg")  # no GUI needed -- just save PNG files
import matplotlib.pyplot as plt

sys.path.insert(0, "src")
from data_loader.load_dataset import load_all_splits

FIGURES_DIR = "reports/figures"
SUMMARY_PATH = "reports/dataset_summary.md"
TEXT_COLUMNS = ["standard_bangla", "dialect_bangla", "english"]


def length_stats(df, col):
    """Return character and word length stats for one column."""
    chars = df[col].astype(str).map(len)
    words = df[col].astype(str).map(lambda s: len(s.split()))
    return {
        "char_mean": chars.mean(),
        "char_median": chars.median(),
        "char_min": chars.min(),
        "char_max": chars.max(),
        "word_mean": words.mean(),
        "word_min": words.min(),
        "word_max": words.max(),
    }


def plot_length_distribution(df, col, split_name, out_dir):
    """Save a histogram of character lengths for one column, one split."""
    lengths = df[col].astype(str).map(len)
    plt.figure(figsize=(8, 4))
    plt.hist(lengths, bins=40)
    plt.title(f"{col} character length -- {split_name}")
    plt.xlabel("characters")
    plt.ylabel("row count")
    plt.tight_layout()
    path = os.path.join(out_dir, f"{split_name}_{col}_length.png")
    plt.savefig(path, dpi=100)
    plt.close()
    return path


def character_set_report(df, columns):
    """
    Count characters outside expected ranges: Bangla script
    (U+0980-U+09FF), basic Latin letters/digits, and common
    punctuation.
    """
    unexpected = Counter()
    for col in columns:
        for text in df[col].astype(str):
            for ch in text:
                if ch.isspace():
                    continue
                code = ord(ch)
                is_bangla = 0x0980 <= code <= 0x09FF
                is_basic_latin = ch.isascii() and (ch.isalnum() or ch in ".,!?'\"-")
                if not (is_bangla or is_basic_latin):
                    name = unicodedata.name(ch, f"U+{code:04X}")
                    unexpected[f"{ch!r} ({name})"] += 1
    return unexpected


def leakage_check(splits):
    """
    Check whether any exact (dialect_bangla, standard_bangla) pair
    appears in more than one split.
    """
    split_sets = {
        name: set(zip(df["dialect_bangla"], df["standard_bangla"]))
        for name, df in splits.items()
    }
    leaks = {}
    items = list(split_sets.items())
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            name_a, set_a = items[i]
            name_b, set_b = items[j]
            overlap = set_a & set_b
            if overlap:
                leaks[f"{name_a} <-> {name_b}"] = len(overlap)
    return leaks


def identical_rows_by_dialect(df):
    """Count rows where dialect_bangla == standard_bangla, per dialect."""
    identical = df[df["dialect_bangla"] == df["standard_bangla"]]
    return identical["region"].value_counts().to_dict()


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    splits = load_all_splits()
    report_lines = ["# BanglaDialectNet -- Exploratory Data Analysis\n"]

    for split_name, df in splits.items():
        report_lines.append(f"\n## {split_name.capitalize()} split ({len(df)} rows)\n")
        for col in TEXT_COLUMNS:
            stats = length_stats(df, col)
            report_lines.append(
                f"- **{col}**: chars mean={stats['char_mean']:.1f} "
                f"median={stats['char_median']:.0f} "
                f"range=[{stats['char_min']}, {stats['char_max']}] | "
                f"words mean={stats['word_mean']:.1f} "
                f"range=[{stats['word_min']}, {stats['word_max']}]"
            )
            path = plot_length_distribution(df, col, split_name, FIGURES_DIR)
            print(f"Saved {path}")

    print("\nChecking character set (train split)...")
    unexpected = character_set_report(splits["Train"], TEXT_COLUMNS)
    report_lines.append("\n## Unexpected characters (train split)\n")
    if unexpected:
        for ch, count in unexpected.most_common(20):
            report_lines.append(f"- {ch}: {count} occurrences")
    else:
        report_lines.append(
            "- None found -- all text is within expected Bangla/Latin/punctuation ranges."
        )

    print("Checking for train/validation/test leakage...")
    leaks = leakage_check(splits)
    report_lines.append("\n## Train/validation/test leakage check\n")
    if leaks:
        for pair, count in leaks.items():
            report_lines.append(
                f"- [WARN] {pair}: {count} overlapping (dialect, standard) pairs"
            )
    else:
        report_lines.append(
            "- None found -- no exact pair appears in more than one split."
        )

    report_lines.append(
        "\n## Identical dialect/standard text rows, by dialect (train split)\n"
    )
    identical_counts = identical_rows_by_dialect(splits["Train"])
    if identical_counts:
        for dialect, count in identical_counts.items():
            report_lines.append(f"- {dialect}: {count} rows")
    else:
        report_lines.append("- None.")

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\nDone. Summary written to {SUMMARY_PATH}, plots in {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
