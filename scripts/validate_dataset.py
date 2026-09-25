"""
Validate the raw BanglaDialectNet dataset (Phase 1).

"""

import sys
from pathlib import Path

""" Make the project's source directory importable regardless of the current
working directory when this script is run."""
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.load_dataset import load_all_splits  # type: ignore[import-not-found]

EXPECTED_DIALECTS = {"Noakhali", "Barishal", "Chittagong", "Sylhet", "Mymensingh"}


def validate_split(name, df):
    print(f"\n=== {name.upper()} ===")
    print(f"Total rows: {len(df)}")

    counts = df["region"].value_counts()
    print("Rows per dialect:")
    for dialect, count in counts.items():
        flag = "" if dialect in EXPECTED_DIALECTS else "  <-- UNEXPECTED LABEL"
        print(f"  {dialect}: {count}{flag}")

    missing = EXPECTED_DIALECTS - set(counts.index)
    if missing:
        print(f"  MISSING from this split entirely: {missing}")

    for col in df.columns:
        empty = (df[col].isna() | (df[col].astype(str).str.strip() == "")).sum()
        if empty:
            print(f"  [WARN] {empty} empty values in '{col}'")

    dupes = df.duplicated().sum()
    if dupes:
        print(f"  [WARN] {dupes} exact duplicate rows")

    identical = (df["dialect_bangla"] == df["standard_bangla"]).sum()
    if identical:
        pct = 100 * identical / len(df)
        print(
            f"  [WARN] {identical} rows ({pct:.1f}%) have IDENTICAL dialect and standard Bangla text"
        )


def main():
    for name, df in load_all_splits().items():
        validate_split(name, df)
    print("\nDone.")


if __name__ == "__main__":
    main()
