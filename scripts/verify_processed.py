"""
Verification of the processed dataset loads correctly and matches docs/data_card.md.

"""

import sys

sys.path.insert(0, "src")

from data.load_processed import load_processed_splits

EXPECTED_COUNTS = {"train": 9295, "validation": 1237, "test": 1869}


def main():
    splits = load_processed_splits()
    all_ok = True
    for name, df in splits.items():
        actual = len(df)
        expected = EXPECTED_COUNTS[name]
        status = "PASS" if actual == expected else "FAIL"
        if status == "FAIL":
            all_ok = False
        print(f"[{status}] {name}: {actual} rows (expected {expected})")
        print(f"  Dialects: {dict(df['region'].value_counts())}")

    print(
        "\nALL CHECKS PASSED"
        if all_ok
        else "\nMISMATCH -- investigate before continuing"
    )


if __name__ == "__main__":
    main()
