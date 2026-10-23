"""
Confusion matrix for the identify_dialect task (Phase 2 diagnostic).

"""

import pandas as pd

EXPECTED_DIALECTS = ["Noakhali", "Barishal", "Chittagong", "Sylhet", "Mymensingh"]
TASKS_TO_CHECK = ["identify dialect", "identify dialect banglish"]


def build_confusion(df, task_name):
    subset = df[df["task"] == task_name]
    confusion = pd.crosstab(
        subset["region"],
        subset["prediction"].astype(str).str.strip(),
        rownames=["true"],
        colnames=["predicted"],
    )
    known_cols = [c for c in EXPECTED_DIALECTS if c in confusion.columns]
    other_cols = [c for c in confusion.columns if c not in EXPECTED_DIALECTS]
    return confusion[known_cols + other_cols]


def main():
    df = pd.read_csv("reports/predictions_BanglaDINT-T5-production_test.csv")

    for task_name in TASKS_TO_CHECK:
        print(f"\n{'='*60}\nConfusion matrix: {task_name}\n{'='*60}")
        confusion = build_confusion(df, task_name)
        print(confusion.to_string())

        for true_dialect in EXPECTED_DIALECTS:
            if true_dialect not in confusion.index:
                continue
            row = confusion.loc[true_dialect]
            total = row.sum()
            correct = row.get(true_dialect, 0)
            if correct / total < 0.6:  # flag any dialect under 60% for a closer look
                print(
                    f"\n  [WEAK] {true_dialect} ({100*correct/total:.1f}% correct, {total} rows) -- misclassified as:"
                )
                for predicted, count in row.sort_values(ascending=False).items():
                    if predicted != true_dialect and count > 0:
                        print(f"    -> '{predicted}': {count} ({100*count/total:.1f}%)")


if __name__ == "__main__":
    main()
