"""
Shared multi-task example builder for BanglaDialectNet.

Turns one row of the processed dataset into (input, target) pairs
across 3 tasks sharing one banglat5 model:
    normalize: <dialect text>            -> standard Bangla
    translate to english: <dialect text> -> English
    identify dialect: <dialect text>     -> dialect label

"""

from normalizer import normalize as bangla_normalize

TASK_NORMALIZE = "normalize"
TASK_TRANSLATE = "translate to english"
TASK_IDENTIFY = "identify dialect"
ALL_TASKS = (TASK_NORMALIZE, TASK_TRANSLATE, TASK_IDENTIFY)


def build_task_examples(df, tasks=ALL_TASKS):
    """Expand each row into (input, target, task) triples for the given tasks."""
    inputs, targets, task_names = [], [], []

    for _, row in df.iterrows():
        dialect_text = bangla_normalize(str(row["dialect_bangla"]))
        standard_text = bangla_normalize(str(row["standard_bangla"]))
        english_text = str(row["english"])
        region = str(row["region"])

        if TASK_NORMALIZE in tasks:
            inputs.append(f"{TASK_NORMALIZE}: {dialect_text}")
            targets.append(standard_text)
            task_names.append(TASK_NORMALIZE)

        if TASK_TRANSLATE in tasks:
            inputs.append(f"{TASK_TRANSLATE}: {dialect_text}")
            targets.append(english_text)
            task_names.append(TASK_TRANSLATE)

        if TASK_IDENTIFY in tasks:
            inputs.append(f"{TASK_IDENTIFY}: {dialect_text}")
            targets.append(region)
            task_names.append(TASK_IDENTIFY)

    return inputs, targets, task_names
