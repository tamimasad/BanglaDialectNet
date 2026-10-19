"""
Shared multi-task data builder for BanglaDialectNet.

Each task is generated for Bangla and Banglish inputs:
- normalize
- translate to English
- identify dialect

Bangla normalization is applied only to Bangla-script text.
"""

from normalizer import normalize as bangla_normalize

TASK_NORMALIZE = "normalize"
TASK_NORMALIZE_BANGLISH = "normalize banglish"
TASK_TRANSLATE = "translate to english"
TASK_TRANSLATE_BANGLISH = "translate to english banglish"
TASK_IDENTIFY = "identify dialect"
TASK_IDENTIFY_BANGLISH = "identify dialect banglish"

ALL_TASKS = (
    TASK_NORMALIZE,
    TASK_NORMALIZE_BANGLISH,
    TASK_TRANSLATE,
    TASK_TRANSLATE_BANGLISH,
    TASK_IDENTIFY,
    TASK_IDENTIFY_BANGLISH,
)


def build_task_examples(df, tasks=ALL_TASKS, oversample=None):
    """
    oversample: optional dict of {task_name: {region_name: multiplier}}.
    Used only on training data to upweight specific task-region pairs.
    Never apply it to validation/test data.
    """
    oversample = oversample or {}
    inputs, targets, task_names = [], [], []

    for _, row in df.iterrows():
        dialect_bangla_text = bangla_normalize(str(row["dialect_bangla"]))
        dialect_banglish_text = str(row["dialect_banglish"]).strip()
        standard_text = bangla_normalize(str(row["standard_bangla"]))
        english_text = str(row["english"])
        region = str(row["region"])

        def add(task, input_text, target_text):
            multiplier = oversample.get(task, {}).get(region, 1)
            for _ in range(multiplier):
                inputs.append(f"{task}: {input_text}")
                targets.append(target_text)
                task_names.append(task)

        if TASK_NORMALIZE in tasks:
            add(TASK_NORMALIZE, dialect_bangla_text, standard_text)
        if TASK_NORMALIZE_BANGLISH in tasks:
            add(TASK_NORMALIZE_BANGLISH, dialect_banglish_text, standard_text)
        if TASK_TRANSLATE in tasks:
            add(TASK_TRANSLATE, dialect_bangla_text, english_text)
        if TASK_TRANSLATE_BANGLISH in tasks:
            add(TASK_TRANSLATE_BANGLISH, dialect_banglish_text, english_text)
        if TASK_IDENTIFY in tasks:
            add(TASK_IDENTIFY, dialect_bangla_text, region)
        if TASK_IDENTIFY_BANGLISH in tasks:
            add(TASK_IDENTIFY_BANGLISH, dialect_banglish_text, region)

    return inputs, targets, task_names
