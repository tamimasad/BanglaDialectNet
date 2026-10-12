"""
Shared multi-task example builder for BanglaDialectNet.

For each of the 3 output tasks (normalize, translate-to-english,
identify-dialect), TWO input variants are generated:
    - from dialect_bangla   (Bangla script)   -> task prefix as before
    - from dialect_banglish (romanized/Latin) -> task prefix + " banglish"

This keeps everything in ONE multi-task model: the model distinguishes
script both from the prefix and from the actual characters it sees.

bangla_normalize() is applied ONLY to Bangla-script text -- it's a
Bangla-Unicode-specific pipeline, not meant for romanized Latin text.

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


def build_task_examples(df, tasks=ALL_TASKS):
    """Expand each row into (input, target, task) triples for the given tasks."""
    inputs, targets, task_names = [], [], []

    for _, row in df.iterrows():
        dialect_bangla_text = bangla_normalize(str(row["dialect_bangla"]))
        dialect_banglish_text = str(row["dialect_banglish"]).strip()
        standard_text = bangla_normalize(str(row["standard_bangla"]))
        english_text = str(row["english"])
        region = str(row["region"])

        if TASK_NORMALIZE in tasks:
            inputs.append(f"{TASK_NORMALIZE}: {dialect_bangla_text}")
            targets.append(standard_text)
            task_names.append(TASK_NORMALIZE)

        if TASK_NORMALIZE_BANGLISH in tasks:
            inputs.append(f"{TASK_NORMALIZE_BANGLISH}: {dialect_banglish_text}")
            targets.append(standard_text)
            task_names.append(TASK_NORMALIZE_BANGLISH)

        if TASK_TRANSLATE in tasks:
            inputs.append(f"{TASK_TRANSLATE}: {dialect_bangla_text}")
            targets.append(english_text)
            task_names.append(TASK_TRANSLATE)

        if TASK_TRANSLATE_BANGLISH in tasks:
            inputs.append(f"{TASK_TRANSLATE_BANGLISH}: {dialect_banglish_text}")
            targets.append(english_text)
            task_names.append(TASK_TRANSLATE_BANGLISH)

        if TASK_IDENTIFY in tasks:
            inputs.append(f"{TASK_IDENTIFY}: {dialect_bangla_text}")
            targets.append(region)
            task_names.append(TASK_IDENTIFY)

        if TASK_IDENTIFY_BANGLISH in tasks:
            inputs.append(f"{TASK_IDENTIFY_BANGLISH}: {dialect_banglish_text}")
            targets.append(region)
            task_names.append(TASK_IDENTIFY_BANGLISH)

    return inputs, targets, task_names
