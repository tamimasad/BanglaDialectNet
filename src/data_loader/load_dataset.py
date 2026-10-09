"""
Dataset loading utilities for BanglaDialectNet (Phase 1).
See project docs for full column/schema notes.
"""

import glob
import os
import re
import pandas as pd  # type: ignore[import-untyped]

UNIFIED_COLUMNS = [
    "standard_bangla",
    "standard_banglish",
    "dialect_bangla",
    "dialect_banglish",
    "region",
    "english",
    "source_file",
]

_EDGE_INVISIBLES = r"[\s\u200b\u200c\u200d\ufeff\xa0\u2000-\u200a\u202f]"
_EDGE_PATTERN = re.compile(rf"^{_EDGE_INVISIBLES}+|{_EDGE_INVISIBLES}+$")


def _clean_text_edges(value):
    return _EDGE_PATTERN.sub("", str(value))


def _clean_columns(df):
    df.columns = [_clean_text_edges(c) for c in df.columns]
    return df


def _find_dialect_columns(columns):
    dialect_bangla_col = None
    dialect_banglish_col = None
    for col in columns:
        if col in ("bangla_speech", "banglish_speech"):
            continue
        if col.endswith("_bangla_speech"):
            dialect_bangla_col = col
        elif col.endswith("_banglish_speech"):
            dialect_banglish_col = col
    return dialect_bangla_col, dialect_banglish_col


def load_csv_file(path):
    df = pd.read_csv(path)
    df = _clean_columns(df)

    dialect_bangla_col, dialect_banglish_col = _find_dialect_columns(df.columns)
    if dialect_bangla_col is None or dialect_banglish_col is None:
        raise ValueError(
            f"{path}: could not find dialect-specific columns. Columns: {list(df.columns)}"
        )

    out = pd.DataFrame(
        {
            "standard_bangla": df["bangla_speech"],
            "standard_banglish": df["banglish_speech"],
            "dialect_bangla": df[dialect_bangla_col],
            "dialect_banglish": df[dialect_banglish_col],
            "region": df["region_name"],
            "english": df["english_speech"],
        }
    )
    out["source_file"] = os.path.basename(path)

    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].astype(str).map(_clean_text_edges)

    return out


def load_split(split_dir):
    csv_paths = sorted(glob.glob(os.path.join(split_dir, "*.csv")))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {split_dir}")
    frames = [load_csv_file(p) for p in csv_paths]
    return pd.concat(frames, ignore_index=True)[UNIFIED_COLUMNS]


def load_all_splits(data_root="dataset/raw"):
    return {
        split: load_split(os.path.join(data_root, split))
        for split in ["Train", "Validation", "Test"]
    }
