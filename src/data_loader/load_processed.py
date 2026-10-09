"""
Loader for the CLEANED, processed BanglaDialectNet dataset (Phase 2).

"""

import glob
import os
import pandas as pd

PROCESSED_ROOT = "dataset/processed"
SPLIT_FOLDERS = {"train": "Train", "validation": "Validation", "test": "Test"}


def load_processed_split(split_name):
    """Load and concatenate all dialect CSVs for one split (train/validation/test)."""
    folder = SPLIT_FOLDERS[split_name]
    split_dir = os.path.join(PROCESSED_ROOT, folder)
    csv_paths = sorted(glob.glob(os.path.join(split_dir, "*.csv")))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {split_dir}")
    frames = [pd.read_csv(p) for p in csv_paths]
    return pd.concat(frames, ignore_index=True)


def load_processed_splits():
    """Load train, validation, and test. Returns a dict of DataFrames."""
    return {name: load_processed_split(name) for name in SPLIT_FOLDERS}
