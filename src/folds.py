"""Utilities for assigning cross-validation folds."""
from __future__ import annotations

import pandas as pd


def assign_folds(dataset_df: pd.DataFrame, fold_column: str = "fold") -> pd.Series:
    """Return fold id per row, defaulting to metadata fold column."""
    if fold_column not in dataset_df.columns:
        raise KeyError(f"Column '{fold_column}' missing from dataset_df")
    return dataset_df[fold_column].astype(int)


def fold_summary(dataset_df: pd.DataFrame,
                 fold_column: str = "fold",
                 label_column: str = "target_label") -> pd.DataFrame:
    """Aggregate counts per fold/label for inspection."""
    if fold_column not in dataset_df or label_column not in dataset_df:
        raise KeyError("Required columns missing for fold summary")
    return (
        dataset_df.groupby([fold_column, label_column])
        .size()
        .reset_index(name="count")
    )
