"""Dataset helpers for mel spectrogram indices."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import POSITIVE_LABELS

GLASS_LABEL = list(POSITIVE_LABELS.values())[0]
LABEL_TO_ID = {GLASS_LABEL: 1, "background": 0}


def balance_folds(index_df: pd.DataFrame,
                  target_ratio: float = 0.4,
                  random_state: int = 42) -> pd.DataFrame:
    """Downsample/upsample background per fold to approach target glass ratio."""
    rng = np.random.default_rng(random_state)
    balanced_parts = []
    for fold_id, fold_df in index_df.groupby("fold_id"):
        glass_df = fold_df[fold_df["label"] == GLASS_LABEL]
        bg_df = fold_df[fold_df["label"] != GLASS_LABEL]
        desired_bg = int(round(len(glass_df) * (1 - target_ratio) / target_ratio)) if len(glass_df) else 0
        if len(bg_df) > desired_bg > 0:
            bg_sample = bg_df.sample(n=desired_bg, random_state=random_state)
        else:
            bg_sample = bg_df
        balanced_parts.append(pd.concat([glass_df, bg_sample], ignore_index=True))
    if not balanced_parts:
        return index_df.copy()
    return pd.concat(balanced_parts, ignore_index=True)


@dataclass
class MelEntry:
    path: Path
    label_id: int


class MelDataset(Dataset):
    """Torch dataset that loads mel .npy files based on cache index."""

    def __init__(self, index_df: pd.DataFrame):
        entries = []
        for _, row in index_df.iterrows():
            label_id = LABEL_TO_ID.get(row["label"], 0)
            entries.append(MelEntry(path=Path(row["path"]), label_id=label_id))
        self.entries = entries

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        entry = self.entries[idx]
        mel = np.load(entry.path)
        mel_tensor = torch.from_numpy(mel).float()
        return mel_tensor.unsqueeze(0), entry.label_id


__all__ = ["balance_folds", "MelDataset"]
