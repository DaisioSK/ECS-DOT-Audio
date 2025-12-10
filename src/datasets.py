"""Dataset helpers for mel spectrogram indices."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .config import LABEL_TO_ID, NUM_CLASSES


def balance_folds(index_df: pd.DataFrame,
                  target_ratio: float = 0.4,
                  random_state: int = 42) -> pd.DataFrame:
    """Downsample/upsample background per fold to approach target glass ratio."""
    rng = np.random.default_rng(random_state)
    balanced_parts = []
    for fold_id, fold_df in index_df.groupby("fold_id"):
        if "label_ids" not in fold_df.columns:
            fold_df = fold_df.copy()
            fold_df["label_ids"] = fold_df["label"].apply(
                lambda lab: [LABEL_TO_ID[lab]] if lab in LABEL_TO_ID else []
            )
        pos_mask = fold_df["label_ids"].apply(lambda ids: bool(ids))
        pos_df = fold_df[pos_mask]
        bg_df = fold_df[~pos_mask]
        desired_bg = int(round(len(pos_df) * (1 - target_ratio) / target_ratio)) if len(pos_df) else 0
        if len(bg_df) > desired_bg > 0:
            bg_sample = bg_df.sample(n=desired_bg, random_state=random_state)
        else:
            bg_sample = bg_df
        balanced_parts.append(pd.concat([pos_df, bg_sample], ignore_index=True))
    if not balanced_parts:
        return index_df.copy()
    return pd.concat(balanced_parts, ignore_index=True)


@dataclass
class MelEntry:
    path: Path
    label_ids: List[int]


class MelDataset(Dataset):
    """Torch dataset that loads mel .npy files based on cache index."""

    def __init__(self, index_df: pd.DataFrame, max_items: int | None = None):
        entries: List[MelEntry] = []
        for _, row in index_df.iterrows():
            label_ids = row.get("label_ids", [])
            if not isinstance(label_ids, (list, tuple, np.ndarray)):
                label_ids = [LABEL_TO_ID.get(row.get("label"), -1)]
            entries.append(MelEntry(path=Path(row["path"]), label_ids=list(label_ids)))
        if max_items is not None:
            entries = entries[:max_items]
        self.entries = entries

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        entry = self.entries[idx]
        mel = np.load(entry.path)
        mel_tensor = torch.from_numpy(mel).float()
        target = torch.zeros(NUM_CLASSES, dtype=torch.float32)
        for lid in entry.label_ids:
            if 0 <= lid < NUM_CLASSES:
                target[lid] = 1.0
        return mel_tensor.unsqueeze(0), target


def load_index_df(index_path: str | Path) -> pd.DataFrame:
    """Load cache index from CSV or Parquet."""
    path = Path(index_path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def subset_by_folds(index_df: pd.DataFrame, folds: Sequence[int]) -> pd.DataFrame:
    """Return rows belonging to selected folds."""
    return index_df[index_df["fold_id"].isin(folds)].reset_index(drop=True)


def pad_mel_batch(batch: List[Tuple[torch.Tensor, torch.Tensor]],
                  max_frames: int | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad or crop mel tensors in a batch to a common time length."""
    if not batch:
        raise ValueError("Empty batch provided to pad_mel_batch")
    mels, labels = zip(*batch)
    current_max = max(mel.shape[-1] for mel in mels)
    target_frames = min(max_frames, current_max) if max_frames is not None else current_max
    padded = []
    for mel in mels:
        frames = mel.shape[-1]
        if frames > target_frames:
            mel = mel[..., :target_frames]
        elif frames < target_frames:
            mel = F.pad(mel, (0, target_frames - frames))
        padded.append(mel)
    batch_tensor = torch.stack(padded, dim=0)
    label_tensor = torch.stack(labels, dim=0)
    return batch_tensor, label_tensor


def build_dataloaders(index_df: pd.DataFrame,
                      train_folds: Sequence[int],
                      val_folds: Sequence[int],
                      batch_size: int = 64,
                      num_workers: int = 0,
                      smoke_limit: int | None = None,
                      random_state: int = 42,
                      collate_max_frames: int | None = None) -> Tuple[DataLoader, DataLoader]:
    """Create train/validation dataloaders with optional smoke limiting."""
    rng = np.random.default_rng(random_state)

    def _limit(df: pd.DataFrame) -> pd.DataFrame:
        if smoke_limit is None or df.empty:
            return df
        take = min(smoke_limit, len(df))
        indices = rng.choice(len(df), size=take, replace=False)
        return df.iloc[sorted(indices)].reset_index(drop=True)

    train_df = subset_by_folds(index_df, train_folds)
    val_df = subset_by_folds(index_df, val_folds)
    train_df = _limit(train_df)
    val_df = _limit(val_df)

    train_dataset = MelDataset(train_df)
    val_dataset = MelDataset(val_df)

    collate_fn = lambda batch: pad_mel_batch(batch, max_frames=collate_max_frames)  # noqa: E731
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader


__all__ = [
    "balance_folds",
    "MelDataset",
    "load_index_df",
    "subset_by_folds",
    "pad_mel_batch",
    "build_dataloaders",
]
