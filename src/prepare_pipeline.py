"""Prepare pipeline helpers (API-first).

Goal: move stable, reusable steps from notebooks into `src/` so that:
- prepare/train/infer share the same preprocessing
- notebooks become orchestration + QA, not the source of truth
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from .config import (
    BACKGROUND_LABEL,
    HOP_LENGTH,
    LABEL_TO_ID,
    MEL_CENTER,
    MEL_TARGET_FRAMES,
    N_FFT,
    N_MELS,
    PROJECT_ROOT,
    SR,
    TARGET_LABELS,
)
from .data_utils import load_audio
from .preprocess import log_mel, pad_or_crop_frames
from .meta_utils import deduplicate_meta, load_meta_files, map_canonical_labels, sample_gunshot_even


def load_and_split_meta(
    meta_files: Iterable[str | Path],
    *,
    label_map: dict[str, str],
    target_labels: List[str] | None = None,
    include_sources: Optional[List[str]] = None,
    dedup_subset: Optional[List[str]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load meta CSVs, map canonical labels, deduplicate, and split by source.

    This is the stable "ingestion" part of prepare:
    - Reads multiple CSVs with the unified schema.
    - Maps `label` -> `canonical_label` (and sets `is_target`).
    - Deduplicates to avoid leaking the same audio across sources.
    - Splits a working subset (for train/cache) vs holdout (future detection).
    """

    target_labels = target_labels or list(TARGET_LABELS)
    include_sources = include_sources or []
    dedup_subset = dedup_subset or ["md5", "filepath"]

    meta_df = load_meta_files(meta_files)
    meta_df = map_canonical_labels(meta_df, label_map=label_map, target_labels=target_labels)
    meta_df = deduplicate_meta(meta_df, subset=dedup_subset)

    if include_sources:
        working_df = meta_df[meta_df["source"].isin(include_sources)].copy()
        holdout_df = meta_df[~meta_df["source"].isin(include_sources)].copy()
    else:
        working_df = meta_df.copy()
        holdout_df = meta_df.iloc[0:0].copy()

    return meta_df, working_df, holdout_df


def filter_and_sample_gunshot(
    working_df: pd.DataFrame,
    *,
    max_duration_sec: float,
    gunshot_total: int,
    seed: int,
    target_label: str = "gunshot",
    label_col: str = "canonical_label",
    duration_col: str = "duration_sec",
) -> tuple[pd.DataFrame, dict]:
    """Filter long clips and sample gunshots evenly (weapon-balanced).

    Returns:
        (clean_df, debug_info)
    """

    clean_df = working_df[working_df[duration_col] <= max_duration_sec].copy()
    gun_sampled = sample_gunshot_even(
        clean_df, target_label=target_label, total=gunshot_total, seed=seed
    )

    debug: dict = {
        "before_rows": int(len(working_df)),
        "after_duration_filter_rows": int(len(clean_df)),
        "gunshot_sampled_rows": int(len(gun_sampled)),
    }

    if not gun_sampled.empty:
        debug["gunshot_sampled_by_source"] = gun_sampled.groupby("source").size().to_dict()
        # Parent folder is a reasonable proxy for weapon_id in current dataset.
        debug["gunshot_sampled_by_weapon_folder"] = (
            gun_sampled.apply(lambda r: Path(str(r.get("filepath", ""))).parent.name, axis=1)
            .value_counts()
            .to_dict()
        )

    non_gun = clean_df[clean_df[label_col] != target_label]
    clean_df = pd.concat([non_gun, gun_sampled], ignore_index=True)
    debug["final_label_counts"] = clean_df[label_col].value_counts().to_dict()
    debug["final_rows"] = int(len(clean_df))
    return clean_df, debug


def cache_windows_to_mel_index(
    windows_df: pd.DataFrame,
    cache_dir: Path,
    index_csv: Path,
    *,
    target_sr: int = SR,
    target_frames: int = MEL_TARGET_FRAMES,
    center: bool = MEL_CENTER,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    n_mels: int = N_MELS,
    target_labels: Optional[List[str]] = None,
    label_to_id: Optional[dict[str, int]] = None,
    background_label: str = BACKGROUND_LABEL,
    save_float32: bool = True,
) -> pd.DataFrame:
    """Cache window-level audio into mel `.npy` and write a training index CSV.

    Expected input:
      windows_df: one row per window, with at least:
        - `target_label` (canonical label or background-like)
        - `fold_id`
        - `clip_id`
        - `window_id`
      Optional (for loading audio):
        - `audio` (np.ndarray) if already sliced/augmented
        - or filepath columns compatible with `src.data_utils.load_audio` (e.g. `filepath`)

    Output index columns:
      - path: relative path to mel `.npy`
      - target_label: raw target label (may include non-target labels if caller wants)
      - label: main label used for folder grouping (targets or background)
      - labels: list[str] for BCE multi-hot (targets only; background -> [])
      - label_ids: list[int] aligned to TARGET_LABELS
      - fold_id, pipeline, clip_id, window_id, source, orig_label, length_sec, shape
    """

    target_labels = target_labels or list(TARGET_LABELS)
    label_to_id = label_to_id or dict(LABEL_TO_ID)

    cache_dir = Path(cache_dir)
    index_csv = Path(index_csv)
    cache_dir.mkdir(parents=True, exist_ok=True)
    index_csv.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    for idx, row in windows_df.iterrows():
        # Audio: prefer in-row waveform to avoid re-slicing / re-loading.
        y = row.get("audio") if isinstance(row, pd.Series) else None
        if y is not None:
            y = np.asarray(y)
            sr = int(target_sr)
        else:
            y, sr = load_audio(row, sr=target_sr)

        tgt = row.get("target_label", background_label)
        fold = int(row.get("fold_id", -1))
        clip_id = row.get("clip_id") or Path(str(row.get("filepath", ""))).stem or f"clip{idx}"
        window_id = row.get("window_id") or f"w{idx}"

        labels = [tgt] if tgt in target_labels else []
        label_ids = [label_to_id[l] for l in labels]
        main_label = labels[0] if labels else background_label

        mel = log_mel(
            y,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            center=center,
        )
        if target_frames is not None:
            mel = pad_or_crop_frames(mel, target_frames=int(target_frames))

        out_dir = cache_dir / main_label / f"fold{fold}"
        out_dir.mkdir(parents=True, exist_ok=True)
        cache_path = out_dir / f"{clip_id}_{window_id}.npy"
        np.save(cache_path, mel.astype(np.float32, copy=False) if save_float32 else mel)

        records.append(
            {
                "path": str(cache_path.relative_to(PROJECT_ROOT)),
                "target_label": tgt,
                "label": main_label,
                "labels": labels,
                "label_ids": label_ids,
                "fold_id": fold,
                "pipeline": row.get("pipeline", "base"),
                "clip_id": clip_id,
                "window_id": window_id,
                "source": row.get("source", ""),
                "shape": tuple(mel.shape),
                "orig_label": row.get("original_label", row.get("orig_label", tgt)),
                "length_sec": float(row.get("length_sec", len(y) / sr)),
            }
        )

    index_df = pd.DataFrame(records)
    index_df.to_csv(index_csv, index=False)
    return index_df
