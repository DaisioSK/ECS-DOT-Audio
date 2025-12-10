"""Caching utilities for mel spectrogram windows with metadata tracking."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from .config import (
    BACKGROUND_LABEL,
    CACHE_DIR,
    TARGET_LABELS,
    LABEL_TO_ID,
    POSITIVE_LABELS,
    SR,
    SEED,
)
from .data_utils import generate_aligned_windows, load_audio, log_mel_spectrogram
from .augment_pipeline import PIPELINE_REGISTRY, run_pipeline

# Expose pipeline plan slots so notebooks/scripts can reference them.
GLASS_PIPELINE_PLAN: Dict[str, Dict[str, int]] = {
    "shift_gain": {"copies": 2},
    "stretch_reverb": {"copies": 2},
    "shift_mix": {"copies": 2},
    "filter_gain": {"copies": 2},
    "gain_mix": {"copies": 1},
    "stretch_filter": {"copies": 1},
}
GLASS_LABELS: List[str] = list(TARGET_LABELS)


@dataclass
class CacheEntry:
    """Metadata describing a cached mel tile."""

    path: str
    labels: List[str]
    label_ids: List[int]
    label: str  # primary label for backward compatibility
    fold_id: int
    source_filename: str
    clip_id: str
    window_id: str
    pipeline_name: str
    augment_desc: str
    source_type: str


def _save_window(window: np.ndarray,
                 label: str,
                 clip_id: str,
                 fold_id: int,
                 suffix: str,
                 cache_dir: Path) -> Path:
    """Compute mel for a waveform window and write it to disk."""
    label_dir = cache_dir / label / f"fold{fold_id}"
    label_dir.mkdir(parents=True, exist_ok=True)
    cache_path = label_dir / f"{clip_id}_{suffix}.npy"
    mel = log_mel_spectrogram(window, SR)
    np.save(cache_path, mel.astype(np.float32))
    return cache_path


def sample_background_chunk(dataset_df: pd.DataFrame,
                            length: int,
                            rng: np.random.Generator | None = None) -> np.ndarray:
    """Pick a random background clip and return a chunk of desired length."""
    rng = rng or np.random.default_rng(SEED)
    background_df = dataset_df[dataset_df["target_label"] == BACKGROUND_LABEL]
    if background_df.empty:
        raise ValueError("Background dataframe is empty; cannot mix audio.")
    idx = rng.integers(0, len(background_df))
    row = background_df.iloc[idx]
    y_bg, _ = load_audio(row)
    if len(y_bg) < length:
        y_bg = np.pad(y_bg, (0, length - len(y_bg)))
    start = rng.integers(0, max(1, len(y_bg) - length + 1))
    return y_bg[start:start + length]


def _encode_labels(raw_label: str | Sequence[str] | None) -> tuple[List[str], List[int], str]:
    """Normalize raw label input into canonical lists and primary label."""
    if raw_label is None:
        return [], [], BACKGROUND_LABEL
    if isinstance(raw_label, str):
        labels = [] if raw_label == BACKGROUND_LABEL else [raw_label]
    else:
        labels = [lab for lab in raw_label if lab != BACKGROUND_LABEL]
    # Deduplicate while preserving order
    seen = set()
    deduped: List[str] = []
    for lab in labels:
        if lab in seen:
            continue
        if lab not in LABEL_TO_ID:
            raise KeyError(f"Unknown label '{lab}' not in LABEL_TO_ID")
        seen.add(lab)
        deduped.append(lab)
    label_ids = [LABEL_TO_ID[lab] for lab in deduped]
    primary = deduped[0] if deduped else BACKGROUND_LABEL
    return deduped, label_ids, primary


def _cache_glass_row(row: pd.Series,
                     pipeline_plan: Dict[str, Dict[str, int]],
                     background_df: pd.DataFrame,
                     cache_dir: Path,
                     align_labels: Sequence[str],
                     extra_shifts: Sequence[float] | None,
                     energy_threshold: float,
                     peak_ratio_threshold: float,
                     front_peak_ratio: float,
                     rng: np.random.Generator) -> tuple[List[CacheEntry], int]:
    """Cache base and augmented windows for a glass-breaking clip."""
    windows = generate_aligned_windows(
        row,
        align_labels=list(align_labels),
        extra_shifts=list(extra_shifts) if extra_shifts else None,
        energy_threshold=energy_threshold,
        peak_ratio_threshold=peak_ratio_threshold,
        front_peak_ratio=front_peak_ratio,
    )
    clip_id = Path(row["filename"]).stem
    fold_id = int(row.get("fold_id", -1))
    labels, label_ids, primary_label = _encode_labels(row["target_label"])
    entries: List[CacheEntry] = []
    base_count = 0

    for win_idx, window in enumerate(windows):
        base_suffix = f"base_w{win_idx:02d}"
        base_path = _save_window(window, primary_label, clip_id, fold_id, base_suffix, cache_dir)
        entries.append(
            CacheEntry(
                path=str(base_path),
                labels=labels,
                label_ids=label_ids,
                label=primary_label,
                fold_id=fold_id,
                source_filename=row["filename"],
                clip_id=clip_id,
                window_id=f"w{win_idx:02d}",
                pipeline_name="base",
                augment_desc="base",
                source_type="glass_base",
            )
        )

        base_count += 1

        for pipeline_name, cfg in pipeline_plan.items():
            copies = cfg.get("copies", 1)
            for copy_idx in range(copies):
                background = None
                pipeline_ops = PIPELINE_REGISTRY[pipeline_name]
                if "mix" in pipeline_ops:
                    background = sample_background_chunk(background_df, len(window), rng)
                augmented = run_pipeline(window, pipeline_name, background=background)
                aug_suffix = f"{pipeline_name}_w{win_idx:02d}_c{copy_idx:02d}"
                aug_path = _save_window(
                    augmented.audio,
                    primary_label,
                    clip_id,
                    fold_id,
                    aug_suffix,
                    cache_dir,
                )
                entries.append(
                    CacheEntry(
                        path=str(aug_path),
                        labels=labels,
                        label_ids=label_ids,
                        label=primary_label,
                        fold_id=fold_id,
                        source_filename=row["filename"],
                        clip_id=clip_id,
                        window_id=f"w{win_idx:02d}",
                        pipeline_name=pipeline_name,
                        augment_desc=augmented.description,
                        source_type="glass_aug",
                    )
                )
    return entries, base_count


def _cache_background_row(row: pd.Series,
                          cache_dir: Path,
                          align_labels: Sequence[str],
                          energy_threshold: float) -> List[CacheEntry]:
    """Cache sliding-window mel tiles for a background clip."""
    windows = generate_aligned_windows(row, align_labels=list(align_labels), energy_threshold=energy_threshold)
    clip_id = Path(row["filename"]).stem
    fold_id = int(row.get("fold_id", -1))
    labels, label_ids, primary_label = _encode_labels(row.get("target_label"))
    entries: List[CacheEntry] = []
    for win_idx, window in enumerate(windows):
        suffix = f"base_w{win_idx:02d}"
        cache_path = _save_window(window, primary_label, clip_id, fold_id, suffix, cache_dir)
        entries.append(
            CacheEntry(
                path=str(cache_path),
                labels=labels,
                label_ids=label_ids,
                label=primary_label,
                fold_id=fold_id,
                source_filename=row["filename"],
                clip_id=clip_id,
                window_id=f"w{win_idx:02d}",
                pipeline_name="base",
                augment_desc="base",
                source_type="background_raw",
            )
        )
    return entries


def build_cache_index(dataset_df: pd.DataFrame,
                      pipeline_plan: Dict[str, Dict[str, int]] | None = None,
                      cache_dir: Path = CACHE_DIR,
                      align_labels: Sequence[str] | None = None,
                      extra_shifts: Sequence[float] | None = None,
                      energy_threshold: float = 0.2,
                      peak_ratio_threshold: float = 0.8,
                      front_peak_ratio: float = 0.5,
                      seed: int = SEED) -> pd.DataFrame:
    """Generate mel cache for entire dataset and return metadata index."""
    align_labels = align_labels or GLASS_LABELS
    pipeline_plan = pipeline_plan or GLASS_PIPELINE_PLAN
    background_df = dataset_df[dataset_df["target_label"] == BACKGROUND_LABEL]
    rng = np.random.default_rng(seed)
    entries: List[CacheEntry] = []

    background_pool: List[CacheEntry] = []
    glass_base_count = 0

    for _, row in dataset_df.iterrows():
        if row["target_label"] in align_labels:
            glass_entries, base_count = _cache_glass_row(
                row,
                pipeline_plan=pipeline_plan,
                background_df=background_df,
                cache_dir=cache_dir,
                align_labels=align_labels,
                extra_shifts=extra_shifts,
                energy_threshold=energy_threshold,
                peak_ratio_threshold=peak_ratio_threshold,
                front_peak_ratio=front_peak_ratio,
                rng=rng,
            )
            entries.extend(glass_entries)
            glass_base_count += base_count
        else:
            background_pool.extend(
                _cache_background_row(
                    row,
                    cache_dir=cache_dir,
                    align_labels=align_labels,
                    energy_threshold=energy_threshold,
                )
            )

    if background_pool:
        entries.extend(background_pool)

    index_df = pd.DataFrame([asdict(entry) for entry in entries])
    return index_df


__all__ = ["CacheEntry", "build_cache_index", "GLASS_PIPELINE_PLAN", "GLASS_LABELS", "sample_background_chunk"]
