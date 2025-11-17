"""Data loading and windowing helpers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import librosa

from .config import (
    AUDIO_DIR,
    HOP_LENGTH,
    META_FILE,
    N_FFT,
    N_MELS,
    POSITIVE_LABELS,
    BACKGROUND_LABEL,
    BACKGROUND_MULTIPLIER,
    SR,
    WINDOW_HOP,
    WINDOW_SECONDS,
)


def build_dataset(meta_df: pd.DataFrame,
                  positive_map: Dict[str, str] | None = None,
                  background_label: str = BACKGROUND_LABEL,
                  background_multiplier: int = BACKGROUND_MULTIPLIER,
                  seed: int = 42) -> pd.DataFrame:
    """Return a trimmed metadata table with glass vs. background labels.

    Args:
        meta_df: Full ESC-50 metadata dataframe.
        positive_map: Mapping from original category to target label string.
        background_label: Label name for non-positive categories.
        background_multiplier: Max ratio of background clips to positives.
        seed: RNG seed used when sampling background clips.
    """
    positive_map = positive_map or POSITIVE_LABELS
    pos_mask = meta_df['category'].isin(positive_map.keys())
    pos_df = meta_df[pos_mask].copy()
    pos_df['target_label'] = pos_df['category'].map(positive_map)

    bg_df = meta_df[~pos_mask].copy()
    n_bg = min(len(bg_df), max(background_multiplier * len(pos_df), len(pos_df)))
    bg_sample = bg_df.sample(n=n_bg, random_state=seed)
    bg_sample['target_label'] = background_label

    dataset_df = pd.concat([pos_df, bg_sample], ignore_index=True)
    return dataset_df


def audio_path(row: pd.Series) -> Path:
    """Resolve on-disk path for the audio file described by a metadata row."""
    return AUDIO_DIR / row['filename']


def load_audio(row: pd.Series, sr: int = SR) -> Tuple[np.ndarray, int]:
    """Load an audio clip as mono float32 at the requested sampling rate."""
    path = audio_path(row)
    y, sr_out = librosa.load(path, sr=sr)
    return y, sr_out


def log_mel_spectrogram(y: np.ndarray,
                        sr: int,
                        n_fft: int = N_FFT,
                        hop_length: int = HOP_LENGTH,
                        n_mels: int = N_MELS) -> np.ndarray:
    """Compute a log-mel spectrogram tile (shape: n_mels x frames)."""
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft,
                                         hop_length=hop_length, n_mels=n_mels)
    return librosa.power_to_db(mel, ref=np.max)


def sliding_windows(y: np.ndarray, sr: int,
                    window_seconds: float = WINDOW_SECONDS,
                    hop_seconds: float = WINDOW_HOP) -> Iterable[np.ndarray]:
    """Yield fixed-length waveform segments using sliding-window slicing."""
    window_len = int(window_seconds * sr)
    hop_len = int(hop_seconds * sr)
    if len(y) < window_len:
        y = np.pad(y, (0, window_len - len(y)))
    for start in range(0, len(y) - window_len + 1, hop_len):
        yield y[start:start + window_len]


def center_peak_window(y: np.ndarray,
                        sr: int,
                        window_seconds: float = WINDOW_SECONDS,
                        hop: int = HOP_LENGTH,
                        shift_seconds: float = 0.0) -> np.ndarray:
    """Return a window centered around the strongest RMS peak (plus shift)."""
    window_len = int(window_seconds * sr)
    if len(y) < window_len:
        y = np.pad(y, (0, window_len - len(y)))
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    peak_frame = int(np.argmax(rms))
    peak_sample = peak_frame * hop
    center = peak_sample + int(shift_seconds * sr)
    start = max(0, center - window_len // 2)
    start = min(start, len(y) - window_len)
    window = y[start:start + window_len]
    if len(window) < window_len:
        window = np.pad(window, (0, window_len - len(window)))
    return window


def generate_aligned_windows(row: pd.Series,
                             align_labels: List[str],
                             extra_shifts: List[float] | None = None) -> List[np.ndarray]:
    """Produce waveform windows for a row, aligning positives to peak."""
    y, sr = load_audio(row)
    label = row['target_label']
    windows: List[np.ndarray] = []
    if label in align_labels:
        shifts = [0.0]
        if extra_shifts:
            shifts.extend(extra_shifts)
        for shift in shifts:
            windows.append(center_peak_window(y, sr, shift_seconds=shift))
    else:
        windows.extend(list(sliding_windows(y, sr)))
    return windows


__all__ = [
    "build_dataset",
    "audio_path",
    "load_audio",
    "log_mel_spectrogram",
    "sliding_windows",
    "center_peak_window",
    "generate_aligned_windows",
]
