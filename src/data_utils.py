"""Data loading and windowing helpers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import librosa

from .config import (
    AUDIO_DIR,
    PROJECT_ROOT,
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
    # Prefer explicit filepath if present
    if 'filepath' in row and pd.notna(row['filepath']):
        p = Path(row['filepath'])
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        return p
    return AUDIO_DIR / row['filename']


def load_audio(row: pd.Series, sr: int = SR) -> Tuple[np.ndarray, int]:
    """Load an audio clip resampled to `sr` and downmixed to mono float32."""
    path = audio_path(row)
    y, sr_out = librosa.load(path, sr=sr, mono=True)
    return y, sr_out


def trim_silence(y: np.ndarray,
                 sr: int,
                 top_db: float = 20.0,
                 min_keep_seconds: float = 0.0) -> np.ndarray:
    """Remove silent intervals below threshold; drop very short kept chunks."""
    intervals = librosa.effects.split(y, top_db=top_db)
    if intervals.size == 0:
        return y
    keep = []
    min_keep = int(min_keep_seconds * sr)
    for start, end in intervals:
        if end - start < min_keep:
            continue
        keep.append(y[start:end])
    if not keep:
        return y
    return np.concatenate(keep)


def log_mel_spectrogram(y: np.ndarray,
                        sr: int,
                        n_fft: int = N_FFT,
                        hop_length: int = HOP_LENGTH,
                        n_mels: int = N_MELS) -> np.ndarray:
    """Compute a log-mel spectrogram tile (shape: n_mels x frames)."""
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft,
                                         hop_length=hop_length, n_mels=n_mels)
    return librosa.power_to_db(mel, ref=np.max)


def _iter_windows(y: np.ndarray,
                  sr: int,
                  window_seconds: float = WINDOW_SECONDS,
                  hop_seconds: float = WINDOW_HOP) -> Iterable[Tuple[np.ndarray, int]]:
    """Yield fixed-length waveform segments using sliding-window slicing."""
    window_len = int(window_seconds * sr)
    hop_len = int(hop_seconds * sr)
    if len(y) < window_len:
        y = np.pad(y, (0, window_len - len(y)))
    for start in range(0, len(y) - window_len + 1, hop_len):
        yield y[start:start + window_len], start


def sliding_windows(y: np.ndarray, sr: int,
                    window_seconds: float = WINDOW_SECONDS,
                    hop_seconds: float = WINDOW_HOP) -> Iterable[np.ndarray]:
    """Public window generator (without returning offsets)."""
    for window, _ in _iter_windows(y, sr, window_seconds, hop_seconds):
        yield window


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


def _energy_mask(y: np.ndarray,
                 sr: int,
                 window_seconds: float,
                 hop_seconds: float,
                 threshold_ratio: float) -> np.ndarray:
    """Return boolean mask per window indicating if average energy exceeds threshold."""
    hop_len = int(hop_seconds * sr)
    frame_length = min(int(window_seconds * sr), 2048)
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_len)[0]
    if rms.size == 0 or rms.max() <= 0:
        return np.ones(rms.shape, dtype=bool)
    energy = rms / rms.max()
    return energy > threshold_ratio


def generate_aligned_windows(row: pd.Series,
                             align_labels: List[str],
                             extra_shifts: List[float] | None = None,
                             energy_threshold: float = 0.2,
                             peak_ratio_threshold: float = 0.7,
                             front_peak_ratio: float = 0.5,
                             trim_silence_before: bool = False,
                             trim_top_db: float = 20.0,
                             trim_min_keep_seconds: float = 0.0,
                             debug: bool = False,
                             debug_sink: list | None = None) -> List[np.ndarray]:
    """Produce waveform windows for a row with energy filtering."""
    y_raw, sr = load_audio(row)
    orig_len = len(y_raw)
    # Compute lead/tail trim via silence split, but keep raw waveform for window grid.
    lead_trim = 0
    tail_trim = 0
    if trim_silence_before:
        intervals = librosa.effects.split(y_raw, top_db=trim_top_db)
        min_keep = int(trim_min_keep_seconds * sr)
        kept = [(s, e) for (s, e) in intervals if (e - s) >= min_keep]
        if kept:
            lead_trim = kept[0][0]
            tail_trim = max(0, orig_len - kept[-1][1])
    # Use trimmed slice for energy eval; keep raw for window timing
    y_trim = y_raw[lead_trim:orig_len - tail_trim] if (lead_trim or tail_trim) else y_raw
    window_len = int(WINDOW_SECONDS * sr)
    hop_len = int(WINDOW_HOP * sr)
    n_windows_raw = max(1, int(np.floor(max(orig_len - window_len, 0) / max(hop_len, 1))) + 1)
    n_windows_trim = max(1, int(np.floor(max(len(y_trim) - window_len, 0) / max(hop_len, 1))) + 1)
    if debug_sink is not None:
        debug_sink.append({
            "start_sec": None,
            "end_sec": None,
            "peak_ratio": None,
            "peak_position": None,
            "status": "info",
            "reason": f"windows_raw={n_windows_raw} windows_trim={n_windows_trim} len_raw={orig_len/sr:.2f}s len_trim={len(y_trim)/sr:.2f}s lead_trim={lead_trim/sr:.2f}s tail_trim={tail_trim/sr:.2f}s",
        })
    label = row['target_label']
    windows: List[np.ndarray] = []
    if label in align_labels:
        energy_global = np.max(y_trim ** 2) + 1e-8
        for window_raw, start in _iter_windows(y_raw, sr):
            start_sec = start / sr
            end = start + window_len
            end_sec = end / sr
            # Skip windows fully in trimmed-out regions
            if (start < lead_trim and end <= lead_trim) or (start >= orig_len - tail_trim):
                if debug_sink is not None:
                    debug_sink.append({
                        "start_sec": start_sec,
                        "end_sec": end_sec,
                        "peak_ratio": None,
                        "peak_position": None,
                        "status": "skip",
                        "reason": "silent_trim",
                    })
                if debug:
                    print(f"[SKIP silent trim] t={start_sec:.2f}s")
                continue
            # Map to trimmed waveform slice for evaluation
            start_trim = max(0, start - lead_trim)
            end_trim = start_trim + window_len
            if end_trim > len(y_trim):
                # partial coverage after trim; skip
                if debug_sink is not None:
                    debug_sink.append({
                        "start_sec": start_sec,
                        "end_sec": end_sec,
                        "peak_ratio": None,
                        "peak_position": None,
                        "status": "skip",
                        "reason": "partial_trim_overlap",
                    })
                if debug:
                    print(f"[SKIP partial overlap] t={start_sec:.2f}s")
                continue
            window = y_trim[start_trim:end_trim]
            energy = window ** 2
            peak_idx = int(np.argmax(energy))
            peak_ratio = energy[peak_idx] / energy_global
            peak_position = peak_idx / max(len(window), 1)
            if debug_sink is not None:
                debug_sink.append({
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "peak_ratio": float(peak_ratio),
                    "peak_position": float(peak_position),
                    "status": "pending",
                    "reason": "",
                })
            if peak_ratio < peak_ratio_threshold:
                if debug:
                    print(f"[SKIP low peak] t={start_sec:.2f}s peak_ratio={peak_ratio:.3f} pos={peak_position:.3f}")
                if debug_sink is not None:
                    debug_sink[-1]["status"] = "skip"
                    debug_sink[-1]["reason"] = "low_peak_ratio"
                continue
            if peak_position > front_peak_ratio:
                if debug_sink is not None:
                    debug_sink[-1]["status"] = "skip"
                    debug_sink[-1]["reason"] = "late_peak"
                continue
            windows.append(window)
            if debug_sink is not None:
                debug_sink[-1]["status"] = "keep"
                debug_sink[-1]["reason"] = "pass"
        if not windows:
            shifts = [0.0]
            if extra_shifts:
                shifts.extend(extra_shifts)
            for shift in shifts:
                windows.append(center_peak_window(y_trim, sr, shift_seconds=shift))
                if debug_sink is not None:
                    debug_sink.append({
                        "start_sec": None,
                        "peak_ratio": None,
                        "peak_position": None,
                        "status": "keep",
                        "reason": f"fallback_shift_{shift}",
                    })
                break
    else:
        mask = _energy_mask(y_trim, sr, WINDOW_SECONDS, WINDOW_HOP, energy_threshold)
        for idx, (window_raw, start) in enumerate(_iter_windows(y_raw, sr)):
            if debug_sink is not None:
                debug_sink.append({
                    "start_sec": start / sr,
                    "end_sec": (start + window_len) / sr,
                    "peak_ratio": None,
                    "peak_position": None,
                    "status": "pending",
                    "reason": "",
                })
            if idx < len(mask) and not mask[idx]:
                if debug_sink is not None:
                    debug_sink[-1]["status"] = "skip"
                    debug_sink[-1]["reason"] = "mask_false"
                continue
            # Skip windows fully in trimmed regions
            if (start < lead_trim and (start + window_len) <= lead_trim) or (start >= orig_len - tail_trim):
                if debug_sink is not None:
                    debug_sink[-1]["status"] = "skip"
                    debug_sink[-1]["reason"] = "silent_trim"
                continue
            # Map to trimmed portion for energy eval
            start_trim = max(0, start - lead_trim)
            end_trim = start_trim + window_len
            if end_trim > len(y_trim):
                if debug_sink is not None:
                    debug_sink[-1]["status"] = "skip"
                    debug_sink[-1]["reason"] = "partial_trim_overlap"
                continue
            window = y_trim[start_trim:end_trim]
            windows.append(window)
            if debug_sink is not None:
                debug_sink[-1]["status"] = "keep"
                debug_sink[-1]["reason"] = "pass"
        if not windows:
            windows.append(center_peak_window(y_trim, sr))
            if debug_sink is not None:
                debug_sink.append({
                    "start_sec": None,
                    "peak_ratio": None,
                    "peak_position": None,
                    "status": "keep",
                    "reason": "fallback_bg_center",
                })
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
