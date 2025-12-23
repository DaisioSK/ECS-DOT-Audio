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
    MEL_CENTER,
    N_FFT,
    N_MELS,
    POSITIVE_LABELS,
    BACKGROUND_LABEL,
    BACKGROUND_MULTIPLIER,
    SR,
    WINDOW_HOP,
    WINDOW_SECONDS,
    WINDOW_PARAMS,
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
    fp = row.get('filepath')
    if isinstance(fp, (str, Path)) and fp:
        p = Path(fp)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        return p
    # Fallback: raw_filepath (already relative to project), then filename
    raw_fp = row.get('raw_filepath')
    if isinstance(raw_fp, (str, Path)) and raw_fp:
        p = Path(raw_fp)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        return p
    fname = row.get('filename')
    if isinstance(fname, str) and fname:
        return AUDIO_DIR / fname
    raise KeyError("Missing audio path: expected one of ['filepath', 'raw_filepath', 'filename']")


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
                        n_mels: int = N_MELS,
                        center: bool = MEL_CENTER) -> np.ndarray:
    """Compute a log-mel spectrogram tile (shape: n_mels x frames)."""
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft,
                                         hop_length=hop_length, n_mels=n_mels,
                                         center=center)
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


def generate_aligned_windows_legacy(row: pd.Series,
                                    align_labels: List[str],
                                    extra_shifts: List[float] | None = None,
                                    energy_threshold: float = 0.2,
                                    peak_ratio_threshold: float = 0.7,
                                    front_peak_ratio: float = 0.5,
                                    late_peak_keep_ratio: float = 0.8,
                                    trim_silence_before: bool = False,
                                    trim_top_db: float = 20.0,
                                    trim_min_keep_seconds: float = 0.0,
                                    debug: bool = False,
                                    debug_sink: list | None = None) -> List[np.ndarray]:
    """Legacy window generator with inline trim/energy logic."""
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
            front_half_peak = float(np.max(energy[: max(1, len(energy) // 2)]))
            front_half_ratio = front_half_peak / energy_global
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
                if front_half_ratio >= late_peak_keep_ratio:
                    windows.append(window)
                    if debug_sink is not None:
                        debug_sink[-1]["status"] = "keep"
                        debug_sink[-1]["reason"] = "late_peak_keep_front_energy"
                    continue
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


def generate_aligned_windows(row: pd.Series,
                             align_labels: List[str],
                             extra_shifts: List[float] | None = None,
                             energy_threshold: float = 0.2,
                             peak_ratio_threshold: float = 0.7,
                             front_peak_ratio: float = 0.5,
                             late_peak_keep_ratio: float = 0.8,
                             trim_silence_before: bool = False,
                             trim_top_db: float = 20.0,
                             trim_min_keep_seconds: float = 0.0,
                             debug: bool = False,
                             label_params: Dict[str, Dict[str, float]] | None = None,
                             debug_sink: list | None = None) -> List[np.ndarray]:
    """Generate windows aligned to label semantics with full debug logging.

    Logs every candidate (and trimmed regions) to debug_sink with keys:
    start_sec/end_sec (relative to raw audio), status, reason, peak_ratio, peak_position.
    """
    label = row['target_label']

    # Per-label overrides: first from call site, else from config WINDOW_PARAMS.
    label_cfg = label_params or WINDOW_PARAMS

    def get_param(name: str, default: float):
        if label in label_cfg and name in label_cfg[label]:
            return label_cfg[label][name]
        return default

    energy_thr = get_param("energy_threshold", energy_threshold)
    peak_ratio_thr = get_param("peak_ratio_threshold", peak_ratio_threshold)
    front_peak_thr = get_param("front_peak_ratio", front_peak_ratio)
    late_keep_thr = get_param("late_peak_keep_ratio", late_peak_keep_ratio)
    shifts = [0.0]
    if extra_shifts:
        shifts.extend(extra_shifts)
    if label in label_cfg and "extra_shifts" in label_cfg[label]:
        # Override completely if provided per label
        shifts = label_cfg[label]["extra_shifts"]
        if isinstance(shifts, tuple):
            shifts = list(shifts)

    y_raw, sr = load_audio(row)
    orig_len = len(y_raw)

    # Head/tail trim only; keep mapping to original timeline.
    lead_trim = 0
    tail_trim = 0
    if trim_silence_before:
        intervals = librosa.effects.split(y_raw, top_db=trim_top_db)
        min_keep = int(trim_min_keep_seconds * sr)
        kept = [(s, e) for (s, e) in intervals if (e - s) >= min_keep]
        if kept:
            lead_trim = kept[0][0]
            tail_trim = max(0, orig_len - kept[-1][1])
    y_trim = y_raw[lead_trim:orig_len - tail_trim] if (lead_trim or tail_trim) else y_raw

    window_len = int(WINDOW_SECONDS * sr)
    hop_len = int(WINDOW_HOP * sr)

    def log_entry(start_sec, end_sec, status, reason, peak_ratio=None, peak_position=None):
        if debug_sink is None:
            return
        debug_sink.append({
            "start_sec": start_sec,
            "end_sec": end_sec,
            "peak_ratio": None if peak_ratio is None else float(peak_ratio),
            "peak_position": None if peak_position is None else float(peak_position),
            "status": status,
            "reason": reason,
        })

    # Log overall info and trimmed regions.
    log_entry(None, None, "info",
              f"len_raw={orig_len/sr:.3f}s len_trim={len(y_trim)/sr:.3f}s lead_trim={lead_trim/sr:.3f}s tail_trim={tail_trim/sr:.3f}s window={WINDOW_SECONDS}s hop={WINDOW_HOP}s")
    if lead_trim > 0:
        log_entry(0.0, lead_trim / sr, "remove", "silent_trim_head")
    if tail_trim > 0:
        log_entry((orig_len - tail_trim) / sr, orig_len / sr, "remove", "silent_trim_tail")

    windows: List[np.ndarray] = []

    if label in align_labels:
        energy_global = float(np.max(y_trim ** 2) + 1e-8)
        for idx, (win, start_trim) in enumerate(_iter_windows(y_trim, sr, WINDOW_SECONDS, WINDOW_HOP)):
            start_sec = (lead_trim + start_trim) / sr
            end_sec = start_sec + WINDOW_SECONDS
            energy = win ** 2
            peak_idx = int(np.argmax(energy))
            peak_ratio = energy[peak_idx] / energy_global
            peak_position = peak_idx / max(len(win), 1)
            front_half_peak = float(np.max(energy[: max(1, len(energy) // 2)]))
            front_half_ratio = front_half_peak / energy_global

            status = "keep"
            reason = "pass"
            if peak_ratio < peak_ratio_thr:
                status = "skip"
                reason = "low_peak_ratio"
            elif peak_position > front_peak_thr:
                if front_half_ratio >= late_keep_thr:
                    status = "keep"
                    reason = "late_peak_keep_front_energy"
                else:
                    status = "skip"
                    reason = "late_peak"

            log_entry(start_sec, end_sec, status, reason, peak_ratio, peak_position)
            if status == "keep":
                windows.append(win)

        if not windows:
            for shift in shifts:
                win = center_peak_window(y_trim, sr, shift_seconds=shift)
                log_entry(None, None, "keep", f"fallback_center_shift_{shift}")
                windows.append(win)
                break
    else:
        mask = _energy_mask(y_trim, sr, WINDOW_SECONDS, WINDOW_HOP, energy_thr)
        for idx, (win, start_trim) in enumerate(_iter_windows(y_trim, sr, WINDOW_SECONDS, WINDOW_HOP)):
            start_sec = (lead_trim + start_trim) / sr
            end_sec = start_sec + WINDOW_SECONDS
            status = "keep"
            reason = "pass"
            if idx < len(mask) and not mask[idx]:
                status = "skip"
                reason = "mask_below_threshold"
            log_entry(start_sec, end_sec, status, reason, None, None)
            if status == "keep":
                windows.append(win)
        if not windows:
            win = center_peak_window(y_trim, sr)
            log_entry(None, None, "keep", "fallback_bg_center")
            windows.append(win)

    return windows


__all__ = [
    "build_dataset",
    "audio_path",
    "load_audio",
    "log_mel_spectrogram",
    "sliding_windows",
    "center_peak_window",
    "generate_aligned_windows_legacy",
    "generate_aligned_windows",
]
