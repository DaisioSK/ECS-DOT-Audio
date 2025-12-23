"""Reusable preprocessing helpers shared by prepare/train/infer.

This module focuses on making train/infer consistent:
- Load audio (optionally resample) with a clear notion of original SR.
- Compute log-mel features with stable parameters.
- Pad/crop mel frames and pack into BCHW layout for CNN/ONNX.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import librosa

from .config import (
    HOP_LENGTH,
    MEL_CENTER,
    MEL_TARGET_FRAMES,
    N_FFT,
    N_MELS,
    SR,
    WINDOW_HOP,
    WINDOW_SECONDS,
)


@dataclass(frozen=True)
class AudioRecord:
    """A loaded audio file, optionally resampled."""

    path: Path
    y: np.ndarray
    sr: int
    sr0: int


@dataclass
class WindowSlice:
    """One fixed-length window slice extracted from an audio file."""

    path: Path
    start_sec: float
    duration_sec: float
    wave: np.ndarray
    mel: np.ndarray


def load_audio_file(
    path: Path | str,
    target_sr: int = SR,
    mono: bool = True,
    dtype: np.dtype = np.float32,
) -> AudioRecord:
    """Load audio from disk and resample to `target_sr`.

    Notes:
    - `librosa.load(sr=target_sr)` does not expose the original sampling rate.
      We explicitly load with `sr=None` first to obtain `sr0`, then resample.
    - This is the safest way to avoid SR confusion when debugging mixed inputs
      (wav/mp3/etc).

    Returns:
        AudioRecord(path, y, sr, sr0) where:
          - `sr0` is the original sampling rate from the file
          - `sr` is the (possibly resampled) rate, usually `target_sr`
    """

    p = Path(path)
    y0, sr0 = librosa.load(p.as_posix(), sr=None, mono=mono)
    y0 = y0.astype(dtype, copy=False)
    if int(sr0) == int(target_sr):
        return AudioRecord(path=p, y=y0, sr=int(sr0), sr0=int(sr0))
    y = librosa.resample(y0, orig_sr=int(sr0), target_sr=int(target_sr))
    y = y.astype(dtype, copy=False)
    return AudioRecord(path=p, y=y, sr=int(target_sr), sr0=int(sr0))


def iter_sliding_windows(
    y: np.ndarray,
    sr: int,
    window_seconds: float = WINDOW_SECONDS,
    hop_seconds: float = WINDOW_HOP,
    pad_short: bool = True,
) -> Iterable[Tuple[np.ndarray, float]]:
    """Yield waveform windows and their start time (seconds)."""

    win = int(round(window_seconds * sr))
    hop = int(round(hop_seconds * sr))
    if win <= 0 or hop <= 0:
        raise ValueError("window_seconds and hop_seconds must be > 0")
    if len(y) < win and pad_short:
        y = np.pad(y, (0, win - len(y)))
    for start in range(0, max(0, len(y) - win) + 1, hop):
        seg = y[start : start + win]
        yield seg, start / sr


def log_mel(
    y: np.ndarray,
    sr: int,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    n_mels: int = N_MELS,
    center: bool = MEL_CENTER,
) -> np.ndarray:
    """Compute log-mel spectrogram (shape: n_mels x frames).

    This mirrors `src.data_utils.log_mel_spectrogram`, but exposes `center`
    explicitly so infer/export can match training behavior precisely.
    """

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        center=center,
    )
    return librosa.power_to_db(mel, ref=np.max)


def pad_or_crop_frames(
    mel: np.ndarray,
    target_frames: int,
    mode: str = "constant",
) -> np.ndarray:
    """Pad or crop mel frames to `target_frames` along time axis."""

    if mel.ndim != 2:
        raise ValueError(f"Expected mel with shape (n_mels, frames), got {mel.shape}")
    frames = int(mel.shape[1])
    if frames == target_frames:
        return mel
    if frames > target_frames:
        return mel[:, :target_frames]
    pad = target_frames - frames
    return np.pad(mel, ((0, 0), (0, pad)), mode=mode)


def mel_to_bchw(mel: np.ndarray) -> np.ndarray:
    """Convert (n_mels, frames) -> (1, 1, n_mels, frames) float32."""

    if mel.ndim != 2:
        raise ValueError(f"Expected mel with shape (n_mels, frames), got {mel.shape}")
    return mel.astype(np.float32, copy=False)[None, None, :, :]


def slice_audio_to_mels(
    record: AudioRecord,
    *,
    window_seconds: float = WINDOW_SECONDS,
    hop_seconds: float = WINDOW_HOP,
    target_frames: int = MEL_TARGET_FRAMES,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    n_mels: int = N_MELS,
    center: bool = MEL_CENTER,
    pad_short: bool = True,
) -> List[WindowSlice]:
    """Convert one audio file into training-consistent window slices.

    This is the main “single source of truth” entry used by infer/export.
    It mirrors the training feature stack:
      waveform -> sliding windows -> log-mel -> pad/crop to fixed frames

    Returns:
      A list of WindowSlice, each containing waveform + mel + metadata.
    """

    slices: List[WindowSlice] = []
    for seg, start_s in iter_sliding_windows(
        record.y,
        record.sr,
        window_seconds=window_seconds,
        hop_seconds=hop_seconds,
        pad_short=pad_short,
    ):
        mel = log_mel(
            seg,
            sr=record.sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            center=center,
        )
        if target_frames is not None:
            mel = pad_or_crop_frames(mel, target_frames=int(target_frames))
        slices.append(
            WindowSlice(
                path=record.path,
                start_sec=float(start_s),
                duration_sec=float(len(seg) / record.sr),
                wave=seg,
                mel=mel,
            )
        )
    return slices


def slices_to_batch(
    slices: Sequence[WindowSlice],
    *,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Pack window mels into a BCHW batch: (B, 1, n_mels, frames)."""

    if not slices:
        raise ValueError("Empty slices")
    mels = [s.mel for s in slices]
    batch = np.stack([mel_to_bchw(m)[0] for m in mels], axis=0)  # (B,1,n_mels,T)
    return batch.astype(dtype, copy=False)


def files_to_slices_and_batch(
    paths: Sequence[Path | str],
    *,
    target_sr: int = SR,
    mono: bool = True,
    window_seconds: float = WINDOW_SECONDS,
    hop_seconds: float = WINDOW_HOP,
    target_frames: int = MEL_TARGET_FRAMES,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    n_mels: int = N_MELS,
    center: bool = MEL_CENTER,
    pad_short: bool = True,
    max_files: int | None = None,
    max_slices_per_file: int | None = None,
    seed: int = 42,
) -> tuple[List[AudioRecord], List[WindowSlice], np.ndarray]:
    """Load multiple files, slice into mels, and pack into a single batch.

    This is designed for infer/export notebooks:
    - Applies training-consistent SR/mono
    - Slices each file with fixed windows
    - Optionally subsamples per-file windows for faster demo runs
    """

    files = [Path(p) for p in paths]
    if max_files is not None:
        files = files[:max_files]

    rng = np.random.default_rng(seed)

    audio_records: List[AudioRecord] = []
    all_slices: List[WindowSlice] = []
    for p in files:
        rec = load_audio_file(p, target_sr=target_sr, mono=mono)
        audio_records.append(rec)
        slices = slice_audio_to_mels(
            rec,
            window_seconds=window_seconds,
            hop_seconds=hop_seconds,
            target_frames=target_frames,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            center=center,
            pad_short=pad_short,
        )
        if max_slices_per_file is not None and len(slices) > max_slices_per_file:
            idx = sorted(rng.choice(len(slices), size=max_slices_per_file, replace=False))
            slices = [slices[i] for i in idx]
        all_slices.extend(slices)

    batch = slices_to_batch(all_slices)
    return audio_records, all_slices, batch


__all__ = [
    "AudioRecord",
    "WindowSlice",
    "load_audio_file",
    "iter_sliding_windows",
    "log_mel",
    "pad_or_crop_frames",
    "mel_to_bchw",
    "slice_audio_to_mels",
    "slices_to_batch",
    "files_to_slices_and_batch",
]
