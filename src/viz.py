"""Visualization utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa.display
from pathlib import Path

from .config import HOP_LENGTH
from .data_utils import load_audio, log_mel_spectrogram


def plot_wave_and_mel(row: pd.Series | None = None,
                      y: np.ndarray | None = None,
                      sr: int | None = None,
                      title: str | None = None) -> None:
    """Plot waveform and log-mel spectrogram for a sample.

    Accepts either a metadata row (will load audio) or raw waveform+sr.
    """
    # Allow calling as plot_wave_and_mel(y, sr, ...) by detecting ndarray
    if isinstance(row, np.ndarray):
        y, sr, row = row, y, None

    # Accept row or (y, sr); if row provided, load audio if needed.
    if row is not None and (y is None or sr is None):
        y, sr = load_audio(row)
        if title is None:
            fname = row.get("filepath") or row.get("filename", "")
            title = f"{row.get('canonical_label', row.get('target_label', ''))} | {row.get('source', '')} | {Path(str(fname)).name}"
    if y is None or sr is None:
        raise ValueError("Provide either row or (y, sr).")

    mel = log_mel_spectrogram(y, sr)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=False)
    times = np.arange(len(y)) / sr
    axes[0].plot(times, y)
    axes[0].set_title(title or "Waveform")
    axes[0].set_xlabel('Time [s]')
    axes[0].set_ylabel('Amplitude')

    librosa.display.specshow(mel, sr=sr, hop_length=HOP_LENGTH,
                             x_axis='time', y_axis='mel', ax=axes[1])
    axes[1].set_title('Log-Mel Spectrogram (dB)')
    fig.tight_layout()
    plt.show()


__all__ = ["plot_wave_and_mel"]
