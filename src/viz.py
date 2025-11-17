"""Visualization utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa.display

from .config import HOP_LENGTH
from .data_utils import load_audio, log_mel_spectrogram


def plot_wave_and_mel(row: pd.Series) -> None:
    """Plot waveform and log-mel spectrogram for a sample row."""
    y, sr = load_audio(row)
    mel = log_mel_spectrogram(y, sr)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=False)
    times = np.arange(len(y)) / sr
    axes[0].plot(times, y)
    axes[0].set_title(f"Waveform | {row['target_label']} | {row['filename']}")
    axes[0].set_xlabel('Time [s]')
    axes[0].set_ylabel('Amplitude')

    librosa.display.specshow(mel, sr=sr, hop_length=HOP_LENGTH,
                             x_axis='time', y_axis='mel', ax=axes[1])
    axes[1].set_title('Log-Mel Spectrogram (dB)')
    fig.tight_layout()
    plt.show()


__all__ = ["plot_wave_and_mel"]
