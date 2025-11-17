"""Helpers for reconstructing audio snippets from cached mel tiles."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import librosa

from .config import SR, N_FFT, HOP_LENGTH


def mel_db_to_waveform(mel_db: np.ndarray,
                       sr: int = SR,
                       n_fft: int = N_FFT,
                       hop_length: int = HOP_LENGTH,
                       n_iter: int = 32) -> np.ndarray:
    """Reconstruct waveform approximation from a log-mel spectrogram (dB)."""
    mel_power = librosa.db_to_power(mel_db)
    audio = librosa.feature.inverse.mel_to_audio(
        mel_power,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_iter=n_iter,
    )
    return audio


def export_mel_to_wav(mel_path: Path,
                      out_path: Path,
                      sr: int = SR,
                      n_fft: int = N_FFT,
                      hop_length: int = HOP_LENGTH,
                      n_iter: int = 32) -> Path:
    """Convert a saved log-mel .npy tile back to a wav file for QA listening."""
    mel_db = np.load(mel_path)
    audio = mel_db_to_waveform(mel_db, sr=sr, n_fft=n_fft, hop_length=hop_length, n_iter=n_iter)
    sf.write(out_path, audio, sr)
    return out_path


__all__ = ["mel_db_to_waveform", "export_mel_to_wav"]
