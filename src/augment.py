"""Audio augmentation primitives."""
from __future__ import annotations

import random
from typing import Tuple

import numpy as np
import librosa
from scipy import signal

from .config import SR


def augment_time_shift(y: np.ndarray, sr: int = SR, max_shift: float = 0.15) -> np.ndarray:
    """Shift waveform within +/- max_shift seconds without wrapping the peak outside the window."""
    length = len(y)
    if length == 0:
        return y
    peak_idx = int(np.argmax(np.abs(y)))
    shift_samples = int(random.uniform(-max_shift, max_shift) * sr)
    margin = max(1, int(0.1 * length))
    max_left = peak_idx - margin
    max_right = length - margin - peak_idx
    shift_samples = max(-max_left, min(shift_samples, max_right))
    if shift_samples == 0:
        return y
    shifted = np.zeros_like(y)
    if shift_samples > 0:
        shifted[shift_samples:] = y[:-shift_samples]
    else:
        shifted[:shift_samples] = y[-shift_samples:]
    return shifted


def augment_time_stretch(y: np.ndarray, rate_range: Tuple[float, float] = (0.95, 1.05)) -> np.ndarray:
    """Stretch/compress waveform in time, padding/truncating to original length."""
    rate = random.uniform(*rate_range)
    stretched = librosa.effects.time_stretch(y=y, rate=rate)
    if len(stretched) >= len(y):
        return stretched[:len(y)]
    return np.pad(stretched, (0, len(y) - len(stretched)))


def augment_gain(y: np.ndarray, db_range: Tuple[float, float] = (-5.0, 5.0)) -> np.ndarray:
    """Apply random gain (dB) to simulate different loudness levels."""
    gain_db = random.uniform(*db_range)
    gain = librosa.db_to_amplitude(gain_db)
    return y * gain


def mix_with_background(y: np.ndarray,
                        background: np.ndarray,
                        snr_db_range: Tuple[float, float] = (3.0, 9.0),
                        bg_max_ratio: float = 0.1) -> np.ndarray:
    """Blend background audio into the signal using a random target SNR with a cap on background energy."""
    if len(background) < len(y):
        repeat = int(np.ceil(len(y) / len(background)))
        background = np.tile(background, repeat)
    background = background[:len(y)]
    signal_power = np.mean(y ** 2) + 1e-8
    noise_power = np.mean(background ** 2) + 1e-8
    max_noise_power = signal_power * bg_max_ratio
    if noise_power > max_noise_power:
        scale = np.sqrt(max_noise_power / noise_power)
        background = background * scale
        noise_power = np.mean(background ** 2) + 1e-8
    snr_db = random.uniform(*snr_db_range)
    scale = np.sqrt(signal_power / (noise_power * 10 ** (snr_db / 10)))
    return y + background * scale


def apply_simple_reverb(y: np.ndarray, decay: float = 0.3, delay_ms: int = 50) -> np.ndarray:
    """Add a basic multi-tap echo tail to mimic room reverb."""
    delay_samples = int(delay_ms * 1e-3 * SR)
    impulse = np.zeros(delay_samples * 4)
    for i in range(4):
        impulse[i * delay_samples] = decay ** i
    reverbed = signal.fftconvolve(y, impulse)[:len(y)]
    return reverbed


def apply_simple_filter(y: np.ndarray, cutoff: float = 4000.0, sr: int = SR, kind: str = 'lowpass') -> np.ndarray:
    """Apply a 4th-order Butterworth low/high-pass filter."""
    nyq = 0.5 * sr
    norm_cutoff = cutoff / nyq
    b, a = signal.butter(4, norm_cutoff, btype='low' if kind == 'lowpass' else 'high')
    return signal.lfilter(b, a, y)


__all__ = [
    "augment_time_shift",
    "augment_time_stretch",
    "augment_gain",
    "mix_with_background",
    "apply_simple_reverb",
    "apply_simple_filter",
]
