"""Audio augmentation primitives."""
from __future__ import annotations

import random
from typing import Tuple

import numpy as np
import librosa
from scipy import signal

from .config import SR


def _sample_normal_with_clip(mu: float, sigma: float, clip_range: Tuple[float, float]) -> float:
    """Sample from N(mu, sigma) then clip to clip_range."""
    val = np.random.normal(mu, sigma)
    return float(np.clip(val, clip_range[0], clip_range[1]))


def augment_time_shift(
    y: np.ndarray,
    sr: int = SR,
    max_shift: float = 0.06,
    sigma: float = 0.02,
    zero_prob: float = 0.5,
) -> np.ndarray:
    """
    Shift waveform within +/- max_shift seconds.

    - shift_seconds ~ N(0, sigma) clipped to [-max_shift, max_shift]
    - with probability zero_prob, shift is forced to 0 (保留一部分原位)
    - still防止将峰值移出窗口
    """
    length = len(y)
    if length == 0:
        return y

    if random.random() < zero_prob:
        return y

    peak_idx = int(np.argmax(np.abs(y)))
    shift_seconds = _sample_normal_with_clip(0.0, sigma, (-max_shift, max_shift))
    shift_samples = int(shift_seconds * sr)

    # 保证高能峰不过界
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


def augment_gain(
    y: np.ndarray,
    db_range: Tuple[float, float] = (-4.0, 4.0),
    sigma_db: float = 2.0,
    zero_prob: float = 0.3,
) -> np.ndarray:
    """
    Apply random gain (dB).

    - gain_db ~ N(0, sigma_db) clipped to db_range
    - with probability zero_prob, gain_db=0
    """
    if random.random() < zero_prob:
        gain_db = 0.0
    else:
        gain_db = _sample_normal_with_clip(0.0, sigma_db, db_range)
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
    """Add a short, randomized multi-tap echo to mimic room reverb."""
    # Randomize decay/delay with safe bounds to avoid over-blurring.
    delay_ms = int(np.clip(delay_ms, 20, 70))  # clamp default in case caller overrides
    delay_ms = random.uniform(20.0, 70.0)
    decay = random.uniform(0.2, 0.5)
    taps = 4
    delay_samples = int(delay_ms * 1e-3 * SR)
    if delay_samples <= 0:
        return y
    impulse = np.zeros(delay_samples * taps)
    for i in range(taps):
        impulse[i * delay_samples] = decay ** i
    reverbed = signal.fftconvolve(y, impulse)[:len(y)]
    # Keep wet mix modest to avoid drowning transients.
    wet_mix = random.uniform(0.25, 0.35)
    return ((1.0 - wet_mix) * y + wet_mix * reverbed).astype(np.float32, copy=False)


def apply_simple_filter(y: np.ndarray, cutoff: float = 4000.0, sr: int = SR, kind: str = 'lowpass') -> np.ndarray:
    """Apply a randomized 4th-order Butterworth filter (low/high/band-pass)."""
    nyq = 0.5 * sr

    # When kind is not explicitly low/high, choose a mode with safer randomized cutoffs.
    mode = kind
    if kind not in ('lowpass', 'highpass'):
        mode = random.choice(['lowpass', 'highpass', 'bandpass'])

    if mode == 'lowpass':
        cutoff = random.uniform(2500.0, 8000.0)
        norm = cutoff / nyq
        b, a = signal.butter(4, norm, btype='low')
    elif mode == 'highpass':
        cutoff = random.uniform(150.0, 800.0)
        norm = cutoff / nyq
        b, a = signal.butter(4, norm, btype='high')
    else:  # bandpass
        low = random.uniform(150.0, 800.0)
        high = random.uniform(2500.0, 6500.0)
        # enforce a minimum bandwidth and valid ordering
        if high - low < 800.0:
            high = low + 800.0
        high = min(high, nyq * 0.95)
        low = max(low, 60.0)
        b, a = signal.butter(4, [low / nyq, high / nyq], btype='band')

    return signal.lfilter(b, a, y).astype(np.float32, copy=False)


__all__ = [
    "augment_time_shift",
    "augment_time_stretch",
    "augment_gain",
    "mix_with_background",
    "apply_simple_reverb",
    "apply_simple_filter",
]
