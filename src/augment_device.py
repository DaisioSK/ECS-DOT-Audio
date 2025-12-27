"""Device/chain simulation augmentations.

These ops approximate playback+recording artifacts (EQ/RIR/codec/comp) in a
lightweight way without external assets. They are designed to be composed
with existing event-level augmentations.
"""
from __future__ import annotations

import random
from typing import Dict, Sequence

import numpy as np
from scipy import signal

from .config import SR

# -------------------------
# Primitive device effects
# -------------------------


def random_eq_tilt(y: np.ndarray, sr: int = SR, max_db: float = 6.0) -> np.ndarray:
    """Apply a gentle spectral tilt (low-shelf vs high-shelf)."""
    tilt_db = random.uniform(-max_db, max_db)
    # design first-order shelf: positive -> boost highs, negative -> boost lows
    # implement as simple FIR slope
    n = 33
    window = signal.windows.hann(n, sym=False)
    freq = np.linspace(0, 1, n)
    gain = 10 ** (tilt_db / 20.0)
    shape = np.linspace(1.0 if tilt_db < 0 else gain, gain if tilt_db < 0 else 1.0, n)
    fir = window * shape
    fir /= np.sum(np.abs(fir)) + 1e-8
    return signal.lfilter(fir, [1.0], y).astype(np.float32, copy=False)


def random_band_filter(y: np.ndarray, sr: int = SR) -> np.ndarray:
    """Random low/high/band-pass to mimic device bandwidth limits."""
    nyq = 0.5 * sr
    modes = ["low", "high", "band"]
    mode = random.choice(modes)
    if mode == "low":
        cutoff = random.uniform(2500, 8000)
        b, a = signal.butter(4, cutoff / nyq, btype="low")
    elif mode == "high":
        cutoff = random.uniform(100, 500)
        b, a = signal.butter(4, cutoff / nyq, btype="high")
    else:
        low = random.uniform(100, 800)
        high = random.uniform(2000, 6000)
        low, high = min(low, high * 0.7), max(high, low * 1.3)
        b, a = signal.butter(4, [low / nyq, high / nyq], btype="band")
    return signal.lfilter(b, a, y).astype(np.float32, copy=False)


def apply_simple_rir(y: np.ndarray, sr: int = SR) -> np.ndarray:
    """Convolve with a synthetic room impulse (short, light decay + dry/wet mix)."""
    # 收紧混响：更短 RT60、较短延迟、较少 tap，并保留 70% dry
    rt60 = random.uniform(0.08, 0.20)
    delay_ms = random.uniform(15, 50)
    taps = 4
    delay_samples = int(delay_ms * 1e-3 * sr)
    if delay_samples <= 0:
        return y
    impulse = np.zeros(delay_samples * taps, dtype=np.float32)
    for i in range(taps):
        impulse[i * delay_samples] = np.exp(-i * delay_ms / (rt60 * 1000 + 1e-6))
    wet = signal.fftconvolve(y, impulse)[: len(y)]
    wet_mix = 0.3  # 30% wet, 70% dry
    out = (1.0 - wet_mix) * y + wet_mix * wet
    return out.astype(np.float32, copy=False)


def codec_like_resample(y: np.ndarray, sr: int = SR) -> np.ndarray:
    """Down/upsample to mimic low-bitrate codec bandlimit."""
    low_sr = random.choice([8000, 12000, 16000])
    if sr <= low_sr:
        return y
    y_down = signal.resample_poly(y, up=low_sr, down=sr)
    y_up = signal.resample_poly(y_down, up=sr, down=low_sr)
    if len(y_up) < len(y):
        y_up = np.pad(y_up, (0, len(y) - len(y_up)))
    return y_up[: len(y)].astype(np.float32, copy=False)


def soft_compress(y: np.ndarray, drive_db: float = 6.0, mix: float = 0.5) -> np.ndarray:
    """Soft clipping compression with dry/wet mix."""
    drive = 10 ** (drive_db / 20.0)
    driven = np.tanh(y * drive)
    out = (1 - mix) * y + mix * driven
    return out.astype(np.float32, copy=False)


# -------------------------
# Registry + runner
# -------------------------

DEVICE_PIPELINE: Dict[str, Sequence[str]] = {
    "eq": ("random_eq_tilt",),
    "band": ("random_band_filter",),
    "rir": ("apply_simple_rir",),
    "codec": ("codec_like_resample",),
    "compress": ("soft_compress",),
    # 复合示例：低码率+压缩
    "codec_compress": ("codec_like_resample", "soft_compress"),
}


def run_device_pipeline(y: np.ndarray, name: str, sr: int = SR) -> np.ndarray:
    """Apply a device pipeline by name."""
    steps = DEVICE_PIPELINE.get(name)
    if not steps:
        return y
    out = y
    for step in steps:
        if step == "random_eq_tilt":
            out = random_eq_tilt(out, sr=sr)
        elif step == "random_band_filter":
            out = random_band_filter(out, sr=sr)
        elif step == "apply_simple_rir":
            out = apply_simple_rir(out, sr=sr)
        elif step == "codec_like_resample":
            out = codec_like_resample(out, sr=sr)
        elif step == "soft_compress":
            out = soft_compress(out)
    return out


__all__ = [
    "DEVICE_PIPELINE",
    "run_device_pipeline",
    "random_eq_tilt",
    "random_band_filter",
    "apply_simple_rir",
    "codec_like_resample",
    "soft_compress",
]
