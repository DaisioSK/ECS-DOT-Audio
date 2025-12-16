"""Composable augmentation pipelines."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np

from .augment import (
    augment_gain,
    augment_time_shift,
    augment_time_stretch,
    apply_simple_filter,
    apply_simple_reverb,
    mix_with_background,
)


# PIPELINE_REGISTRY: Dict[str, Sequence[str]] = {
#     "shift_gain": ("time_shift", "gain"),
#     "stretch_reverb": ("time_stretch", "reverb"),
#     "shift_mix": ("time_shift", "mix"),
#     "filter_gain": ("filter", "gain"),
#     "gain_mix": ("gain", "mix"),
#     "stretch_filter": ("time_stretch", "filter"),
# }

PIPELINE_REGISTRY: Dict[str, Sequence[str]] = {
    "stretch_gain": ("time_stretch", "gain"),
    "reverb_gain": ("reverb", "gain"),
    "mix_gain": ("mix", "gain"),
    "filter_gain": ("filter", "gain"),
    "mix_filter_gain": ("mix", "filter", "gain"),
    "stretch_mix_gain": ("time_stretch", "mix", "gain"),
    "stretch_filter_gain": ("time_stretch", "filter", "gain"),
    "reverb_mix_gain": ("reverb", "mix", "gain"),
    "reverb_filter_gain": ("reverb", "filter", "gain"),
    "stretch_mix_filter_gain": ("time_stretch", "mix", "filter", "gain"),
    "reverb_mix_filter_gain": ("reverb", "mix", "filter", "gain"),
}

@dataclass
class AugmentedWindow:
    """Container describing an augmented waveform."""

    audio: np.ndarray
    description: str


def apply_pipeline(y: np.ndarray,
                   pipeline: Sequence[str],
                   background: np.ndarray | None = None) -> np.ndarray:
    """Apply a sequence of primitive names to waveform y."""
    out = y
    for name in pipeline:
        if name == "time_shift":
            out = augment_time_shift(out)
        elif name == "time_stretch":
            out = augment_time_stretch(out)
        elif name == "gain":
            out = augment_gain(out)
        elif name == "mix":
            if background is None:
                raise ValueError("Background audio required for 'mix'")
            out = mix_with_background(out, background)
        elif name == "reverb":
            out = apply_simple_reverb(out)
        elif name == "filter":
            kind = random.choice(["lowpass", "highpass"])
            out = apply_simple_filter(out, kind=kind)
        else:
            raise KeyError(f"Unknown augmentation primitive: {name}")
    return out


def run_pipeline(y: np.ndarray,
                 pipeline_name: str,
                 background: np.ndarray | None = None) -> AugmentedWindow:
    """Apply a named pipeline and return the augmented window + description."""
    pipeline = PIPELINE_REGISTRY[pipeline_name]
    desc = "+".join(pipeline)
    return AugmentedWindow(
        audio=apply_pipeline(y, pipeline, background=background),
        description=f"{pipeline_name}({desc})",
    )


def choose_pipeline(name: str | None = None) -> Sequence[str]:
    """Return a pipeline definition by name or random choice."""
    if name:
        return PIPELINE_REGISTRY[name]
    return random.choice(list(PIPELINE_REGISTRY.values()))


__all__ = ["PIPELINE_REGISTRY", "AugmentedWindow", "apply_pipeline", "run_pipeline", "choose_pipeline"]
