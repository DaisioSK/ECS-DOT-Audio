"""Project-wide constants and path helpers."""
from __future__ import annotations

from pathlib import Path


def locate_project_root(start: Path | None = None, marker: str = "data") -> Path:
    """Walk up directories until a marker folder is found."""
    start = (start or Path.cwd()).resolve()
    for path in (start,) + tuple(start.parents):
        if (path / marker).exists():
            return path
    raise FileNotFoundError(f"Could not locate project root from {start}")


PROJECT_ROOT = locate_project_root()
DATA_ROOT = PROJECT_ROOT / "data"

# Data/meta configuration (multi-source friendly)
META_FILES = [
    DATA_ROOT / "meta" / "esc50.csv",
    DATA_ROOT / "meta" / "gunshot_kaggle.csv",
    DATA_ROOT / "meta" / "freesound.csv",
]
# Optional raw audio roots by source key; used when filepath is missing.
RAW_AUDIO_ROOTS = {
    "esc50": DATA_ROOT / "esc50",
    "gunshot_kaggle": DATA_ROOT / "gunshot_kaggle",
    "freesound": DATA_ROOT / "freesound",
}
# Legacy single-file paths kept for backward compatibility (prefer META_FILES/RAW_AUDIO_ROOTS).
AUDIO_DIR = DATA_ROOT / "audio"
META_FILE = DATA_ROOT / "meta" / "esc50.csv"

# Cache roots
CACHE_ROOT = PROJECT_ROOT / "cache"
CACHE_MEL64 = CACHE_ROOT / "mel64"
CACHE_MEL64.mkdir(parents=True, exist_ok=True)
# Backward-compatible alias
CACHE_DIR = CACHE_MEL64

# Audio + feature parameters
SR = 21333
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256
# Log-mel framing behavior. `center=True` matches librosa defaults and most of our cached data.
MEL_CENTER = False
# Optional fixed number of frames for deployment (pad/crop time axis to this length).
MEL_TARGET_FRAMES = None
WINDOW_SECONDS = 1.0
WINDOW_HOP = 0.5
SEED = 42

# Per-label windowing defaults (can be overridden at call site)
WINDOW_PARAMS = {
    "glass": {
        "peak_ratio_threshold": 0.8,
        "front_peak_ratio": 0.5,
        "energy_threshold": 0.2,
        "extra_shifts": [0.0],
    },
    "gunshot": {
        "peak_ratio_threshold": 0.6,
        "front_peak_ratio": 0.7,
        "energy_threshold": 0.2,
        "extra_shifts": [0.0, 0.1, -0.1],
    },
    "background": {
        "energy_threshold": 0.15,
        "extra_shifts": [0.0],
    },
}

# Label configuration
# Canonical target labels (multi-label ready).
TARGET_LABELS = ["glass", "gunshot"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(TARGET_LABELS)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}
NUM_CLASSES = len(TARGET_LABELS)
BACKGROUND_LABEL = "background"
BACKGROUND_MULTIPLIER = 3
# Mapping from dataset categories to canonical labels.
POSITIVE_LABELS = {
    "glass_breaking": "glass",
    "gunshot": "gunshot",
}

# Case study / event detection defaults
CASE_STUDY_DIR = CACHE_DIR / "case_study"
CASE_STUDY_META_FILES = META_FILES
CASE_STUDY_DEFAULTS = {
    "background_gain_db": -8.0,
    "crossfade_ms": 15.0,
    "gap_range": (0.2, 8.0),
    "glass_gain_db": 0.0,
    "hyst_high": None,
    "hyst_low": None,
    "merge_gap": 0.12,
    "min_event_dur": 0.3,
    "smooth_k": 1,
    "snr_range_db": (4.0, 9.0),
    "split_top_db": 20.0,
    "start_offset_range": (0.5, 6.0),
    "target_bed_duration": 80.0,
    "threshold": 0.7,
    "tolerance": 0.5,
    "background_only": False,
    "max_event_dur": 1.0,
}

CASE_STUDY_SCHEMA_VERSION = "v1"
HARD_BG_CLASSES = [
    "keyboard_typing",
    "crow",
    "chainsaw",
    "door_wood_knock",
    "coughing"
]
HARD_BG_WEIGHT = 2.0
MAX_EVENT_DUR = 99999.0
