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
AUDIO_DIR = DATA_ROOT / "audio"
META_FILE = DATA_ROOT / "meta" / "esc50.csv"
CACHE_DIR = PROJECT_ROOT / "cache" / "mel64"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Audio + feature parameters
SR = 22050
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256
WINDOW_SECONDS = 1.0
WINDOW_HOP = 0.5
SEED = 42

POSITIVE_LABELS = {
    "glass_breaking": "glass_breaking",
}
BACKGROUND_LABEL = "background"
BACKGROUND_MULTIPLIER = 3

# Case study / event detection defaults
CASE_STUDY_DIR = CACHE_DIR / "case_study"
CASE_STUDY_DEFAULTS = {
    "target_bed_duration": 60.0,
    "background_gain_db": -8.0,
    "glass_gain_db": 0.0,
    "start_offset_range": (0.5, 6.0),
    "gap_range": (-0.5, 8.0),
    "snr_range_db": (3.0, 9.0),
    "crossfade_ms": 15.0,
    "split_top_db": 35.0,
    "min_event_dur": 0.08,
    "threshold": 0.5,
    "merge_gap": 0.25,
    "tolerance": 0.5,
}

CASE_STUDY_SCHEMA_VERSION = "v1"
