"""Generate unified meta CSV for freesound downloads (full rebuild)."""
from __future__ import annotations

import csv
import hashlib
import wave
from pathlib import Path

import soundfile as sf

FIELDNAMES = [
    "sno",
    "filepath",
    "label",
    "source",
    "fold_id",
    "duration_sec",
    "duration_samples",
    "sr",
    "channels",
    "bit_depth",
    "md5",
    "extra_meta",
]


def compute_md5(path: Path, chunk: int = 8192) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            data = f.read(chunk)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


def wav_info(path: Path) -> tuple[int, int, int, int]:
    """Return channels, sample rate, frame count, bit depth for a wav file."""
    try:
        info = sf.info(path)
        channels = info.channels
        sr = info.samplerate
        frames = info.frames
        subtype = info.subtype or ""
        # subtype examples: PCM_16, PCM_24, PCM_32, PCM_U8, DOUBLE
        bit_depth = None
        if "PCM" in subtype and "_" in subtype:
            try:
                bit_depth = int(subtype.split("_")[1])
            except Exception:
                bit_depth = None
        if bit_depth is None and "24" in info.subtype_info:
            bit_depth = 24
        elif bit_depth is None and "16" in info.subtype_info:
            bit_depth = 16
        elif bit_depth is None and "32" in info.subtype_info:
            bit_depth = 32
        return channels, sr, frames, bit_depth or 16
    except Exception:
        # Fallback to wave for simple PCM
        with wave.open(path.as_posix(), "rb") as wf:
            channels = wf.getnchannels()
            sr = wf.getframerate()
            frames = wf.getnframes()
            sampwidth = wf.getsampwidth()
            bit_depth = sampwidth * 8
        return channels, sr, frames, bit_depth


def build_meta():
    base_dir = Path("data/freesound")
    paths = sorted(base_dir.rglob("*.wav"))
    if not paths:
        raise SystemExit(f"No wav files found under {base_dir}")

    rows = []
    sno = 1
    for p in paths:
        channels, sr, frames, bit_depth = wav_info(p)
        md5 = compute_md5(p)
        # derive category from filename prefix before first underscore
        stem = p.stem
        if "_" in stem:
            category = stem.split("_", 1)[0]
        else:
            category = stem
        rows.append(
            {
                "sno": sno,
                "filepath": p.as_posix(),
                "label": category,
                "source": "freesound",
                "fold_id": "",
                "duration_sec": f"{frames / sr:.1f}" if sr else "",
                "duration_samples": frames,
                "sr": sr,
                "channels": channels,
                "bit_depth": bit_depth,
                "md5": md5,
                "extra_meta": "",
            }
        )
        sno += 1

    out_path = Path("data/meta/freesound.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    build_meta()
