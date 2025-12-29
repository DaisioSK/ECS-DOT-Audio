"""Generic meta builder for arbitrary audio folders (wav, recursive).

Usage examples:
- Default output/source from folder name:
  python3 data/meta/build_meta.py --base-dir data/freesound_extra
- Custom output path:
  python3 data/meta/build_meta.py --base-dir data/my_audio --output data/meta/my_audio.csv

Defaults:
- output: data/meta/<foldername>.csv
- source: <foldername>
- label: filename prefix before first underscore (or full stem if no underscore)
"""
from __future__ import annotations

import argparse
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
        with wave.open(path.as_posix(), "rb") as wf:
            channels = wf.getnchannels()
            sr = wf.getframerate()
            frames = wf.getnframes()
            sampwidth = wf.getsampwidth()
            bit_depth = sampwidth * 8
        return channels, sr, frames, bit_depth


def derive_label(path: Path) -> str:
    """Label = filename prefix before first underscore (or full stem)."""
    stem = path.stem
    if "_" in stem:
        return stem.split("_", 1)[0]
    return stem


def build_meta(base_dir: Path, output: Path, source: str) -> int:
    paths = sorted(base_dir.rglob("*.wav"))
    if not paths:
        raise SystemExit(f"No wav files found under {base_dir}")

    rows = []
    for sno, p in enumerate(paths, start=1):
        channels, sr, frames, bit_depth = wav_info(p)
        md5 = compute_md5(p)
        label = derive_label(p)
        rows.append(
            {
                "sno": sno,
                "filepath": p.as_posix(),
                "label": label,
                "source": source,
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

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output} (source={source}, label=prefix_of_filename)")
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description="Build unified meta CSV for an arbitrary audio folder (wav).")
    parser.add_argument("--base-dir", required=True, type=Path, help="Folder to scan recursively for .wav files.")
    parser.add_argument("--output", type=Path, help="Output CSV path. Default: data/meta/<foldername>.csv")
    parser.add_argument("--source", type=str, help="Source name stored in meta. Default: folder name.")
    args = parser.parse_args()

    base_dir = args.base_dir
    if not base_dir.exists():
        raise SystemExit(f"Base dir not found: {base_dir}")
    folder_name = base_dir.name
    output = args.output or Path("data/meta") / f"{folder_name}.csv"
    source = args.source or folder_name

    build_meta(base_dir=base_dir, output=output, source=source)


if __name__ == "__main__":
    main()
