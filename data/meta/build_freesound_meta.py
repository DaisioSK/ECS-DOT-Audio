"""Generate unified meta CSV for freesound downloads (full rebuild)."""
from __future__ import annotations

import csv
import hashlib
import wave
from pathlib import Path


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
        rows.append(
            {
                "sno": sno,
                "filepath": p.as_posix(),
                "label": "glass",
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
