"""Resample all audio to TARGET_SR mono and write to a new root with updated meta."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import List

import librosa
import numpy as np
import pandas as pd
import wave

PROJECT_ROOT = Path(__file__).resolve().parents[1]
META_FILES = [
    PROJECT_ROOT / "data/meta/esc50.csv",
    PROJECT_ROOT / "data/meta/gunshot_kaggle.csv",
    PROJECT_ROOT / "data/meta/freesound.csv",
]
OUT_ROOT = PROJECT_ROOT / "data_resampled"
OUT_META_DIR = PROJECT_ROOT / "data/meta_resampled"
TARGET_SR = 22050


def compute_md5(path: Path, chunk: int = 8192) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            data = f.read(chunk)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


def resample_and_write(src_path: Path, dst_path: Path, target_sr: int = TARGET_SR) -> tuple[int, float]:
    """Load audio, resample to mono target_sr, write 16-bit PCM wav. Returns frames and duration."""
    y, _ = librosa.load(src_path, sr=target_sr, mono=True)
    y_int16 = np.clip(y * 32767, -32768, 32767).astype(np.int16)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(dst_path.as_posix(), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(target_sr)
        wf.writeframes(y_int16.tobytes())
    frames = len(y_int16)
    return frames, frames / target_sr


def process_meta(meta_path: Path, out_root: Path, out_meta_dir: Path, target_sr: int = TARGET_SR) -> pd.DataFrame:
    df = pd.read_csv(meta_path)
    rows: List[dict] = []
    for _, row in df.iterrows():
        src_rel = Path(row["filepath"]) if "filepath" in row else Path("data/audio") / row["filename"]
        src_path = src_rel if src_rel.is_absolute() else PROJECT_ROOT / src_rel
        dst_rel = out_root / src_rel.relative_to(PROJECT_ROOT) if src_path.is_absolute() else out_root / src_rel
        frames, dur = resample_and_write(src_path, dst_rel, target_sr=target_sr)
        rows.append(
            {
                "sno": row.get("sno", len(rows) + 1),
                "filepath": dst_rel.relative_to(PROJECT_ROOT).as_posix(),
                "label": row.get("label"),
                "source": row.get("source"),
                "fold_id": row.get("fold_id", ""),
                "duration_sec": round(dur, 3),
                "duration_samples": frames,
                "sr": target_sr,
                "channels": 1,
                "bit_depth": 16,
                "md5": compute_md5(dst_rel),
                "extra_meta": row.get("extra_meta", ""),
            }
        )
    out_meta_dir.mkdir(parents=True, exist_ok=True)
    out_meta_path = out_meta_dir / meta_path.name
    pd.DataFrame(rows).to_csv(out_meta_path, index=False)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Resample all audio to TARGET_SR mono and update meta.")
    parser.add_argument("--meta", nargs="+", default=META_FILES, help="List of meta CSV files")
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT, help="Output root for resampled audio")
    parser.add_argument("--out-meta-dir", type=Path, default=OUT_META_DIR, help="Output directory for new meta CSVs")
    parser.add_argument("--sr", type=int, default=TARGET_SR, help="Target sample rate")
    args = parser.parse_args()

    all_stats = []
    for meta_path in args.meta:
        meta_path = Path(meta_path)
        print(f"Processing {meta_path} ...")
        df_out = process_meta(meta_path, args.out_root, args.out_meta_dir, target_sr=args.sr)
        print("  After resample counts:", df_out["label"].value_counts().to_dict())
        print("  sr/ch:", df_out["sr"].unique(), df_out["channels"].unique())
        all_stats.append((meta_path.name, len(df_out)))
    print("Done. Stats:", all_stats)


if __name__ == "__main__":
    main()
