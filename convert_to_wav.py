#!/usr/bin/env python3
"""CLI/utility: convert non-WAV audio files in a directory (recursively) to WAV.

Defaults: in-place (same directory), keep originals, skip if .wav already exists,
no resample or downmix unless指定 --sr/--mono。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Sequence


DEFAULT_FORMATS = {
    ".mp3",
    ".flac",
    ".ogg",
    ".m4a",
    ".aac",
    ".wma",
    ".aiff",
    ".aif",
    ".aifc",
    ".alac",
    ".opus",
}


def find_audio_files(root: Path, formats: Sequence[str]) -> Iterable[Path]:
    """Yield audio files under root matching given extensions (case-insensitive)."""
    fmts = {f.lower() for f in formats}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in fmts:
            yield path


def build_ffmpeg_cmd(src: Path, dst: Path, sr: int | None, mono: bool) -> List[str]:
    """Compose ffmpeg command for conversion."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        src.as_posix(),
    ]
    if mono:
        cmd += ["-ac", "1"]
    if sr:
        cmd += ["-ar", str(sr)]
    cmd.append(dst.as_posix())
    return cmd


def convert_tree(
    input_dir: Path,
    out_dir: Path | None = None,
    sr: int | None = None,
    mono: bool = False,
    formats: Sequence[str] | None = None,
    overwrite: bool = False,
) -> tuple[int, int, List[tuple[Path, str]]]:
    """Convert non-WAV audio files under a directory; returns (converted, skipped, errors)."""
    formats = formats or DEFAULT_FORMATS
    src_root = input_dir.resolve()
    candidates = list(find_audio_files(src_root, formats))
    converted = 0
    skipped = 0
    errors: List[tuple[Path, str]] = []

    for src in candidates:
        rel = src.relative_to(src_root)
        dst = (out_dir / rel if out_dir else src).with_suffix(".wav")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and not overwrite:
            skipped += 1
            continue
        cmd = build_ffmpeg_cmd(src, dst, sr, mono)
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
        except FileNotFoundError:
            raise RuntimeError("ffmpeg not found in PATH. Please install ffmpeg.")
        if result.returncode != 0:
            errors.append((src, result.stderr.strip()))
            continue
        converted += 1

    return converted, skipped, errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert non-WAV audio files to WAV.")
    parser.add_argument("input_dir", type=Path, help="Root directory to scan (recursively).")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output root (preserve relative structure). Default: in-place beside source.",
    )
    parser.add_argument("--sr", type=int, default=None, help="Target sample rate (Hz). Default: keep original.")
    parser.add_argument("--mono", action="store_true", help="Downmix to mono. Default: keep original channels.")
    parser.add_argument(
        "--formats",
        type=str,
        default=",".join(sorted(DEFAULT_FORMATS)),
        help="Comma-separated list of source extensions to convert.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing WAV outputs.")
    args = parser.parse_args()

    src_root = args.input_dir.resolve()
    if not src_root.exists():
        parser.error(f"Input directory not found: {src_root}")

    formats = {f if f.startswith(".") else f".{f}" for f in args.formats.split(",") if f}
    converted, skipped, errors = convert_tree(
        input_dir=src_root,
        out_dir=args.out_dir,
        sr=args.sr,
        mono=args.mono,
        formats=formats,
        overwrite=args.overwrite,
    )
    print(f"Done. Converted: {converted}, skipped (exists): {skipped}, errors: {len(errors)}")
    if errors:
        print("\nErrors:")
        for src, err in errors[:10]:
            print(f"- {src}: {err.splitlines()[-1] if err else 'unknown error'}")
        if len(errors) > 10:
            print(f"... and {len(errors) - 10} more")
    return 0


if __name__ == "__main__":
    sys.exit(main())
