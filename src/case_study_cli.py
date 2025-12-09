"""CLI entry for case study generation, inference, smoothing, and logging."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

from src.config import (
    AUDIO_DIR,
    META_FILE,
    POSITIVE_LABELS,
    SR,
    WINDOW_HOP,
    WINDOW_SECONDS,
    CASE_STUDY_DEFAULTS,
    CASE_STUDY_DIR,
    CASE_STUDY_SCHEMA_VERSION,
    SEED,
)
from src.event_detection import (
    ClipSpec,
    GLASS_LABEL,
    build_background_bed,
    bucket_delay,
    bucket_recall_by_snr,
    match_events,
    match_events_with_pairs,
    merge_events,
    mix_glass_on_bed,
    predict_glass_probs,
    sliding_log_mel_windows,
    smooth_probabilities,
)
from src.inference import create_onnx_session, load_torch_checkpoint


def _load_config(path: Path | None) -> dict[str, Any]:
    cfg = {**CASE_STUDY_DEFAULTS}
    if path and path.exists():
        user = json.loads(path.read_text())
        cfg.update(user)
    return cfg


def run_case_study(cfg_path: Path | None, output_dir: Path | None, seed: int | None):
    cfg = _load_config(cfg_path)
    rng = np.random.default_rng(seed)

    run_id = f"run_{int(time.time())}"
    out_dir = output_dir or CASE_STUDY_DIR
    out_dir = out_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save config snapshot
    snapshot = {
        "schema": CASE_STUDY_SCHEMA_VERSION,
        "params": cfg,
        "rng_seed": seed,
        "run_id": run_id,
    }
    (out_dir / "run_config.json").write_text(json.dumps(snapshot, indent=2))

    # Collect clips
    external_dir = Path("data/external")
    glass_paths = sorted(external_dir.glob("glass_ext_*"))
    if not glass_paths:
        raise FileNotFoundError("No external glass clips found under data/external")
    glass_specs = [ClipSpec(path=p, label=GLASS_LABEL, gain_db=cfg["glass_gain_db"]) for p in glass_paths]

    import pandas as pd

    meta_df = pd.read_csv(META_FILE)
    non_glass_df = meta_df[~meta_df["category"].isin(POSITIVE_LABELS.keys())]
    bg_sample_n = max(18, len(glass_specs) * 4)
    bg_samples = non_glass_df.sample(n=bg_sample_n, random_state=seed, replace=len(non_glass_df) < bg_sample_n)
    bg_specs = [
        ClipSpec(path=AUDIO_DIR / row["filename"], label="background", gain_db=cfg["background_gain_db"])
        for _, row in bg_samples.iterrows()
    ]

    # Build bed
    bed = build_background_bed(
        bg_specs,
        sr=SR,
        target_duration=cfg["target_bed_duration"],
        crossfade_ms=cfg["crossfade_ms"],
        normalize=True,
    )

    # Mix glass and detect GT
    audio, gt_events = mix_glass_on_bed(
        bed,
        glass_specs,
        sr=SR,
        start_offset_range=cfg["start_offset_range"],
        gap_range=cfg["gap_range"],
        crossfade_ms=cfg["crossfade_ms"],
        snr_range_db=cfg["snr_range_db"],
        split_top_db=cfg["split_top_db"],
        min_event_dur=cfg["min_event_dur"],
        seed=None,
        rng=rng,
    )
    mix_path = out_dir / "mix.wav"
    sf.write(mix_path, audio, SR)

    # Sliding windows
    batch, spans = sliding_log_mel_windows(
        audio,
        sr=SR,
        window_seconds=WINDOW_SECONDS,
        hop_seconds=WINDOW_HOP,
        min_coverage=0.7,
    )

    # Inference (torch, onnx optional)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path("cache/experiments/tinyglassnet_best.pt")
    onnx_path = Path("cache/experiments/tinyglassnet_best.onnx")

    model, _ = load_torch_checkpoint(ckpt_path, device=device)
    torch_probs, _ = predict_glass_probs(batch, spans, model=model, device=device)

    onnx_probs = None
    if onnx_path.exists():
        onnx_sess = create_onnx_session(onnx_path)
        onnx_probs, _ = predict_glass_probs(batch, spans, session=onnx_sess, device="cpu")

    # Smoothing
    smooth_k = int(cfg.get("smooth_k", 1) or 1)
    torch_probs_smooth = smooth_probabilities(torch_probs, kernel_size=smooth_k)

    # Merge and eval
    pred_events = merge_events(spans, torch_probs_smooth, threshold=cfg["threshold"], merge_gap=cfg["merge_gap"])
    metrics = match_events(pred_events, gt_events, tolerance=cfg["tolerance"])
    pairs, unmatched_gt, unmatched_pred = match_events_with_pairs(pred_events, gt_events, tolerance=cfg["tolerance"])

    matched_gt_indices = [gt_events.index(gt) for gt, _, _ in pairs]
    snr_recalls = bucket_recall_by_snr(gt_events, matched_gt_indices)
    delay_buckets = bucket_delay(pairs)

    # Persist results
    results = {
        "run_id": run_id,
        "mix_path": str(mix_path),
        "gt_events": [ev.__dict__ for ev in gt_events],
        "pred_events": pred_events,
        "metrics": metrics,
        "spans": spans,
        "torch_probs": torch_probs,
        "torch_probs_smooth": torch_probs_smooth,
        "onnx_probs": onnx_probs,
        "snr_recalls": snr_recalls,
        "delay_buckets": delay_buckets,
        "unmatched_gt": unmatched_gt,
        "unmatched_pred": unmatched_pred,
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=float))
    print(f"run_id={run_id} metrics={metrics} mix={mix_path}")


def main():
    parser = argparse.ArgumentParser(description="Run case study pipeline")
    parser.add_argument("--config", type=Path, default=None, help="Override defaults via JSON file")
    parser.add_argument("--output", type=Path, default=None, help="Output directory (default CASE_STUDY_DIR)")
    parser.add_argument("--seed", type=int, default=SEED, help="RNG seed")
    args = parser.parse_args()
    run_case_study(args.config, args.output, args.seed)


if __name__ == "__main__":
    main()
