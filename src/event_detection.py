"""Event-level inference helpers for sliding-window detection."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import librosa
import numpy as np
import torch

from .config import HOP_LENGTH, N_FFT, N_MELS, POSITIVE_LABELS, SR, WINDOW_HOP, WINDOW_SECONDS
from .data_utils import log_mel_spectrogram
from .inference import InferenceResult, run_onnx_inference, run_torch_inference


GLASS_LABEL = list(POSITIVE_LABELS.values())[0]


@dataclass
class ClipSpec:
    """Describe a clip to be placed on the timeline."""

    path: Path
    label: str
    gain_db: float = 0.0


@dataclass
class GroundTruthEvent:
    """Ground-truth event location on the composed timeline."""

    label: str
    start: float
    end: float
    source: str
    snr_db: float | None = None


def _apply_gain(audio: np.ndarray, gain_db: float) -> np.ndarray:
    if abs(gain_db) < 1e-6:
        return audio
    gain = librosa.db_to_amplitude(gain_db)
    return audio * gain


def _apply_fade(audio: np.ndarray, fade_samples: int) -> np.ndarray:
    if fade_samples <= 0 or len(audio) == 0:
        return audio
    fade_samples = min(fade_samples, len(audio))
    ramp = np.linspace(0.0, 1.0, fade_samples, endpoint=False, dtype=np.float32)
    out = audio.copy()
    out[:fade_samples] *= ramp  # fade in
    out[-fade_samples:] *= ramp[::-1]  # fade out
    return out


def _crossfade_blend(timeline: List[np.ndarray], waveform: np.ndarray, crossfade: int) -> tuple[int, np.ndarray]:
    """Blend waveform onto the last clip with crossfade; return used overlap and trimmed waveform."""
    if not timeline or crossfade <= 0:
        return 0, waveform
    overlap = min(crossfade, len(timeline[-1]), len(waveform))
    if overlap > 0:
        fade_out = np.linspace(1.0, 0.0, overlap, endpoint=False, dtype=np.float32)
        fade_in = np.linspace(0.0, 1.0, overlap, endpoint=False, dtype=np.float32)
        timeline[-1][-overlap:] = timeline[-1][-overlap:] * fade_out + waveform[:overlap] * fade_in
        waveform = waveform[overlap:]
    return overlap, waveform


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)) + 1e-8))


def _scale_to_snr(fg: np.ndarray, bg: np.ndarray, snr_db: float) -> np.ndarray:
    bg_rms = _rms(bg)
    if bg_rms < 1e-6:
        bg_rms = 1e-3  # avoid zero background leading to huge gains
    target_fg_rms = bg_rms * (10 ** (snr_db / 20.0))
    fg_rms = _rms(fg)
    if fg_rms < 1e-6:
        return fg
    scale = target_fg_rms / fg_rms
    return fg * scale


def detect_events_in_clip(audio: np.ndarray,
                          sr: int,
                          top_db: float = 35.0,
                          min_event_dur: float = 0.08) -> List[Tuple[float, float]]:
    """Detect active segments in a clip using non-silent intervals."""
    intervals = librosa.effects.split(audio, top_db=top_db)
    events: List[Tuple[float, float]] = []
    for start, end in intervals:
        dur = (end - start) / sr
        if dur < min_event_dur:
            continue
        events.append((start / sr, end / sr))
    if not events:
        events.append((0.0, len(audio) / sr))
    return events


def compose_timeline(clips: Sequence[ClipSpec],
                     sr: int = SR,
                     crossfade_ms: float = 15.0,
                     normalize: bool = True) -> Tuple[np.ndarray, List[GroundTruthEvent]]:
    """Concatenate clips with optional crossfade and return audio + GT events."""
    timeline: List[np.ndarray] = []
    events: List[GroundTruthEvent] = []
    cursor = 0  # in samples
    crossfade = int(sr * crossfade_ms / 1000.0)

    for spec in clips:
        waveform, _ = librosa.load(spec.path, sr=sr)
        waveform = waveform.astype(np.float32)
        clip_len = len(waveform)
        waveform = _apply_gain(waveform, spec.gain_db)
        waveform = _apply_fade(waveform, crossfade) if crossfade > 0 else waveform

        overlap, waveform = _crossfade_blend(timeline, waveform, crossfade)
        start_sample = max(cursor - overlap, 0)
        end_sample = start_sample + clip_len
        timeline.append(waveform)
        cursor = end_sample

        if spec.label != "background":
            events.append(
                GroundTruthEvent(
                    label=spec.label,
                    start=start_sample / sr,
                    end=end_sample / sr,
                    source=spec.path.stem,
                )
            )

    if not timeline:
        return np.zeros(0, dtype=np.float32), []

    audio = np.concatenate(timeline).astype(np.float32)
    if normalize:
        peak = np.max(np.abs(audio)) + 1e-8
        if peak > 1.0:
            audio = audio / peak * 0.98
    return audio, events


def build_background_bed(clips: Sequence[ClipSpec],
                         sr: int = SR,
                         target_duration: float | None = None,
                         crossfade_ms: float = 15.0,
                         normalize: bool = True) -> np.ndarray:
    """Create a background bed by concatenating clips (repeat if needed to reach duration)."""
    if not clips:
        return np.zeros(0, dtype=np.float32)
    timeline: List[np.ndarray] = []
    crossfade = int(sr * crossfade_ms / 1000.0)
    cursor = 0
    clip_idx = 0
    target_samples = int(target_duration * sr) if target_duration else None

    while True:
        spec = clips[clip_idx % len(clips)]
        waveform, _ = librosa.load(spec.path, sr=sr)
        waveform = waveform.astype(np.float32)
        waveform = _apply_gain(waveform, spec.gain_db)
        waveform = _apply_fade(waveform, crossfade) if crossfade > 0 else waveform

        overlap, waveform = _crossfade_blend(timeline, waveform, crossfade)
        timeline.append(waveform)
        cursor += len(waveform)

        clip_idx += 1
        if target_samples is None or cursor >= target_samples:
            break

    bed = np.concatenate(timeline).astype(np.float32)
    if target_samples and len(bed) > target_samples:
        bed = bed[:target_samples]
    if normalize:
        peak = np.max(np.abs(bed)) + 1e-8
        if peak > 1.0:
            bed = bed / peak * 0.98
    return bed


def mix_glass_on_bed(background_bed: np.ndarray,
                     glass_clips: Sequence[ClipSpec],
                     sr: int = SR,
                     start_offset_range: Tuple[float, float] = (0.5, 5.0),
                     gap_range: Tuple[float, float] = (-0.5, 8.0),
                     crossfade_ms: float = 15.0,
                     snr_range_db: Tuple[float, float] | None = (3.0, 9.0),
                     split_top_db: float = 35.0,
                     min_event_dur: float = 0.08,
                     seed: int | None = 42,
                     rng: np.random.Generator | None = None) -> Tuple[np.ndarray, List[GroundTruthEvent]]:
    """Overlay glass clips on top of a background bed with random spacing and SNR."""
    if background_bed.ndim != 1:
        raise ValueError("background_bed must be mono waveform array")
    audio = background_bed.astype(np.float32).copy()
    events: List[GroundTruthEvent] = []
    crossfade = int(sr * crossfade_ms / 1000.0)
    rng = rng or np.random.default_rng(seed)

    glass_order = list(glass_clips)
    rng.shuffle(glass_order)
    cursor = int(rng.uniform(start_offset_range[0], start_offset_range[1]) * sr)

    for spec in glass_order:
        waveform, _ = librosa.load(spec.path, sr=sr)
        waveform = waveform.astype(np.float32)
        waveform = _apply_gain(waveform, spec.gain_db)
        waveform = _apply_fade(waveform, crossfade) if crossfade > 0 else waveform

        start = cursor
        end = start + len(waveform)

        if end > len(audio):
            pad = end - len(audio)
            audio = np.pad(audio, (0, pad))

        snr_used = None
        if snr_range_db is not None:
            snr_used = float(rng.uniform(snr_range_db[0], snr_range_db[1]))
            bg_slice = audio[start:end]
            waveform = _scale_to_snr(waveform, bg_slice, snr_used)

        audio[start:end] += waveform

        clip_events = detect_events_in_clip(waveform, sr=sr, top_db=split_top_db, min_event_dur=min_event_dur)
        for ev_start, ev_end in clip_events:
            events.append(
                GroundTruthEvent(
                    label=spec.label,
                    start=(start / sr) + ev_start,
                    end=(start / sr) + ev_end,
                    source=spec.path.stem,
                    snr_db=snr_used,
                )
            )

        gap = rng.uniform(gap_range[0], gap_range[1])
        cursor = end + int(gap * sr)

    peak = np.max(np.abs(audio)) + 1e-8
    if peak > 1.0:
        audio = audio / peak * 0.98
    return audio, events


def sliding_log_mel_windows(y: np.ndarray,
                            sr: int = SR,
                            window_seconds: float = WINDOW_SECONDS,
                            hop_seconds: float = WINDOW_HOP,
                            min_coverage: float = 0.7,
                            n_fft: int = N_FFT,
                            hop_length: int = HOP_LENGTH,
                            n_mels: int = N_MELS) -> Tuple[torch.Tensor, List[Tuple[float, float]]]:
    """Create log-mel windows and their time spans from a long waveform."""
    win_len = int(window_seconds * sr)
    hop_len = int(hop_seconds * sr)
    windows: List[torch.Tensor] = []
    spans: List[Tuple[float, float]] = []

    if len(y) == 0:
        return torch.empty(0), spans

    if len(y) < win_len:
        coverage = len(y) / max(win_len, 1)
        if coverage < min_coverage:
            return torch.empty(0), spans
        y = np.pad(y, (0, win_len - len(y)))

    for start in range(0, len(y), hop_len):
        segment = y[start:start + win_len]
        coverage = len(segment) / win_len
        if coverage < min_coverage:
            continue
        if len(segment) < win_len:
            segment = np.pad(segment, (0, win_len - len(segment)))
        mel_db = log_mel_spectrogram(
            segment,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        mel_tensor = torch.from_numpy(mel_db).unsqueeze(0).float()  # (1, n_mels, frames)
        windows.append(mel_tensor)
        spans.append((start / sr, (start + win_len) / sr))

        if len(segment) < win_len:
            break

    if not windows:
        return torch.empty(0), spans
    batch = torch.stack(windows, dim=0)  # (B, 1, n_mels, frames)
    return batch, spans


def merge_events(window_spans: Sequence[Tuple[float, float]],
                 glass_probs: Sequence[float],
                 threshold: float = 0.5,
                 merge_gap: float = 0.25) -> List[dict]:
    """Merge consecutive high-probability windows into predicted events."""
    events: List[dict] = []
    for (start, end), prob in zip(window_spans, glass_probs):
        if prob < threshold:
            continue
        if not events or start > events[-1]["end"] + merge_gap:
            events.append({"start": start, "end": end, "max_prob": prob})
        else:
            events[-1]["end"] = max(events[-1]["end"], end)
            events[-1]["max_prob"] = max(events[-1]["max_prob"], prob)
    return events


def match_events(pred_events: Sequence[dict],
                 gt_events: Sequence[GroundTruthEvent],
                 tolerance: float = 0.5) -> dict:
    """Match predicted events to ground truth with temporal tolerance."""
    matched_gt = set()
    tp = 0
    fp = 0
    for pred in pred_events:
        hit = False
        for idx, gt in enumerate(gt_events):
            if idx in matched_gt:
                continue
            overlaps = pred["end"] + tolerance >= gt.start and pred["start"] - tolerance <= gt.end
            if overlaps:
                matched_gt.add(idx)
                tp += 1
                hit = True
                break
        if not hit:
            fp += 1
    fn = len(gt_events) - tp
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def predict_glass_probs(batch: torch.Tensor,
                        spans: Sequence[Tuple[float, float]],
                        model: torch.nn.Module | None = None,
                        session=None,
                        device: torch.device | str = "cpu") -> Tuple[List[float], InferenceResult]:
    """Run Torch or ONNX inference on a batch and return per-window glass probabilities."""
    if model is None and session is None:
        raise ValueError("Provide either a Torch model or an ONNX session.")
    if model is not None:
        result = run_torch_inference(model, batch, device=device)
    else:
        result = run_onnx_inference(session, batch)
    glass_probs = result.probs[:, 1].tolist()
    if len(glass_probs) != len(spans):
        raise ValueError("Mismatch between spans and probability outputs.")
    return glass_probs, result


__all__ = [
    "ClipSpec",
    "GroundTruthEvent",
    "compose_timeline",
    "build_background_bed",
    "mix_glass_on_bed",
    "detect_events_in_clip",
    "sliding_log_mel_windows",
    "merge_events",
    "match_events",
    "predict_glass_probs",
    "GLASS_LABEL",
]
