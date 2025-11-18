"""Inference utilities for TinyGlassNet models."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import onnxruntime as ort
import torch

from .models import TinyGlassNet


@dataclass
class InferenceResult:
    logits: torch.Tensor
    probs: torch.Tensor
    preds: torch.Tensor


def load_torch_checkpoint(checkpoint_path: str | Path,
                          device: torch.device | str = "cpu") -> tuple[TinyGlassNet, Dict]:
    """Load TinyGlassNet checkpoint and return model + payload."""
    checkpoint_path = Path(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location=device)
    state_dict = payload.get("model_state_dict") or payload.get("model")
    if state_dict is None:
        raise KeyError("Checkpoint missing 'model_state_dict' or 'model' keys.")
    model = TinyGlassNet()
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, payload


def load_mel_batch(index_df, max_items: int | None = None) -> torch.Tensor:
    """Load mel tensors for selected rows and stack into a batch."""
    tensors = []
    for i, (_, row) in enumerate(index_df.iterrows()):
        if max_items is not None and i >= max_items:
            break
        mel = np.load(row["path"])
        tensors.append(torch.from_numpy(mel).unsqueeze(0).float())
    if not tensors:
        raise ValueError("No samples available for inference batch.")
    return torch.stack(tensors, dim=0)


def run_torch_inference(model: TinyGlassNet,
                        batch: torch.Tensor,
                        device: torch.device | str = "cpu") -> InferenceResult:
    """Run forward pass on a batch of mel tensors."""
    batch = batch.to(device)
    with torch.inference_mode():
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
    return InferenceResult(
        logits=logits.cpu(),
        probs=probs.cpu(),
        preds=preds.cpu(),
    )


def create_onnx_session(onnx_path: str | Path,
                        providers: Sequence[str] | None = None) -> ort.InferenceSession:
    """Create ONNX Runtime session for exported TinyGlassNet."""
    onnx_path = Path(onnx_path)
    if providers is None:
        providers = ["CPUExecutionProvider"]
    return ort.InferenceSession(onnx_path.as_posix(), providers=list(providers))


def run_onnx_inference(session: ort.InferenceSession,
                       batch: torch.Tensor) -> InferenceResult:
    """Run ONNX model on cpu batch tensor."""
    inputs = {"input": batch.numpy()}
    logits_np = session.run(["logits"], inputs)[0]
    logits = torch.from_numpy(logits_np)
    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)
    return InferenceResult(logits=logits, probs=probs, preds=preds)


__all__ = [
    "InferenceResult",
    "load_torch_checkpoint",
    "load_mel_batch",
    "run_torch_inference",
    "create_onnx_session",
    "run_onnx_inference",
]
