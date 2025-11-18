"""Export helpers for trained models."""
from __future__ import annotations

from pathlib import Path

import torch


def export_to_onnx(model: torch.nn.Module,
                   example_input: torch.Tensor,
                   onnx_path: str | Path,
                   opset: int = 13) -> Path:
    """Export a model to ONNX format with static TinyML-friendly ops."""
    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    example_input = example_input.to(next(model.parameters()).device)
    torch.onnx.export(
        model,
        example_input,
        onnx_path.as_posix(),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch", 3: "time"}, "logits": {0: "batch"}},
        opset_version=opset,
    )
    return onnx_path


__all__ = ["export_to_onnx"]
