"""Evaluation helpers for classification models."""
from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch


def confusion_matrix(preds: torch.Tensor,
                     targets: torch.Tensor,
                     num_classes: int = 2,
                     normalize: bool = False) -> np.ndarray:
    """Compute confusion matrix; supports multi-hot/single-logit binary by collapsing to class ids."""
    preds = preds.detach().cpu()
    targets = targets.detach().cpu()

    def _collapse_to_class(x: torch.Tensor) -> np.ndarray:
        # If already 1D integer, return directly
        if x.dim() == 1:
            return x.numpy().astype(int)
        # Single-logit binary case: treat >=0.5 as positive class=1 else 0
        if x.shape[1] == 1 and num_classes == 2:
            return (x[:, 0] >= 0.5).numpy().astype(int)
        # Multi-class/multi-hot: use argmax
        return torch.argmax(x, dim=1).numpy().astype(int)

    preds_np = _collapse_to_class(preds)
    targets_np = _collapse_to_class(targets)

    matrix = np.zeros((num_classes, num_classes), dtype=np.float32)
    for p, t in zip(preds_np, targets_np):
        if t >= num_classes or p >= num_classes:
            continue
        matrix[t, p] += 1
    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True) + 1e-8
        matrix = matrix / row_sums
    return matrix


def plot_confusion_matrix(matrix: np.ndarray,
                          class_names: Sequence[str],
                          normalize: bool = False,
                          ax: plt.Axes | None = None) -> plt.Axes:
    """Plot confusion matrix heatmap with readable annotations."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names)
    vmax = matrix.max() if matrix.size else 0.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            text = f"{value:.2f}" if normalize else f"{int(value)}"
            color = "white" if vmax and value > vmax * 0.6 else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=12, fontweight="bold")
    ax.set_title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


__all__ = ["confusion_matrix", "plot_confusion_matrix"]
