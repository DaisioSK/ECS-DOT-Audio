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


def multilabel_confusion(preds: torch.Tensor,
                         targets: torch.Tensor,
                         threshold: float = 0.5) -> np.ndarray:
    """Compute per-class 2x2 confusion for multi-label (tp/fp/fn/tn per class)."""
    preds = preds.detach().cpu().float()
    targets = targets.detach().cpu().float()
    # Align shapes if mismatch
    if preds.dim() == 1:
        preds = preds.unsqueeze(1)
    if targets.dim() == 1:
        targets = targets.unsqueeze(1)
    if preds.shape[1] != targets.shape[1]:
        c = min(preds.shape[1], targets.shape[1])
        preds = preds[:, :c]
        targets = targets[:, :c]
    pred_bin = (preds >= threshold).float()
    targ_bin = (targets >= 0.5).float()
    tp = (pred_bin * targ_bin).sum(dim=0)
    fp = (pred_bin * (1 - targ_bin)).sum(dim=0)
    fn = ((1 - pred_bin) * targ_bin).sum(dim=0)
    tn = ((1 - pred_bin) * (1 - targ_bin)).sum(dim=0)
    # shape: (num_classes, 4) -> [tn, fp, fn, tp]
    return torch.stack([tn, fp, fn, tp], dim=1).numpy()


def plot_multilabel_confusions(confusions: np.ndarray,
                               class_names: Sequence[str],
                               normalize: bool = False) -> Tuple[plt.Figure, Sequence[plt.Axes]]:
    """Plot per-class 2x2 confusion matrices for multi-label outputs."""
    n_classes = confusions.shape[0]
    fig, axes = plt.subplots(1, n_classes, figsize=(3 * n_classes, 3), squeeze=False)
    axes = axes.ravel()
    for idx, (ax, name) in enumerate(zip(axes, class_names)):
        tn, fp, fn, tp = confusions[idx]
        mat = np.array([[tn, fp], [fn, tp]], dtype=float)
        if normalize:
            denom = mat.sum() + 1e-8
            mat = mat / denom
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=mat.max() + 1e-8)
        ax.set_title(name)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred 0", "Pred 1"])
        ax.set_yticklabels(["True 0", "True 1"])
        vmax = mat.max() if mat.size else 0.0
        for i in range(2):
            for j in range(2):
                val = mat[i, j]
                text = f"{val:.2f}" if normalize else f"{int(val)}"
                color = "white" if vmax and val > vmax * 0.6 else "black"
                ax.text(j, i, text, ha="center", va="center", color=color, fontsize=10, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Multi-label Confusion Matrices" + (" (Normalized)" if normalize else ""))
    fig.tight_layout()
    return fig, axes


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


__all__ = [
    "confusion_matrix",
    "plot_confusion_matrix",
    "multilabel_confusion",
    "plot_multilabel_confusions",
]
