"""Model definitions for TinyML audio experiments."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .config import NUM_CLASSES


class TinyGlassNet(nn.Module):
    """Compact CNN with only Conv/Pool/ReLU/FC blocks."""

    def __init__(self, in_channels: int = 1, base_channels: int = 16, num_classes: int = NUM_CLASSES):
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(c2, c3, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(c3, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


def count_parameters(model: nn.Module) -> int:
    """Return trainable parameter count."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@dataclass
class TrainingArtifacts:
    """Container for trainer outputs."""

    history: list[dict]
    best_state_dict: dict


__all__ = ["TinyGlassNet", "count_parameters", "TrainingArtifacts"]
