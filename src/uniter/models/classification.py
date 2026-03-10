from __future__ import annotations

from torch import nn


class ClassificationHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, num_classes),
        )

    def forward(self, features):
        return self.classifier(features)
