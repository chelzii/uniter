from __future__ import annotations

from uniter.models.classification import ClassificationHead


class SentimentHead(ClassificationHead):
    def __init__(self, input_dim: int, num_classes: int, dropout: float) -> None:
        super().__init__(input_dim=input_dim, num_classes=num_classes, dropout=dropout)
