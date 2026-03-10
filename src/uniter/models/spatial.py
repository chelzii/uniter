from __future__ import annotations

import torch
from torch import nn
from transformers import SegformerForSemanticSegmentation


class SpatialEncoder(nn.Module):
    """
    SegFormer-based spatial encoder.

    The encoder serves two roles:
    1. produce segmentation logits for explainable spatial indicators
    2. produce an image-level embedding by globally pooling class evidence
    """

    def __init__(self, model_name: str, freeze_encoder: bool = True) -> None:
        super().__init__()
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        if freeze_encoder:
            for parameter in self.model.parameters():
                parameter.requires_grad = False

    @property
    def num_labels(self) -> int:
        return int(self.model.config.num_labels)

    @property
    def id2label(self) -> dict[int, str]:
        return {int(key): value for key, value in self.model.config.id2label.items()}

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.model(pixel_values=pixel_values)
        logits = outputs.logits
        pooled_logits = logits.mean(dim=(-2, -1))
        return pooled_logits, logits
