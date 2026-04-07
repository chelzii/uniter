from __future__ import annotations

import torch
from torch import nn

from uniter.config import AppConfig
from uniter.data.dataset import RegionBatch
from uniter.models.multimodal import MultimodalRegionModel


class _DummySpatialEncoder(nn.Module):
    def __init__(
        self,
        model_name: str,
        freeze_encoder: bool = True,
        train_decode_head: bool = True,
    ) -> None:
        super().__init__()
        self._num_labels = 2
        self._id2label = {0: "road", 1: "building"}

    @property
    def num_labels(self) -> int:
        return self._num_labels

    @property
    def id2label(self) -> dict[int, str]:
        return self._id2label

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = pixel_values.shape[0]
        pooled_logits = torch.ones(batch_size, self._num_labels)
        segmentation_logits = torch.ones(batch_size, self._num_labels, 2, 2)
        return pooled_logits, segmentation_logits


class _DummyTextEncoder(nn.Module):
    def __init__(
        self,
        model_name: str,
        *,
        pooling: str = "mean",
        freeze_encoder: bool = True,
    ) -> None:
        super().__init__()
        self._hidden_size = 4

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        pooled = input_ids.float().sum(dim=1, keepdim=True)
        return torch.cat([pooled, pooled + 1, pooled + 2, pooled + 3], dim=1)


def test_multimodal_model_identity_branch_produces_iai_and_sentiment_mdi_mask(
    monkeypatch,
) -> None:
    import uniter.models.multimodal as multimodal

    monkeypatch.setattr(multimodal, "SpatialEncoder", _DummySpatialEncoder)
    monkeypatch.setattr(multimodal, "TextEncoder", _DummyTextEncoder)

    config = AppConfig()
    config.identity.enabled = True
    config.lost_space.enabled = True
    model = MultimodalRegionModel(config)
    assert model.current_sentiment_head is not None
    assert model.historical_sentiment_head is not None
    assert model.current_sentiment_head is not model.historical_sentiment_head

    batch = RegionBatch(
        region_ids=["region_a"],
        pixel_values=torch.ones(1, 3, 2, 2),
        segmentation_labels=None,
        image_region_index=torch.tensor([0]),
        image_is_satellite=torch.tensor([False]),
        image_point_ids=["region_a"],
        image_view_directions=["north"],
        image_longitudes=[108.95],
        image_latitudes=[34.25],
        current_input_ids=torch.tensor([[1, 2, 0]]),
        current_attention_mask=torch.tensor([[1, 1, 0]]),
        current_region_index=torch.tensor([0]),
        current_sentiment_labels=torch.tensor([1]),
        historical_input_ids=torch.tensor([[2, 2, 0]]),
        historical_attention_mask=torch.tensor([[1, 1, 0]]),
        historical_region_index=torch.tensor([0]),
        historical_sentiment_labels=torch.tensor([2]),
        identity_input_ids=torch.tensor([[3, 1, 0]]),
        identity_attention_mask=torch.tensor([[1, 1, 0]]),
        identity_region_index=torch.tensor([0]),
        metadata=[{"region_name": "示例区域"}],
        targets={
            "lost_space_label": torch.tensor([2]),
            "sentiment_label": torch.tensor([1]),
            "historical_sentiment_label": torch.tensor([2]),
            "ifi": torch.tensor([float("nan")]),
            "mdi": torch.tensor([float("nan")]),
            "iai": torch.tensor([float("nan")]),
        },
    )

    outputs = model(batch)

    assert bool(outputs.identity_region_mask[0].item()) is True
    assert bool(outputs.mdi_sentiment_mask[0].item()) is True
    assert torch.isfinite(outputs.iai).all()
    assert outputs.lost_space_logits is not None
