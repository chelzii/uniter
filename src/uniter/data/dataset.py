from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from torch.utils.data import Dataset

from uniter.data.manifest import RegionRecord


class RegionDataset(Dataset[RegionRecord]):
    """Region-level dataset with multiple images and texts per sample."""

    def __init__(self, samples: list[RegionRecord]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> RegionRecord:
        return self.samples[index]


@dataclass(slots=True)
class RegionBatch:
    region_ids: list[str]
    pixel_values: Any
    segmentation_labels: Any | None
    image_region_index: Any
    current_input_ids: Any
    current_attention_mask: Any
    current_region_index: Any
    historical_input_ids: Any | None
    historical_attention_mask: Any | None
    historical_region_index: Any | None
    identity_input_ids: Any | None
    identity_attention_mask: Any | None
    identity_region_index: Any | None
    metadata: list[dict[str, Any]]
    targets: dict[str, torch.Tensor]


def resolve_sample_splits(records: list[RegionRecord]) -> dict[str, list[RegionRecord]]:
    buckets: dict[str, list[RegionRecord]] = {"train": [], "val": [], "test": []}
    for record in records:
        buckets.setdefault(record.split, []).append(record)
    return buckets
