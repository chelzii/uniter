from __future__ import annotations

import torch

from uniter.data.dataset import RegionBatch


def resolve_device(configured: str) -> torch.device:
    if configured == "cpu":
        return torch.device("cpu")
    if configured == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def move_region_batch_to_device(batch: RegionBatch, device: torch.device) -> RegionBatch:
    tensor_fields = [
        "pixel_values",
        "segmentation_labels",
        "image_region_index",
        "current_input_ids",
        "current_attention_mask",
        "current_region_index",
        "historical_input_ids",
        "historical_attention_mask",
        "historical_region_index",
        "identity_input_ids",
        "identity_attention_mask",
        "identity_region_index",
    ]
    for field_name in tensor_fields:
        value = getattr(batch, field_name)
        if value is not None:
            setattr(batch, field_name, value.to(device))
    for target_name, value in batch.targets.items():
        batch.targets[target_name] = value.to(device)
    return batch
