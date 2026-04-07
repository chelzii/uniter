from __future__ import annotations

import logging

import torch

from uniter.data.dataset import RegionBatch


logger = logging.getLogger(__name__)


def resolve_device(configured: str) -> torch.device:
    if configured == "cpu":
        return torch.device("cpu")
    if configured == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if not torch.cuda.is_available():
        return torch.device("cpu")

    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
    except RuntimeError:
        return torch.device("cuda")

    minimum_free_bytes = 6 * 1024**3
    if free_bytes < minimum_free_bytes:
        logger.warning(
            "CUDA is available but only %.2f GiB / %.2f GiB is free; "
            "falling back to CPU because experiment.device=auto.",
            free_bytes / 1024**3,
            total_bytes / 1024**3,
        )
        return torch.device("cpu")
    return torch.device("cuda")


def move_region_batch_to_device(batch: RegionBatch, device: torch.device) -> RegionBatch:
    tensor_fields = [
        "pixel_values",
        "segmentation_labels",
        "image_region_index",
        "current_input_ids",
        "current_attention_mask",
        "current_region_index",
        "current_sentiment_labels",
        "historical_input_ids",
        "historical_attention_mask",
        "historical_region_index",
        "historical_sentiment_labels",
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
