from __future__ import annotations

import torch


def masked_mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    masked = last_hidden_state * mask
    denom = mask.sum(dim=1).clamp_min(1e-6)
    return masked.sum(dim=1) / denom


def aggregate_by_region(
    features: torch.Tensor,
    region_index: torch.Tensor,
    num_regions: int,
) -> torch.Tensor:
    pooled, _ = aggregate_by_region_with_counts(features, region_index, num_regions)
    return pooled


def aggregate_by_region_with_counts(
    features: torch.Tensor,
    region_index: torch.Tensor,
    num_regions: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if features.ndim != 2:
        raise ValueError("Expected a [num_items, hidden_dim] feature tensor.")

    pooled = torch.zeros(
        num_regions,
        features.shape[-1],
        device=features.device,
        dtype=features.dtype,
    )
    pooled.index_add_(0, region_index, features)

    counts = torch.bincount(region_index, minlength=num_regions)
    normalized = pooled / counts.clamp_min(1).unsqueeze(-1)
    return normalized, counts
