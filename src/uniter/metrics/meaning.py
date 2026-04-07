from __future__ import annotations

import torch
import torch.nn.functional as F


def _nan_vector(reference: torch.Tensor) -> torch.Tensor:
    return torch.full(
        (reference.shape[0],),
        torch.nan,
        device=reference.device,
        dtype=reference.dtype,
    )


def compute_embedding_drift(
    current_embeddings: torch.Tensor,
    historical_embeddings: torch.Tensor | None,
    *,
    normalizer: float,
    historical_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if historical_embeddings is None:
        return _nan_vector(current_embeddings)

    mask = (
        historical_mask.to(device=current_embeddings.device, dtype=torch.bool)
        if historical_mask is not None
        else torch.ones(
            current_embeddings.shape[0],
            device=current_embeddings.device,
            dtype=torch.bool,
        )
    )
    drift = _nan_vector(current_embeddings)
    if not torch.any(mask):
        return drift

    similarity = F.cosine_similarity(
        current_embeddings[mask],
        historical_embeddings[mask],
        dim=-1,
    )
    normalized = (1.0 - similarity) / max(normalizer, 1e-6)
    drift[mask] = normalized.clamp(min=0.0, max=1.0)
    return drift


def _sentiment_class_positions(logits: torch.Tensor) -> torch.Tensor:
    return torch.linspace(
        -1.0,
        1.0,
        steps=logits.shape[-1],
        device=logits.device,
        dtype=logits.dtype,
    )


def compute_sentiment_drift(
    current_logits: torch.Tensor,
    historical_logits: torch.Tensor | None,
    *,
    normalizer: float,
    historical_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if historical_logits is None:
        return _nan_vector(current_logits)

    mask = (
        historical_mask.to(device=current_logits.device, dtype=torch.bool)
        if historical_mask is not None
        else torch.ones(
            current_logits.shape[0],
            device=current_logits.device,
            dtype=torch.bool,
        )
    )
    drift = _nan_vector(current_logits)
    if not torch.any(mask):
        return drift

    class_positions = _sentiment_class_positions(current_logits)
    current_scores = torch.softmax(current_logits[mask], dim=-1) @ class_positions
    historical_scores = torch.softmax(historical_logits[mask], dim=-1) @ class_positions
    score_range = float(class_positions.max().item() - class_positions.min().item())
    normalized = torch.abs(current_scores - historical_scores) / max(
        normalizer * max(score_range, 1e-6),
        1e-6,
    )
    drift[mask] = normalized.clamp(min=0.0, max=1.0)
    return drift


def _aggregate_region_distributions(
    logits: torch.Tensor,
    region_index: torch.Tensor,
    num_regions: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    working_logits = logits.float()
    probabilities = torch.softmax(working_logits, dim=-1)
    num_classes = probabilities.shape[-1]
    mean_probs = torch.zeros(
        num_regions,
        num_classes,
        device=probabilities.device,
        dtype=probabilities.dtype,
    )
    counts = torch.bincount(region_index, minlength=num_regions)
    mean_probs.index_add_(0, region_index, probabilities)
    mean_probs = mean_probs / counts.clamp_min(1).unsqueeze(-1)
    return mean_probs, counts


def _aggregate_region_polarity_std(
    logits: torch.Tensor,
    region_index: torch.Tensor,
    num_regions: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    working_logits = logits.float()
    class_positions = _sentiment_class_positions(working_logits)
    polarity_scores = torch.softmax(working_logits, dim=-1) @ class_positions
    counts = torch.bincount(region_index, minlength=num_regions)
    means = torch.zeros(num_regions, device=logits.device, dtype=polarity_scores.dtype)
    means.index_add_(0, region_index, polarity_scores)
    means = means / counts.clamp_min(1)
    squared_diff = (polarity_scores - means[region_index]) ** 2
    variances = torch.zeros(num_regions, device=logits.device, dtype=polarity_scores.dtype)
    variances.index_add_(0, region_index, squared_diff)
    variances = variances / counts.clamp_min(1)
    return variances.sqrt(), counts


def compute_distributional_sentiment_drift(
    current_record_logits: torch.Tensor,
    current_region_index: torch.Tensor,
    historical_record_logits: torch.Tensor | None,
    historical_region_index: torch.Tensor | None,
    *,
    num_regions: int,
    historical_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if historical_record_logits is None or historical_region_index is None:
        return _nan_vector(torch.zeros(num_regions, device=current_record_logits.device))

    mask = (
        historical_mask.to(device=current_record_logits.device, dtype=torch.bool)
        if historical_mask is not None
        else torch.ones(num_regions, device=current_record_logits.device, dtype=torch.bool)
    )
    drift = _nan_vector(torch.zeros(num_regions, device=current_record_logits.device))
    if not torch.any(mask):
        return drift

    current_distribution, current_counts = _aggregate_region_distributions(
        current_record_logits,
        current_region_index,
        num_regions,
    )
    historical_distribution, historical_counts = _aggregate_region_distributions(
        historical_record_logits,
        historical_region_index,
        num_regions,
    )
    current_std, _ = _aggregate_region_polarity_std(
        current_record_logits,
        current_region_index,
        num_regions,
    )
    historical_std, _ = _aggregate_region_polarity_std(
        historical_record_logits,
        historical_region_index,
        num_regions,
    )

    valid_mask = mask & current_counts.gt(0) & historical_counts.gt(0)
    if not torch.any(valid_mask):
        return drift

    current_probs = current_distribution[valid_mask].clamp_min(1e-6)
    historical_probs = historical_distribution[valid_mask].clamp_min(1e-6)
    midpoint = ((current_probs + historical_probs) * 0.5).clamp_min(1e-6)
    js_divergence = 0.5 * (
        (current_probs * (current_probs.log() - midpoint.log())).sum(dim=-1)
        + (historical_probs * (historical_probs.log() - midpoint.log())).sum(dim=-1)
    )
    js_divergence = js_divergence / max(float(torch.log(torch.tensor(2.0)).item()), 1e-6)

    score_range = 2.0
    std_gap = torch.abs(current_std[valid_mask] - historical_std[valid_mask]) / score_range
    blended = js_divergence * 0.75 + std_gap * 0.25
    drift[valid_mask] = blended.clamp(min=0.0, max=1.0)
    return drift


def compute_mdi(
    current_embeddings: torch.Tensor,
    historical_embeddings: torch.Tensor | None,
    *,
    normalizer: float,
    mode: str = "hybrid",
    sentiment_weight: float = 1.0,
    current_sentiment_logits: torch.Tensor | None = None,
    historical_sentiment_logits: torch.Tensor | None = None,
    current_sentiment_record_logits: torch.Tensor | None = None,
    current_sentiment_region_index: torch.Tensor | None = None,
    historical_sentiment_record_logits: torch.Tensor | None = None,
    historical_sentiment_region_index: torch.Tensor | None = None,
    historical_mask: torch.Tensor | None = None,
    sentiment_mode_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    embedding_drift = compute_embedding_drift(
        current_embeddings,
        historical_embeddings,
        normalizer=normalizer,
        historical_mask=historical_mask,
    )

    sentiment_mask = (
        sentiment_mode_mask.to(device=embedding_drift.device, dtype=torch.bool)
        if sentiment_mode_mask is not None
        else (
            historical_mask.to(device=embedding_drift.device, dtype=torch.bool)
            if historical_mask is not None
            else torch.ones(
                embedding_drift.shape[0],
                device=embedding_drift.device,
                dtype=torch.bool,
            )
        )
    )

    if (
        current_sentiment_record_logits is not None
        and current_sentiment_region_index is not None
        and historical_sentiment_record_logits is not None
        and historical_sentiment_region_index is not None
    ):
        sentiment_drift = compute_distributional_sentiment_drift(
            current_sentiment_record_logits,
            current_sentiment_region_index,
            historical_sentiment_record_logits,
            historical_sentiment_region_index,
            num_regions=embedding_drift.shape[0],
            historical_mask=sentiment_mask,
        ).to(dtype=embedding_drift.dtype)
    elif current_sentiment_logits is not None and historical_sentiment_logits is not None:
        sentiment_drift = compute_sentiment_drift(
            current_sentiment_logits,
            historical_sentiment_logits,
            normalizer=normalizer,
            historical_mask=sentiment_mask,
        ).to(dtype=embedding_drift.dtype)
    else:
        return embedding_drift

    if not torch.any(sentiment_mask):
        return embedding_drift

    if mode == "sentiment_only":
        mdi = embedding_drift.clone()
        sentiment_available = ~torch.isnan(sentiment_drift)
        mdi[sentiment_available] = sentiment_drift[sentiment_available]
        return mdi
    if mode == "sentiment_first":
        mdi = embedding_drift.clone()
        sentiment_available = ~torch.isnan(sentiment_drift)
        mdi[sentiment_available] = sentiment_drift[sentiment_available]
        return mdi
    if mode == "thesis":
        mdi = embedding_drift.clone()
        embedding_available = ~torch.isnan(embedding_drift)
        sentiment_available = ~torch.isnan(sentiment_drift)
        blend_mask = sentiment_mask & embedding_available & sentiment_available
        sentiment_only_mask = sentiment_mask & ~embedding_available & sentiment_available
        if torch.any(blend_mask):
            mdi[blend_mask] = (
                embedding_drift[blend_mask] * (1.0 - sentiment_weight)
                + sentiment_drift[blend_mask] * sentiment_weight
            )
        if torch.any(sentiment_only_mask):
            mdi[sentiment_only_mask] = sentiment_drift[sentiment_only_mask]
        return mdi

    mdi = embedding_drift.clone()
    if sentiment_weight <= 0.0:
        return mdi

    embedding_available = ~torch.isnan(embedding_drift)
    sentiment_available = ~torch.isnan(sentiment_drift)
    blend_mask = sentiment_mask & embedding_available & sentiment_available
    sentiment_only_mask = sentiment_mask & ~embedding_available & sentiment_available
    if torch.any(blend_mask):
        mdi[blend_mask] = (
            embedding_drift[blend_mask] * (1.0 - sentiment_weight)
            + sentiment_drift[blend_mask] * sentiment_weight
        )
    if torch.any(sentiment_only_mask):
        mdi[sentiment_only_mask] = sentiment_drift[sentiment_only_mask]
    return mdi
