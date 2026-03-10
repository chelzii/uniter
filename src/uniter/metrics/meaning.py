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
    drift[mask] = (1.0 - similarity) / max(normalizer, 1e-6)
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
    drift[mask] = torch.abs(current_scores - historical_scores) / max(
        normalizer * max(score_range, 1e-6),
        1e-6,
    )
    return drift


def compute_mdi(
    current_embeddings: torch.Tensor,
    historical_embeddings: torch.Tensor | None,
    *,
    normalizer: float,
    current_sentiment_logits: torch.Tensor | None = None,
    historical_sentiment_logits: torch.Tensor | None = None,
    historical_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if current_sentiment_logits is not None and historical_sentiment_logits is not None:
        return compute_sentiment_drift(
            current_sentiment_logits,
            historical_sentiment_logits,
            normalizer=normalizer,
            historical_mask=historical_mask,
        )

    return compute_embedding_drift(
        current_embeddings,
        historical_embeddings,
        normalizer=normalizer,
        historical_mask=historical_mask,
    )
