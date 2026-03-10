from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_iai(
    identity_embeddings: torch.Tensor | None,
    *,
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    normalizer: float,
    identity_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    drift = image_embeddings.new_full((image_embeddings.shape[0],), float("nan"))
    if identity_embeddings is None:
        return drift

    mask = (
        identity_mask.to(device=image_embeddings.device, dtype=torch.bool)
        if identity_mask is not None
        else torch.ones(image_embeddings.shape[0], device=image_embeddings.device, dtype=torch.bool)
    )
    if not torch.any(mask):
        return drift

    multimodal_center = F.normalize(image_embeddings[mask] + text_embeddings[mask], dim=-1)
    identity_center = F.normalize(identity_embeddings[mask], dim=-1)
    similarity = F.cosine_similarity(multimodal_center, identity_center, dim=-1)
    drift[mask] = (1.0 - similarity) / max(normalizer, 1e-6)
    return drift
