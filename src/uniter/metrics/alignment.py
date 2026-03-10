from __future__ import annotations

import torch
import torch.nn.functional as F


def cosine_alignment_gap(
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
) -> torch.Tensor:
    similarity = F.cosine_similarity(image_embeddings, text_embeddings, dim=-1)
    return 1.0 - similarity
