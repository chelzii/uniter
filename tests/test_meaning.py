from __future__ import annotations

import torch

from uniter.metrics.meaning import compute_mdi


def test_compute_mdi_prefers_sentiment_drift_when_logits_are_available() -> None:
    current_embeddings = torch.tensor([[1.0, 0.0]])
    historical_embeddings = torch.tensor([[0.0, 1.0]])
    current_logits = torch.tensor([[0.0, 0.0, 8.0]])
    historical_logits = torch.tensor([[8.0, 0.0, 0.0]])

    mdi = compute_mdi(
        current_embeddings,
        historical_embeddings,
        normalizer=1.0,
        current_sentiment_logits=current_logits,
        historical_sentiment_logits=historical_logits,
        historical_mask=torch.tensor([True]),
    )

    class_positions = torch.tensor([-1.0, 0.0, 1.0])
    expected = torch.abs(
        torch.softmax(current_logits, dim=-1) @ class_positions
        - torch.softmax(historical_logits, dim=-1) @ class_positions
    ) / 2.0
    assert torch.allclose(mdi, expected, atol=1e-6)


def test_compute_mdi_falls_back_to_embedding_drift_without_sentiment_logits() -> None:
    current_embeddings = torch.tensor([[1.0, 0.0]])
    historical_embeddings = torch.tensor([[0.0, 1.0]])

    mdi = compute_mdi(
        current_embeddings,
        historical_embeddings,
        normalizer=1.0,
        historical_mask=torch.tensor([True]),
    )

    assert torch.equal(mdi, torch.tensor([1.0]))


def test_compute_mdi_marks_regions_without_historical_text_as_nan() -> None:
    current_embeddings = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    historical_embeddings = torch.tensor([[0.0, 1.0], [0.0, 0.0]])
    current_logits = torch.tensor([[0.0, 0.0, 8.0], [8.0, 0.0, 0.0]])
    historical_logits = torch.tensor([[8.0, 0.0, 0.0], [0.0, 8.0, 0.0]])

    mdi = compute_mdi(
        current_embeddings,
        historical_embeddings,
        normalizer=1.0,
        current_sentiment_logits=current_logits,
        historical_sentiment_logits=historical_logits,
        historical_mask=torch.tensor([True, False]),
    )

    class_positions = torch.tensor([-1.0, 0.0, 1.0])
    expected = torch.abs(
        torch.softmax(current_logits[:1], dim=-1) @ class_positions
        - torch.softmax(historical_logits[:1], dim=-1) @ class_positions
    ) / 2.0
    assert torch.allclose(mdi[:1], expected, atol=1e-6)
    assert torch.isnan(mdi[1])
