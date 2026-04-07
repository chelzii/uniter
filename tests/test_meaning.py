from __future__ import annotations

import torch

from uniter.metrics.meaning import compute_mdi


def test_compute_mdi_blends_embedding_and_sentiment_drift_when_logits_are_available() -> None:
    current_embeddings = torch.tensor([[1.0, 0.0]])
    historical_embeddings = torch.tensor([[0.0, 1.0]])
    current_logits = torch.tensor([[0.0, 0.0, 8.0]])
    historical_logits = torch.tensor([[8.0, 0.0, 0.0]])

    mdi = compute_mdi(
        current_embeddings,
        historical_embeddings,
        normalizer=1.0,
        sentiment_weight=0.7,
        current_sentiment_logits=current_logits,
        historical_sentiment_logits=historical_logits,
        historical_mask=torch.tensor([True]),
    )

    class_positions = torch.tensor([-1.0, 0.0, 1.0])
    sentiment_drift = torch.abs(
        torch.softmax(current_logits, dim=-1) @ class_positions
        - torch.softmax(historical_logits, dim=-1) @ class_positions
    ) / 2.0
    embedding_drift = torch.tensor([1.0])
    expected = embedding_drift * 0.3 + sentiment_drift * 0.7
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


def test_compute_mdi_uses_record_level_distribution_drift_when_available() -> None:
    current_embeddings = torch.tensor([[1.0, 0.0]])
    historical_embeddings = torch.tensor([[1.0, 0.0]])
    current_record_logits = torch.tensor(
        [
            [0.0, 0.0, 8.0],
            [8.0, 0.0, 0.0],
            [0.0, 8.0, 0.0],
        ]
    )
    historical_record_logits = torch.tensor(
        [
            [0.0, 0.0, 8.0],
            [0.0, 0.0, 8.0],
            [0.0, 8.0, 0.0],
        ]
    )

    mdi = compute_mdi(
        current_embeddings,
        historical_embeddings,
        normalizer=1.0,
        mode="sentiment_first",
        current_sentiment_record_logits=current_record_logits,
        current_sentiment_region_index=torch.tensor([0, 0, 0]),
        historical_sentiment_record_logits=historical_record_logits,
        historical_sentiment_region_index=torch.tensor([0, 0, 0]),
        historical_mask=torch.tensor([True]),
    )

    assert mdi.shape == torch.Size([1])
    assert float(mdi[0].item()) > 0.05
    assert float(mdi[0].item()) < 1.0


def test_compute_mdi_can_disable_sentiment_blending() -> None:
    current_embeddings = torch.tensor([[1.0, 0.0]])
    historical_embeddings = torch.tensor([[0.0, 1.0]])
    current_logits = torch.tensor([[0.0, 0.0, 8.0]])
    historical_logits = torch.tensor([[8.0, 0.0, 0.0]])

    mdi = compute_mdi(
        current_embeddings,
        historical_embeddings,
        normalizer=1.0,
        sentiment_weight=0.0,
        current_sentiment_logits=current_logits,
        historical_sentiment_logits=historical_logits,
        historical_mask=torch.tensor([True]),
    )

    assert torch.equal(mdi, torch.tensor([1.0]))


def test_compute_mdi_can_use_sentiment_first_mode() -> None:
    current_embeddings = torch.tensor([[1.0, 0.0]])
    historical_embeddings = torch.tensor([[0.0, 1.0]])
    current_logits = torch.tensor([[0.0, 0.0, 8.0]])
    historical_logits = torch.tensor([[8.0, 0.0, 0.0]])

    mdi = compute_mdi(
        current_embeddings,
        historical_embeddings,
        normalizer=1.0,
        mode="sentiment_first",
        current_sentiment_logits=current_logits,
        historical_sentiment_logits=historical_logits,
        historical_mask=torch.tensor([True]),
    )

    assert torch.allclose(mdi, torch.tensor([1.0]), atol=1e-3)


def test_compute_mdi_thesis_mode_preserves_narrative_embedding_drift() -> None:
    current_embeddings = torch.tensor([[1.0, 0.0]])
    historical_embeddings = torch.tensor([[0.0, 1.0]])
    current_logits = torch.tensor([[0.0, 0.0, 8.0]])
    historical_logits = torch.tensor([[8.0, 0.0, 0.0]])

    mdi = compute_mdi(
        current_embeddings,
        historical_embeddings,
        normalizer=1.0,
        mode="thesis",
        sentiment_weight=0.7,
        current_sentiment_logits=current_logits,
        historical_sentiment_logits=historical_logits,
        historical_mask=torch.tensor([True]),
    )

    assert float(mdi[0].item()) < 1.0
    assert float(mdi[0].item()) > 0.7


def test_compute_mdi_sentiment_only_mode_falls_back_when_sentiment_is_missing() -> None:
    current_embeddings = torch.tensor([[1.0, 0.0]])
    historical_embeddings = torch.tensor([[0.0, 1.0]])

    mdi = compute_mdi(
        current_embeddings,
        historical_embeddings,
        normalizer=1.0,
        mode="sentiment_only",
        historical_mask=torch.tensor([True]),
    )

    assert torch.equal(mdi, torch.tensor([1.0]))


def test_compute_mdi_is_clamped_to_unit_interval() -> None:
    current_embeddings = torch.tensor([[1.0, 0.0]])
    historical_embeddings = torch.tensor([[0.0, 1.0]])
    current_logits = torch.tensor([[0.0, 0.0, 8.0]])
    historical_logits = torch.tensor([[8.0, 0.0, 0.0]])

    mdi = compute_mdi(
        current_embeddings,
        historical_embeddings,
        normalizer=0.25,
        sentiment_weight=1.0,
        current_sentiment_logits=current_logits,
        historical_sentiment_logits=historical_logits,
        historical_mask=torch.tensor([True]),
    )

    assert torch.allclose(mdi, torch.tensor([1.0]))
