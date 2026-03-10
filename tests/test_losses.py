from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from uniter.training.losses import (
    masked_accuracy,
    masked_cross_entropy_loss,
    segmentation_cross_entropy_loss,
)


def test_masked_cross_entropy_only_uses_labeled_rows() -> None:
    logits = torch.tensor(
        [
            [5.0, 0.1, 0.1],
            [0.2, 2.5, 0.1],
            [0.3, 0.4, 1.8],
        ]
    )
    targets = torch.tensor([-100, 1, -100])

    result = masked_cross_entropy_loss(
        logits,
        targets,
        ignore_index=-100,
        label_smoothing=0.0,
    )

    expected = F.cross_entropy(logits[1:2], torch.tensor([1]))
    assert result.valid_count == 1
    assert torch.allclose(result.loss, expected)


def test_masked_cross_entropy_returns_zero_without_labels() -> None:
    logits = torch.tensor([[0.1, 0.2, 0.3]])
    targets = torch.tensor([-100])

    result = masked_cross_entropy_loss(
        logits,
        targets,
        ignore_index=-100,
        label_smoothing=0.0,
    )

    assert result.valid_count == 0
    assert torch.equal(result.loss, torch.tensor(0.0))


def test_masked_cross_entropy_rejects_out_of_range_labels() -> None:
    logits = torch.tensor([[0.1, 0.2, 0.3]])
    targets = torch.tensor([3])

    with pytest.raises(ValueError, match="Sentiment labels must be within the range"):
        masked_cross_entropy_loss(
            logits,
            targets,
            ignore_index=-100,
            label_smoothing=0.0,
        )


def test_masked_accuracy_ignores_unlabeled_rows() -> None:
    logits = torch.tensor(
        [
            [5.0, 0.1, 0.1],
            [0.2, 2.5, 0.1],
            [0.3, 0.4, 1.8],
        ]
    )
    targets = torch.tensor([0, -100, 1])

    correct, total = masked_accuracy(logits, targets, ignore_index=-100)

    assert correct == 1
    assert total == 2


def test_segmentation_cross_entropy_uses_ignore_index() -> None:
    logits = torch.tensor(
        [
            [
                [[4.0, 0.2], [0.1, 0.3]],
                [[0.1, 2.5], [2.1, 0.4]],
            ]
        ]
    )
    targets = torch.tensor([[[0, 1], [1, 255]]])

    result = segmentation_cross_entropy_loss(logits, targets, ignore_index=255)

    expected = F.cross_entropy(logits, targets, ignore_index=255)
    assert torch.allclose(result.loss, expected)
    assert result.labeled_images == 1
    assert result.labeled_pixels == 3


def test_segmentation_cross_entropy_returns_zero_without_labels() -> None:
    logits = torch.zeros(1, 2, 2, 2)

    result = segmentation_cross_entropy_loss(logits, None, ignore_index=255)

    assert torch.equal(result.loss, torch.tensor(0.0))
    assert result.labeled_images == 0
    assert result.labeled_pixels == 0
