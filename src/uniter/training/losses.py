from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(slots=True)
class ClassificationLossResult:
    loss: torch.Tensor
    valid_count: int


@dataclass(slots=True)
class SegmentationLossResult:
    loss: torch.Tensor
    labeled_images: int
    labeled_pixels: int


def symmetric_contrastive_loss(
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    if image_embeddings.shape[0] != text_embeddings.shape[0]:
        raise ValueError("Image and text embeddings must have the same batch size.")
    if image_embeddings.shape[0] < 2:
        return image_embeddings.new_tensor(0.0)

    logits = (image_embeddings @ text_embeddings.T) / temperature
    targets = torch.arange(logits.shape[0], device=logits.device)
    loss_i2t = F.cross_entropy(logits, targets)
    loss_t2i = F.cross_entropy(logits.T, targets)
    return 0.5 * (loss_i2t + loss_t2i)


def masked_cross_entropy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    ignore_index: int,
    label_smoothing: float = 0.0,
) -> ClassificationLossResult:
    valid_mask = targets != ignore_index
    if not torch.any(valid_mask):
        return ClassificationLossResult(loss=logits.new_tensor(0.0), valid_count=0)

    valid_targets = targets[valid_mask]
    if torch.any(valid_targets < 0) or torch.any(valid_targets >= logits.shape[-1]):
        raise ValueError(
            "Sentiment labels must be within the range "
            f"[0, {logits.shape[-1] - 1}] or equal to ignore_index."
        )

    loss = F.cross_entropy(
        logits[valid_mask],
        valid_targets,
        label_smoothing=label_smoothing,
    )
    return ClassificationLossResult(loss=loss, valid_count=int(valid_mask.sum().item()))


def segmentation_cross_entropy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor | None,
    *,
    ignore_index: int,
) -> SegmentationLossResult:
    if targets is None:
        return SegmentationLossResult(
            loss=logits.new_tensor(0.0),
            labeled_images=0,
            labeled_pixels=0,
        )

    if logits.shape[0] != targets.shape[0]:
        raise ValueError("Segmentation logits and targets must have the same batch dimension.")

    if logits.shape[-2:] != targets.shape[-2:]:
        logits = F.interpolate(
            logits,
            size=targets.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    valid_mask = targets != ignore_index
    if not torch.any(valid_mask):
        return SegmentationLossResult(
            loss=logits.new_tensor(0.0),
            labeled_images=0,
            labeled_pixels=0,
        )

    valid_targets = targets[valid_mask]
    if torch.any(valid_targets < 0) or torch.any(valid_targets >= logits.shape[1]):
        raise ValueError(
            "Segmentation labels must be within the range "
            f"[0, {logits.shape[1] - 1}] or equal to ignore_index."
        )

    loss = F.cross_entropy(logits, targets, ignore_index=ignore_index)
    labeled_images = int((valid_mask.reshape(valid_mask.shape[0], -1).any(dim=1)).sum().item())
    labeled_pixels = int(valid_mask.sum().item())
    return SegmentationLossResult(
        loss=loss,
        labeled_images=labeled_images,
        labeled_pixels=labeled_pixels,
    )


def masked_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    ignore_index: int,
) -> tuple[int, int]:
    valid_mask = targets != ignore_index
    if not torch.any(valid_mask):
        return 0, 0

    predictions = logits.argmax(dim=-1)
    correct = int((predictions[valid_mask] == targets[valid_mask]).sum().item())
    total = int(valid_mask.sum().item())
    return correct, total
