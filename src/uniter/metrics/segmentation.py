from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


@dataclass(slots=True)
class SegmentationAccumulator:
    num_classes: int
    ignore_index: int
    confusion: torch.Tensor = field(init=False)
    labeled_images: int = field(init=False, default=0)
    labeled_pixels: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.confusion = torch.zeros(
            self.num_classes,
            self.num_classes,
            dtype=torch.long,
        )
        self.labeled_images = 0
        self.labeled_pixels = 0

    def update(self, logits: torch.Tensor, targets: torch.Tensor | None) -> None:
        if targets is None:
            return
        if logits.shape[0] != targets.shape[0]:
            raise ValueError("Segmentation logits and targets must have the same batch dimension.")
        if logits.shape[-2:] != targets.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=targets.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        predictions = logits.argmax(dim=1)
        valid_mask = targets != self.ignore_index
        if not torch.any(valid_mask):
            return

        self.labeled_images += int(
            valid_mask.reshape(valid_mask.shape[0], -1).any(dim=1).sum().item()
        )
        self.labeled_pixels += int(valid_mask.sum().item())
        valid_predictions = predictions[valid_mask].to(dtype=torch.long)
        valid_targets = targets[valid_mask].to(dtype=torch.long)
        encoded = valid_targets * self.num_classes + valid_predictions
        counts = torch.bincount(
            encoded,
            minlength=self.num_classes * self.num_classes,
        ).reshape(self.num_classes, self.num_classes)
        self.confusion += counts.detach().cpu()

    def summarize(self, *, class_names: list[str]) -> dict[str, object]:
        if self.labeled_pixels == 0:
            return {
                "label_count": 0,
                "labeled_images": 0,
                "labeled_pixels": 0,
                "pixel_accuracy": None,
                "mIoU": None,
                "mean_dice": None,
                "per_class": [],
                "confusion_matrix": None,
            }

        confusion = self.confusion
        total_correct = int(torch.diag(confusion).sum().item())
        per_class: list[dict[str, object]] = []
        valid_ious: list[float] = []
        valid_dice: list[float] = []
        for index, label in enumerate(class_names):
            true_positive = int(confusion[index, index].item())
            false_positive = int(confusion[:, index].sum().item()) - true_positive
            false_negative = int(confusion[index, :].sum().item()) - true_positive
            union = true_positive + false_positive + false_negative
            support = int(confusion[index, :].sum().item())
            iou = (true_positive / union) if union > 0 else None
            dice_denominator = (2 * true_positive) + false_positive + false_negative
            dice = (2 * true_positive / dice_denominator) if dice_denominator > 0 else None
            if iou is not None:
                valid_ious.append(iou)
            if dice is not None:
                valid_dice.append(dice)
            per_class.append(
                {
                    "label": label,
                    "support": support,
                    "iou": iou,
                    "dice": dice,
                }
            )

        return {
            "label_count": self.labeled_pixels,
            "labeled_images": self.labeled_images,
            "labeled_pixels": self.labeled_pixels,
            "pixel_accuracy": total_correct / self.labeled_pixels,
            "mIoU": sum(valid_ious) / max(len(valid_ious), 1) if valid_ious else None,
            "mean_dice": sum(valid_dice) / max(len(valid_dice), 1) if valid_dice else None,
            "per_class": per_class,
            "confusion_matrix": {
                "labels": class_names,
                "matrix": confusion.tolist(),
            },
        }
