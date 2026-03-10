from __future__ import annotations

import torch

from uniter.metrics.segmentation import SegmentationAccumulator


def test_segmentation_accumulator_summarizes_iou_and_dice() -> None:
    accumulator = SegmentationAccumulator(num_classes=2, ignore_index=255)
    logits = torch.tensor(
        [
            [
                [[2.0, 0.1], [0.2, 2.0]],
                [[0.1, 2.0], [2.0, 0.1]],
            ]
        ]
    )
    targets = torch.tensor([[[0, 1], [1, 0]]])

    accumulator.update(logits, targets)
    summary = accumulator.summarize(class_names=["road", "building"])

    assert summary["label_count"] == 4
    assert summary["pixel_accuracy"] == 1.0
    assert summary["mIoU"] == 1.0
    assert summary["mean_dice"] == 1.0
