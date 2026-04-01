from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoTokenizer

from uniter.data.dataset import RegionBatch
from uniter.data.manifest import RegionRecord


class RegionBatchCollator:
    """Build a region batch from multiple images and multiple texts."""

    def __init__(
        self,
        *,
        spatial_model_name: str,
        text_model_name: str,
        image_size: int,
        max_length: int,
        max_images_per_region: int,
        max_current_texts_per_region: int,
        max_historical_texts_per_region: int,
        max_identity_texts_per_region: int,
        sentiment_ignore_index: int,
        lost_space_ignore_index: int,
        segmentation_ignore_index: int,
        segmentation_label_mapping: dict[str, int] | None = None,
    ) -> None:
        self.image_processor = AutoImageProcessor.from_pretrained(spatial_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.image_size = image_size
        self.max_length = max_length
        self.max_images_per_region = max_images_per_region
        self.max_current_texts_per_region = max_current_texts_per_region
        self.max_historical_texts_per_region = max_historical_texts_per_region
        self.max_identity_texts_per_region = max_identity_texts_per_region
        self.sentiment_ignore_index = sentiment_ignore_index
        self.lost_space_ignore_index = lost_space_ignore_index
        self.segmentation_ignore_index = segmentation_ignore_index
        self.segmentation_label_mapping = self._normalize_segmentation_label_mapping(
            segmentation_label_mapping or {}
        )

    def _normalize_segmentation_label_mapping(
        self,
        label_mapping: dict[str, int],
    ) -> dict[int, int]:
        normalized: dict[int, int] = {}
        for raw_label, mapped_label in label_mapping.items():
            try:
                normalized[int(raw_label)] = int(mapped_label)
            except ValueError as exc:
                raise ValueError(
                    "spatial_supervision.label_mapping keys must be integer-like strings."
                ) from exc
        return normalized

    def _load_image(self, image_path: Path) -> Image.Image:
        with Image.open(image_path) as image:
            return image.convert("RGB").resize(
                (self.image_size, self.image_size),
                resample=Image.Resampling.BILINEAR,
            )

    def _load_segmentation_mask(self, mask_path: Path | None) -> torch.Tensor:
        if mask_path is None:
            return torch.full(
                (self.image_size, self.image_size),
                fill_value=self.segmentation_ignore_index,
                dtype=torch.long,
            )
        with Image.open(mask_path) as mask_image:
            resized = mask_image.convert("L").resize(
                (self.image_size, self.image_size),
                resample=Image.Resampling.NEAREST,
            )
        mask = torch.tensor(list(resized.getdata()), dtype=torch.long).reshape(
            self.image_size,
            self.image_size,
        )
        return self._remap_segmentation_mask(mask)

    def _remap_segmentation_mask(self, mask: torch.Tensor) -> torch.Tensor:
        if not self.segmentation_label_mapping:
            return mask
        remapped = torch.full_like(mask, fill_value=self.segmentation_ignore_index)
        for raw_label, mapped_label in self.segmentation_label_mapping.items():
            remapped[mask == raw_label] = mapped_label
        return remapped

    def _tokenize_texts(self, texts: list[str]) -> dict[str, torch.Tensor]:
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def _encode_integer_targets(self, values: list[int | None]) -> torch.Tensor:
        return self._encode_integer_targets_with_ignore(
            values,
            ignore_index=self.sentiment_ignore_index,
        )

    def _encode_integer_targets_with_ignore(
        self,
        values: list[int | None],
        *,
        ignore_index: int,
    ) -> torch.Tensor:
        encoded = [
            ignore_index if value is None else int(value)
            for value in values
        ]
        return torch.tensor(encoded, dtype=torch.long)

    def _encode_float_targets(self, values: list[float | int | None]) -> torch.Tensor:
        encoded = [float("nan") if value is None else float(value) for value in values]
        return torch.tensor(encoded, dtype=torch.float32)

    def __call__(self, samples: list[RegionRecord]) -> RegionBatch:
        region_ids: list[str] = []
        region_metadata: list[dict[str, Any]] = []
        region_targets: dict[str, list[Any]] = {
            "lost_space_label": [],
            "sentiment_label": [],
            "historical_sentiment_label": [],
            "ifi": [],
            "mdi": [],
            "iai": [],
        }

        flat_images: list[Image.Image] = []
        segmentation_labels: list[torch.Tensor] = []
        has_segmentation_supervision = False
        image_region_index: list[int] = []
        current_texts: list[str] = []
        current_region_index: list[int] = []
        historical_texts: list[str] = []
        historical_region_index: list[int] = []
        identity_texts: list[str] = []
        identity_region_index: list[int] = []

        for region_index, sample in enumerate(samples):
            region_ids.append(sample.region_id)
            region_metadata.append(sample.metadata)
            region_targets["lost_space_label"].append(sample.targets.lost_space_label)
            region_targets["sentiment_label"].append(sample.targets.sentiment_label)
            region_targets["historical_sentiment_label"].append(
                sample.targets.historical_sentiment_label
            )
            region_targets["ifi"].append(sample.targets.ifi)
            region_targets["mdi"].append(sample.targets.mdi)
            region_targets["iai"].append(sample.targets.iai)

            selected_images = sample.image_paths[: self.max_images_per_region]
            selected_masks = (
                sample.segmentation_mask_paths[: self.max_images_per_region]
                if sample.segmentation_mask_paths
                else [None] * len(selected_images)
            )
            selected_current_texts = sample.current_texts[: self.max_current_texts_per_region]
            selected_historical_texts = sample.historical_texts[
                : self.max_historical_texts_per_region
            ]
            selected_identity_texts = sample.identity_texts[: self.max_identity_texts_per_region]

            if not selected_images:
                raise ValueError(f"Region {sample.region_id} has no images.")
            if not selected_current_texts:
                raise ValueError(f"Region {sample.region_id} has no current texts.")

            for image_path, mask_path in zip(selected_images, selected_masks, strict=True):
                flat_images.append(self._load_image(image_path))
                segmentation_labels.append(self._load_segmentation_mask(mask_path))
                image_region_index.append(region_index)
                has_segmentation_supervision = has_segmentation_supervision or mask_path is not None

            current_texts.extend(selected_current_texts)
            current_region_index.extend([region_index] * len(selected_current_texts))

            historical_texts.extend(selected_historical_texts)
            historical_region_index.extend([region_index] * len(selected_historical_texts))

            identity_texts.extend(selected_identity_texts)
            identity_region_index.extend([region_index] * len(selected_identity_texts))

        pixel_batch = self.image_processor(images=flat_images, return_tensors="pt")
        current_tokens = self._tokenize_texts(current_texts)

        historical_tokens: dict[str, torch.Tensor] | None = None
        if historical_texts:
            historical_tokens = self._tokenize_texts(historical_texts)

        identity_tokens: dict[str, torch.Tensor] | None = None
        if identity_texts:
            identity_tokens = self._tokenize_texts(identity_texts)

        return RegionBatch(
            region_ids=region_ids,
            pixel_values=pixel_batch["pixel_values"],
            segmentation_labels=(
                torch.stack(segmentation_labels, dim=0) if has_segmentation_supervision else None
            ),
            image_region_index=torch.tensor(image_region_index, dtype=torch.long),
            current_input_ids=current_tokens["input_ids"],
            current_attention_mask=current_tokens["attention_mask"],
            current_region_index=torch.tensor(current_region_index, dtype=torch.long),
            historical_input_ids=(
                historical_tokens["input_ids"] if historical_tokens is not None else None
            ),
            historical_attention_mask=(
                historical_tokens["attention_mask"] if historical_tokens is not None else None
            ),
            historical_region_index=(
                torch.tensor(historical_region_index, dtype=torch.long)
                if historical_tokens is not None
                else None
            ),
            identity_input_ids=(
                identity_tokens["input_ids"] if identity_tokens is not None else None
            ),
            identity_attention_mask=(
                identity_tokens["attention_mask"] if identity_tokens is not None else None
            ),
            identity_region_index=(
                torch.tensor(identity_region_index, dtype=torch.long)
                if identity_tokens is not None
                else None
            ),
            metadata=region_metadata,
            targets={
                "lost_space_label": self._encode_integer_targets_with_ignore(
                    region_targets["lost_space_label"],
                    ignore_index=self.lost_space_ignore_index,
                ),
                "sentiment_label": self._encode_integer_targets(
                    region_targets["sentiment_label"]
                ),
                "historical_sentiment_label": self._encode_integer_targets(
                    region_targets["historical_sentiment_label"]
                ),
                "ifi": self._encode_float_targets(region_targets["ifi"]),
                "mdi": self._encode_float_targets(region_targets["mdi"]),
                "iai": self._encode_float_targets(region_targets["iai"]),
            },
        )
