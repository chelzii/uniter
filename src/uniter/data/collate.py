from __future__ import annotations

import csv
from functools import lru_cache
import logging
import math
import random
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError
import torch

from uniter.data.dataset import RegionBatch
from uniter.data.manifest import RegionRecord
from uniter.models.huggingface import load_image_processor, load_tokenizer


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
        random_sample: bool = False,
    ) -> None:
        self.logger = logging.getLogger(__name__)
        self.image_processor = load_image_processor(spatial_model_name)
        self.tokenizer = load_tokenizer(text_model_name)
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
        self.random_sample = random_sample

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

    def _load_image(self, image_path: Path) -> Image.Image | None:
        try:
            with Image.open(image_path) as image:
                return image.convert("RGB").resize(
                    (self.image_size, self.image_size),
                    resample=Image.Resampling.BILINEAR,
                )
        except (UnidentifiedImageError, OSError) as exc:
            self.logger.warning("Skipping unreadable image %s: %s", image_path, exc)
            return None

    def _load_segmentation_mask(self, mask_path: Path | None) -> torch.Tensor:
        if mask_path is None:
            return torch.full(
                (self.image_size, self.image_size),
                fill_value=self.segmentation_ignore_index,
                dtype=torch.long,
            )
        try:
            with Image.open(mask_path) as mask_image:
                resized = mask_image.convert("L").resize(
                    (self.image_size, self.image_size),
                    resample=Image.Resampling.NEAREST,
                )
        except (UnidentifiedImageError, OSError) as exc:
            self.logger.warning(
                "Using ignore-only segmentation mask for unreadable file %s: %s",
                mask_path,
                exc,
            )
            return torch.full(
                (self.image_size, self.image_size),
                fill_value=self.segmentation_ignore_index,
                dtype=torch.long,
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

    def _select_items(self, items: list[Any], *, max_items: int) -> list[Any]:
        if len(items) <= max_items:
            return list(items)
        if not self.random_sample:
            return list(items[:max_items])
        selected_indices = sorted(random.sample(range(len(items)), k=max_items))
        return [items[index] for index in selected_indices]

    def _align_text_labels(
        self,
        texts: list[str],
        labels: list[int | None],
    ) -> list[int | None]:
        if not labels:
            return [None] * len(texts)
        aligned = list(labels[: len(texts)])
        if len(aligned) < len(texts):
            aligned.extend([None] * (len(texts) - len(aligned)))
        return aligned

    def _select_text_label_pairs(
        self,
        texts: list[str],
        labels: list[int | None],
        *,
        max_items: int,
    ) -> tuple[list[str], list[int | None]]:
        aligned_labels = self._align_text_labels(texts, labels)
        paired = list(zip(texts, aligned_labels, strict=True))
        selected = self._select_items(paired, max_items=max_items)
        return [text for text, _ in selected], [label for _, label in selected]

    def _infer_image_type(self, image_path: Path) -> str:
        name = image_path.name.strip().lower()
        if "_satellite." in name or name.endswith("_satellite") or "satellite" in name:
            return "satellite"
        return "street"

    def _workspace_root_from_image_path(self, image_path: Path) -> Path | None:
        current = image_path.resolve()
        for parent in (current.parent, *current.parents):
            if parent.name == "images":
                return parent.parent
        return None

    @lru_cache(maxsize=8)
    def _load_image_metadata_index(
        self,
        workspace_root: str,
    ) -> dict[tuple[str, str], dict[str, str]]:
        image_index_path = Path(workspace_root) / "images" / "image_index.csv"
        if not image_index_path.exists():
            return {}
        with image_index_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = [dict(row) for row in csv.DictReader(handle)]

        index: dict[tuple[str, str], dict[str, str]] = {}
        for row in rows:
            region_id = str(row.get("region_id", "")).strip()
            file_name = str(row.get("file_name", "")).strip()
            if region_id and file_name:
                index[(region_id, file_name)] = row
        return index

    def _lookup_image_metadata(self, image_path: Path) -> dict[str, str] | None:
        workspace_root = self._workspace_root_from_image_path(image_path)
        if workspace_root is None:
            return None
        image_dir_name = image_path.parent.name.strip()
        if not image_dir_name:
            return None
        index = self._load_image_metadata_index(str(workspace_root))
        return index.get((image_dir_name, image_path.name))

    def _parse_optional_float(self, value: object) -> float | None:
        if value in (None, ""):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _parse_image_identity(self, image_path: Path) -> tuple[str, str]:
        stem = image_path.stem.strip().lower()
        if "_" not in stem:
            return stem, "unknown"
        point_id, suffix = stem.rsplit("_", maxsplit=1)
        if suffix == "satellite":
            return point_id, "satellite"
        if suffix in {"north", "east", "south", "west"}:
            return point_id, suffix
        return point_id, "unknown"

    def _select_image_mask_pairs(
        self,
        image_mask_pairs: list[tuple[Path, Path | None]],
        *,
        max_items: int,
    ) -> list[tuple[Path, Path | None]]:
        if len(image_mask_pairs) <= max_items:
            return list(image_mask_pairs)
        if self.random_sample:
            return self._select_items(image_mask_pairs, max_items=max_items)

        street_pairs = [
            pair for pair in image_mask_pairs if self._infer_image_type(pair[0]) == "street"
        ]
        satellite_pairs = [
            pair for pair in image_mask_pairs if self._infer_image_type(pair[0]) == "satellite"
        ]
        if not street_pairs or not satellite_pairs or max_items <= 1:
            return list(image_mask_pairs[:max_items])

        selected: list[tuple[Path, Path | None]] = []
        satellite_quota = max(1, round(max_items * 0.25))
        satellite_quota = min(satellite_quota, len(satellite_pairs), max_items - 1)
        street_quota = min(len(street_pairs), max_items - satellite_quota)

        selected.extend(street_pairs[:street_quota])
        selected.extend(satellite_pairs[:satellite_quota])

        if len(selected) < max_items:
            used_ids = {id(pair) for pair in selected}
            for pair in image_mask_pairs:
                if id(pair) in used_ids:
                    continue
                selected.append(pair)
                if len(selected) >= max_items:
                    break

        selected_ids = {id(pair) for pair in selected}
        return [pair for pair in image_mask_pairs if id(pair) in selected_ids][:max_items]

    def _aggregate_region_class_target(
        self,
        labels: list[int | None],
        *,
        fallback: int | None,
    ) -> int | None:
        valid_labels = [int(label) for label in labels if label is not None]
        if valid_labels:
            return int(math.floor(sum(valid_labels) / len(valid_labels) + 0.5))
        return fallback

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
        image_is_satellite: list[bool] = []
        image_point_ids: list[str] = []
        image_view_directions: list[str] = []
        image_longitudes: list[float | None] = []
        image_latitudes: list[float | None] = []
        current_texts: list[str] = []
        current_region_index: list[int] = []
        current_sentiment_labels: list[int | None] = []
        historical_texts: list[str] = []
        historical_region_index: list[int] = []
        historical_sentiment_labels: list[int | None] = []
        identity_texts: list[str] = []
        identity_region_index: list[int] = []

        for sample in samples:
            image_mask_pairs = list(
                zip(
                    sample.image_paths,
                    sample.segmentation_mask_paths
                    if sample.segmentation_mask_paths
                    else [None] * len(sample.image_paths),
                    strict=True,
                )
            )
            selected_image_mask_pairs = self._select_image_mask_pairs(
                image_mask_pairs,
                max_items=self.max_images_per_region,
            )
            selected_images = [image_path for image_path, _ in selected_image_mask_pairs]
            selected_masks = [mask_path for _, mask_path in selected_image_mask_pairs]
            selected_current_texts, selected_current_sentiment_labels = self._select_text_label_pairs(
                sample.current_texts,
                sample.current_sentiment_labels,
                max_items=self.max_current_texts_per_region,
            )
            selected_historical_texts, selected_historical_sentiment_labels = (
                self._select_text_label_pairs(
                    sample.historical_texts,
                    sample.historical_sentiment_labels,
                    max_items=self.max_historical_texts_per_region,
                )
            )
            selected_identity_texts = self._select_items(
                sample.identity_texts,
                max_items=self.max_identity_texts_per_region,
            )

            if not selected_images:
                raise ValueError(f"Region {sample.region_id} has no images.")
            if not selected_current_texts:
                raise ValueError(f"Region {sample.region_id} has no current texts.")

            sample_images: list[Image.Image] = []
            sample_segmentation_labels: list[torch.Tensor] = []
            sample_has_segmentation_supervision = False
            bad_image_skip_count = 0
            used_image_filenames: list[str] = []
            used_image_types: list[str] = []
            used_point_ids: list[str] = []
            used_view_directions: list[str] = []
            used_longitudes: list[float | None] = []
            used_latitudes: list[float | None] = []

            for image_path, mask_path in zip(selected_images, selected_masks, strict=True):
                image = self._load_image(image_path)
                if image is None:
                    bad_image_skip_count += 1
                    continue
                image_metadata = self._lookup_image_metadata(image_path)
                sample_images.append(image)
                used_image_filenames.append(image_path.name)
                image_type = (
                    str(image_metadata.get("image_type", "")).strip().lower()
                    if image_metadata is not None
                    else ""
                )
                if image_type not in {"street", "satellite"}:
                    image_type = self._infer_image_type(image_path)
                used_image_types.append(image_type)
                point_id, view_direction = self._parse_image_identity(image_path)
                if image_metadata is not None:
                    point_id = str(image_metadata.get("point_id", "")).strip().lower() or point_id
                    view_direction = (
                        str(image_metadata.get("view_direction", "")).strip().lower() or view_direction
                    )
                used_point_ids.append(point_id)
                used_view_directions.append(view_direction)
                used_longitudes.append(
                    self._parse_optional_float(
                        image_metadata.get("lon") if image_metadata is not None else None
                    )
                )
                used_latitudes.append(
                    self._parse_optional_float(
                        image_metadata.get("lat") if image_metadata is not None else None
                    )
                )
                sample_segmentation_labels.append(self._load_segmentation_mask(mask_path))
                sample_has_segmentation_supervision = (
                    sample_has_segmentation_supervision or mask_path is not None
                )

            if not sample_images:
                self.logger.warning(
                    "Skipping region %s because all selected images are unreadable.",
                    sample.region_id,
                )
                continue

            region_index = len(region_ids)
            region_ids.append(sample.region_id)
            sample_metadata = dict(sample.metadata)
            sample_metadata["image_count"] = len(sample.image_paths)
            sample_metadata["current_text_count"] = len(sample.current_texts)
            sample_metadata["historical_text_count"] = len(sample.historical_texts)
            sample_metadata["identity_text_count"] = len(sample.identity_texts)
            sample_metadata["selected_image_count"] = len(sample_images)
            sample_metadata["_selected_image_filenames_full"] = [
                image_path.name for image_path in selected_images
            ]
            sample_metadata["_used_image_filenames"] = used_image_filenames
            sample_metadata["_used_image_types"] = list(used_image_types)
            sample_metadata["selected_satellite_image_count"] = sum(
                image_type == "satellite" for image_type in used_image_types
            )
            sample_metadata["selected_street_image_count"] = sum(
                image_type == "street" for image_type in used_image_types
            )
            sample_metadata["_selected_current_texts"] = list(selected_current_texts)
            sample_metadata["_selected_historical_texts"] = list(selected_historical_texts)
            sample_metadata["_selected_identity_texts"] = list(selected_identity_texts)
            sample_metadata["_used_point_ids"] = list(used_point_ids)
            sample_metadata["_used_view_directions"] = list(used_view_directions)
            sample_metadata["_used_longitudes"] = list(used_longitudes)
            sample_metadata["_used_latitudes"] = list(used_latitudes)
            sample_metadata["bad_image_skip_count"] = bad_image_skip_count
            region_metadata.append(sample_metadata)
            region_targets["lost_space_label"].append(sample.targets.lost_space_label)
            region_targets["sentiment_label"].append(
                self._aggregate_region_class_target(
                    selected_current_sentiment_labels,
                    fallback=sample.targets.sentiment_label,
                )
            )
            region_targets["historical_sentiment_label"].append(
                self._aggregate_region_class_target(
                    selected_historical_sentiment_labels,
                    fallback=sample.targets.historical_sentiment_label,
                )
            )
            region_targets["ifi"].append(sample.targets.ifi)
            region_targets["mdi"].append(sample.targets.mdi)
            region_targets["iai"].append(sample.targets.iai)

            flat_images.extend(sample_images)
            segmentation_labels.extend(sample_segmentation_labels)
            image_region_index.extend([region_index] * len(sample_images))
            image_is_satellite.extend([image_type == "satellite" for image_type in used_image_types])
            image_point_ids.extend(used_point_ids)
            image_view_directions.extend(used_view_directions)
            image_longitudes.extend(used_longitudes)
            image_latitudes.extend(used_latitudes)
            has_segmentation_supervision = (
                has_segmentation_supervision or sample_has_segmentation_supervision
            )

            current_texts.extend(selected_current_texts)
            current_region_index.extend([region_index] * len(selected_current_texts))
            current_sentiment_labels.extend(selected_current_sentiment_labels)

            historical_texts.extend(selected_historical_texts)
            historical_region_index.extend([region_index] * len(selected_historical_texts))
            historical_sentiment_labels.extend(selected_historical_sentiment_labels)

            identity_texts.extend(selected_identity_texts)
            identity_region_index.extend([region_index] * len(selected_identity_texts))

        if not region_ids:
            raise ValueError("All samples in the batch were skipped because their images are unreadable.")

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
            image_is_satellite=torch.tensor(image_is_satellite, dtype=torch.bool),
            image_point_ids=image_point_ids,
            image_view_directions=image_view_directions,
            image_longitudes=image_longitudes,
            image_latitudes=image_latitudes,
            current_input_ids=current_tokens["input_ids"],
            current_attention_mask=current_tokens["attention_mask"],
            current_region_index=torch.tensor(current_region_index, dtype=torch.long),
            current_sentiment_labels=self._encode_integer_targets(current_sentiment_labels),
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
            historical_sentiment_labels=(
                self._encode_integer_targets(historical_sentiment_labels)
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
