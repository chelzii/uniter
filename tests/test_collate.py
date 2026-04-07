from __future__ import annotations

from pathlib import Path
import csv

from PIL import Image
import torch

from uniter.data.collate import RegionBatchCollator
from uniter.data.manifest import RegionRecord, RegionTargets


class _FakeImageProcessor:
    def __call__(self, *, images: list[Image.Image], return_tensors: str) -> dict[str, torch.Tensor]:
        assert return_tensors == "pt"
        return {"pixel_values": torch.zeros((len(images), 3, 8, 8), dtype=torch.float32)}


class _FakeTokenizer:
    def __call__(
        self,
        texts: list[str],
        *,
        padding: bool,
        truncation: bool,
        max_length: int,
        return_tensors: str,
    ) -> dict[str, torch.Tensor]:
        assert padding is True
        assert truncation is True
        assert max_length == 16
        assert return_tensors == "pt"
        batch = len(texts)
        return {
            "input_ids": torch.ones((batch, 4), dtype=torch.long),
            "attention_mask": torch.ones((batch, 4), dtype=torch.long),
        }


def test_collator_skips_unreadable_images_but_keeps_valid_ones(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "uniter.data.collate.load_image_processor",
        lambda _: _FakeImageProcessor(),
    )
    monkeypatch.setattr(
        "uniter.data.collate.load_tokenizer",
        lambda _: _FakeTokenizer(),
    )

    valid_image_path = tmp_path / "ok.png"
    Image.new("RGB", (4, 4), color=(255, 0, 0)).save(valid_image_path)
    broken_image_path = tmp_path / "broken.tif"
    broken_image_path.write_bytes(b"not-a-real-image")

    collator = RegionBatchCollator(
        spatial_model_name="fake-spatial",
        text_model_name="fake-text",
        image_size=8,
        max_length=16,
        max_images_per_region=4,
        max_current_texts_per_region=4,
        max_historical_texts_per_region=4,
        max_identity_texts_per_region=4,
        sentiment_ignore_index=-100,
        lost_space_ignore_index=-100,
        segmentation_ignore_index=255,
    )

    batch = collator(
        [
            RegionRecord(
                region_id="region-a",
                split="train",
                image_paths=[valid_image_path, broken_image_path],
                current_texts=["current text"],
                targets=RegionTargets(),
            )
        ]
    )

    assert batch.region_ids == ["region-a"]
    assert batch.pixel_values.shape == (1, 3, 8, 8)
    assert batch.image_region_index.tolist() == [0]
    assert batch.image_is_satellite.tolist() == [False]
    assert batch.image_point_ids == ["ok"]
    assert batch.image_view_directions == ["unknown"]
    assert batch.image_longitudes == [None]
    assert batch.image_latitudes == [None]
    assert batch.current_region_index.tolist() == [0]


def test_collator_random_sampling_is_only_used_when_enabled(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "uniter.data.collate.load_image_processor",
        lambda _: _FakeImageProcessor(),
    )
    monkeypatch.setattr(
        "uniter.data.collate.load_tokenizer",
        lambda _: _FakeTokenizer(),
    )

    image_paths: list[Path] = []
    for index in range(4):
        image_path = tmp_path / f"image_{index}.png"
        Image.new("RGB", (4, 4), color=(index, 0, 0)).save(image_path)
        image_paths.append(image_path)

    record = RegionRecord(
        region_id="region-a",
        split="train",
        image_paths=image_paths,
        current_texts=["current-0", "current-1", "current-2", "current-3"],
        targets=RegionTargets(),
    )

    deterministic_collator = RegionBatchCollator(
        spatial_model_name="fake-spatial",
        text_model_name="fake-text",
        image_size=8,
        max_length=16,
        max_images_per_region=2,
        max_current_texts_per_region=2,
        max_historical_texts_per_region=4,
        max_identity_texts_per_region=4,
        sentiment_ignore_index=-100,
        lost_space_ignore_index=-100,
        segmentation_ignore_index=255,
        random_sample=False,
    )
    deterministic_batch = deterministic_collator([record])
    assert deterministic_batch.metadata[0]["_used_image_filenames"] == ["image_0.png", "image_1.png"]
    assert deterministic_batch.metadata[0]["_selected_current_texts"] == ["current-0", "current-1"]

    monkeypatch.setattr(
        "uniter.data.collate.random.sample",
        lambda population, k: [3, 1],
    )
    random_collator = RegionBatchCollator(
        spatial_model_name="fake-spatial",
        text_model_name="fake-text",
        image_size=8,
        max_length=16,
        max_images_per_region=2,
        max_current_texts_per_region=2,
        max_historical_texts_per_region=4,
        max_identity_texts_per_region=4,
        sentiment_ignore_index=-100,
        lost_space_ignore_index=-100,
        segmentation_ignore_index=255,
        random_sample=True,
    )
    random_batch = random_collator([record])
    assert random_batch.metadata[0]["_used_image_filenames"] == ["image_1.png", "image_3.png"]
    assert random_batch.metadata[0]["_selected_current_texts"] == ["current-1", "current-3"]


def test_collator_keeps_satellite_images_in_deterministic_sampling(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "uniter.data.collate.load_image_processor",
        lambda _: _FakeImageProcessor(),
    )
    monkeypatch.setattr(
        "uniter.data.collate.load_tokenizer",
        lambda _: _FakeTokenizer(),
    )

    street_path = tmp_path / "p001_north.png"
    street_east_path = tmp_path / "p001_east.png"
    satellite_path = tmp_path / "p001_satellite.tif"
    Image.new("RGB", (4, 4), color=(255, 0, 0)).save(street_path)
    Image.new("RGB", (4, 4), color=(0, 0, 255)).save(street_east_path)
    Image.new("RGB", (4, 4), color=(0, 255, 0)).save(satellite_path)

    collator = RegionBatchCollator(
        spatial_model_name="fake-spatial",
        text_model_name="fake-text",
        image_size=8,
        max_length=16,
        max_images_per_region=2,
        max_current_texts_per_region=2,
        max_historical_texts_per_region=2,
        max_identity_texts_per_region=1,
        sentiment_ignore_index=-100,
        lost_space_ignore_index=-100,
        segmentation_ignore_index=255,
        random_sample=False,
    )

    batch = collator(
        [
            RegionRecord(
                region_id="region-a",
                split="train",
                image_paths=[street_path, street_east_path, satellite_path],
                current_texts=["current-0"],
                targets=RegionTargets(),
            )
        ]
    )

    assert batch.image_is_satellite.tolist() == [False, True]
    assert batch.image_point_ids == ["p001", "p001"]
    assert batch.image_view_directions == ["north", "satellite"]
    assert batch.metadata[0]["selected_street_image_count"] == 1
    assert batch.metadata[0]["selected_satellite_image_count"] == 1


def test_collator_preserves_record_level_sentiment_labels(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "uniter.data.collate.load_image_processor",
        lambda _: _FakeImageProcessor(),
    )
    monkeypatch.setattr(
        "uniter.data.collate.load_tokenizer",
        lambda _: _FakeTokenizer(),
    )

    image_path = tmp_path / "image.png"
    Image.new("RGB", (4, 4), color=(255, 0, 0)).save(image_path)

    collator = RegionBatchCollator(
        spatial_model_name="fake-spatial",
        text_model_name="fake-text",
        image_size=8,
        max_length=16,
        max_images_per_region=1,
        max_current_texts_per_region=3,
        max_historical_texts_per_region=2,
        max_identity_texts_per_region=1,
        sentiment_ignore_index=-100,
        lost_space_ignore_index=-100,
        segmentation_ignore_index=255,
    )
    batch = collator(
        [
            RegionRecord(
                region_id="region-a",
                split="train",
                image_paths=[image_path],
                current_texts=["current-0", "current-1", "current-2"],
                current_sentiment_labels=[2, None, 1],
                historical_texts=["historical-0", "historical-1"],
                historical_sentiment_labels=[0, 2],
                targets=RegionTargets(),
            )
        ]
    )

    assert batch.current_sentiment_labels.tolist() == [2, -100, 1]
    assert batch.historical_sentiment_labels.tolist() == [0, 2]
    assert batch.metadata[0]["_selected_historical_texts"] == ["historical-0", "historical-1"]
    assert batch.targets["sentiment_label"].tolist() == [2]
    assert batch.targets["historical_sentiment_label"].tolist() == [1]


def test_collator_prefers_image_index_metadata_when_available(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "uniter.data.collate.load_image_processor",
        lambda _: _FakeImageProcessor(),
    )
    monkeypatch.setattr(
        "uniter.data.collate.load_tokenizer",
        lambda _: _FakeTokenizer(),
    )

    workspace_root = tmp_path / "workspace"
    image_root = workspace_root / "images" / "kaitong_west_lane"
    image_root.mkdir(parents=True)
    image_path = image_root / "p001_satellite.tif"
    Image.new("RGB", (4, 4), color=(0, 255, 0)).save(image_path)

    image_index_path = workspace_root / "images" / "image_index.csv"
    with image_index_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "region_id",
                "point_id",
                "file_name",
                "image_type",
                "view_direction",
                "lon",
                "lat",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "region_id": "kaitong_west_lane",
                "point_id": "p001",
                "file_name": "p001_satellite.tif",
                "image_type": "satellite",
                "view_direction": "top",
                "lon": "108.95",
                "lat": "34.25",
            }
        )

    collator = RegionBatchCollator(
        spatial_model_name="fake-spatial",
        text_model_name="fake-text",
        image_size=8,
        max_length=16,
        max_images_per_region=1,
        max_current_texts_per_region=1,
        max_historical_texts_per_region=1,
        max_identity_texts_per_region=1,
        sentiment_ignore_index=-100,
        lost_space_ignore_index=-100,
        segmentation_ignore_index=255,
    )
    batch = collator(
        [
            RegionRecord(
                region_id="region-a",
                split="train",
                image_paths=[image_path],
                current_texts=["current-0"],
                targets=RegionTargets(),
            )
        ]
    )

    assert batch.image_is_satellite.tolist() == [True]
    assert batch.image_point_ids == ["p001"]
    assert batch.image_view_directions == ["top"]
    assert batch.image_longitudes == [108.95]
    assert batch.image_latitudes == [34.25]
