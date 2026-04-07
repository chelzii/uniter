from __future__ import annotations

import json
from pathlib import Path

from uniter.data.tools import build_manifest_from_directories, summarize_manifest


def test_build_manifest_from_directories_creates_records_and_summary(tmp_path: Path) -> None:
    image_dir = tmp_path / "images" / "region_a"
    image_dir.mkdir(parents=True)
    (image_dir / "a.jpg").write_bytes(b"jpg")
    (image_dir / "b.tif").write_bytes(b"tif")
    current_text_dir = tmp_path / "current"
    current_text_dir.mkdir()
    (current_text_dir / "region_a.txt").write_text("text one\ntext two\n", encoding="utf-8")
    historical_text_dir = tmp_path / "historical"
    historical_text_dir.mkdir()
    (historical_text_dir / "region_a.json").write_text(
        json.dumps({"texts": ["historic"]}, ensure_ascii=False),
        encoding="utf-8",
    )
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    (metadata_dir / "region_a.json").write_text(
        json.dumps({"city": "西安"}, ensure_ascii=False),
        encoding="utf-8",
    )

    manifest_path, summary = build_manifest_from_directories(
        output_path=tmp_path / "regions.jsonl",
        image_dir=tmp_path / "images",
        current_text_dir=current_text_dir,
        historical_text_dir=historical_text_dir,
        metadata_dir=metadata_dir,
    )

    content = manifest_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(content) == 1
    record = json.loads(content[0])
    assert record["region_id"] == "region_a"
    assert record["image_paths"] == ["images/region_a/a.jpg", "images/region_a/b.tif"]
    assert record["current_texts"] == ["text one", "text two"]
    assert record["historical_texts"] == ["historic"]
    assert record["metadata"]["city"] == "西安"
    assert summary["region_count"] == 1


def test_summarize_manifest_reports_basic_counts(tmp_path: Path) -> None:
    manifest_path = tmp_path / "regions.jsonl"
    manifest_path.write_text(
        "\n".join(
            [
                (
                    '{"region_id":"region_a","split":"train","image_paths":["a.jpg"],'
                    '"current_texts":["x"],"historical_texts":["h"],"identity_texts":["i"],'
                    '"segmentation_mask_paths":[null],"targets":{"sentiment_label":1}}'
                ),
                (
                    '{"region_id":"region_b","split":"val","image_paths":["b.jpg","c.jpg"],'
                    '"current_texts":["x","y"],"targets":{"iai":0.2}}'
                ),
            ]
        ),
        encoding="utf-8",
    )

    summary = summarize_manifest(manifest_path)

    assert summary["region_count"] == 2
    assert summary["split_counts"]["train"] == 1
    assert summary["images_per_region"]["max"] == 2
    assert summary["target_counts"]["sentiment_label"] == 1
    assert summary["target_counts"]["iai"] == 1
