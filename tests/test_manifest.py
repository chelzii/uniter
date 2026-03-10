from __future__ import annotations

from pathlib import Path

from uniter.data.manifest import load_manifest, validate_manifest


def test_validate_manifest_rejects_common_schema_errors(tmp_path: Path) -> None:
    manifest_path = tmp_path / "broken.jsonl"
    manifest_path.write_text(
        "\n".join(
            [
                '{"region_id":"","split":"dev","image_paths":[],"current_texts":["ok"]}',
                (
                    '{"region_id":"r2","split":"train","image_paths":["a.jpg"],'
                    '"current_texts":["", 1],"metadata":[],"targets":{"ifi":"bad"}}'
                ),
                (
                    '{"region_id":"r3","split":"train","image_paths":["a.jpg"],'
                    '"current_texts":["ok"],"targets":{"sentiment_label":true}}'
                ),
                (
                    '{"region_id":"r4","split":"train","image_paths":["a.jpg"],'
                    '"current_texts":["ok"],"targets":{"historical_sentiment_label":1}}'
                ),
                (
                    '{"region_id":"r4","split":"train","image_paths":["a.jpg"],'
                    '"segmentation_mask_paths":["a.png",null],"current_texts":["ok"],'
                    '"identity_texts":["", 1],"targets":{"iai":"bad"}}'
                ),
            ]
        ),
        encoding="utf-8",
    )

    errors = validate_manifest(manifest_path=manifest_path)

    assert any("'region_id' must be a non-empty string" in error for error in errors)
    assert any("'split' must be one of" in error for error in errors)
    assert any("'image_paths' must not be empty" in error for error in errors)
    assert any("'current_texts' entry must be a non-empty string" in error for error in errors)
    assert any("'metadata' must be a JSON object" in error for error in errors)
    assert any("target 'ifi' must be a number or null" in error for error in errors)
    assert any("target 'sentiment_label' must be an integer or null" in error for error in errors)
    assert any(
        "target 'historical_sentiment_label' requires non-empty 'historical_texts'" in error
        for error in errors
    )
    assert any("duplicate 'region_id' 'r4'" in error for error in errors)
    assert any("'segmentation_mask_paths' must match the length" in error for error in errors)
    assert any("'identity_texts' entry must be a non-empty string" in error for error in errors)
    assert any("target 'iai' must be a number or null" in error for error in errors)


def test_load_manifest_resolves_relative_image_paths(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    image_path = image_dir / "sample.jpg"
    image_path.write_bytes(b"fake-image")
    mask_path = image_dir / "sample.png"
    mask_path.write_bytes(b"fake-mask")

    manifest_path = tmp_path / "regions.jsonl"
    manifest_path.write_text(
        (
            '{"region_id":"region_a","split":"train","image_paths":["images/sample.jpg"],'
            '"segmentation_mask_paths":["images/sample.png"],'
            '"current_texts":["text"],"identity_texts":["ritual axis"]}'
        ),
        encoding="utf-8",
    )

    records = load_manifest(manifest_path, check_files=True)

    assert len(records) == 1
    assert records[0].image_paths == [image_path]
    assert records[0].segmentation_mask_paths == [mask_path]
    assert records[0].identity_texts == ["ritual axis"]
