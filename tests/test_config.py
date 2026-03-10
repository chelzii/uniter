from __future__ import annotations

from pathlib import Path

import pytest

from uniter.config import load_config


def test_load_config_resolves_paths_relative_to_config_file(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (tmp_path / "data").mkdir()

    config_path = config_dir / "experiment.toml"
    config_path.write_text(
        "\n".join(
            [
                "[experiment]",
                'output_dir = "../outputs/run_a"',
                "",
                "[data]",
                'manifest_path = "../data/regions.jsonl"',
                'image_root = "../data/images"',
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.experiment.output_dir == str((tmp_path / "outputs" / "run_a").resolve())
    assert config.data.manifest_path == str((tmp_path / "data" / "regions.jsonl").resolve())
    assert config.data.image_root == str((tmp_path / "data" / "images").resolve())


def test_load_config_rejects_invalid_sentiment_class_names(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.toml"
    config_path.write_text(
        "\n".join(
            [
                "[sentiment]",
                "num_classes = 3",
                'class_names = ["negative", "positive"]',
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="class_names length must match"):
        load_config(config_path)


def test_load_config_rejects_negative_historical_sentiment_weight(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.toml"
    config_path.write_text(
        "\n".join(
            [
                "[losses]",
                "historical_sentiment_weight = -0.1",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="historical_sentiment_weight must be non-negative"):
        load_config(config_path)


def test_load_config_rejects_invalid_pooling_strategy(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.toml"
    config_path.write_text(
        "\n".join(
            [
                "[text_model]",
                'pooling = "max"',
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="text_model.pooling must be either"):
        load_config(config_path)


def test_load_config_rejects_invalid_lost_space_class_count(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.toml"
    config_path.write_text(
        "\n".join(
            [
                "[lost_space]",
                "num_classes = 3",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="lost_space.num_classes must be either 2 or 4"):
        load_config(config_path)


def test_load_config_rejects_invalid_segmentation_label_mapping_key(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.toml"
    config_path.write_text(
        "\n".join(
            [
                "[spatial_supervision]",
                'label_mapping = { road = 0 }',
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="keys must be integer-like strings"):
        load_config(config_path)


def test_load_config_resolves_resume_from_relative_to_config(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    checkpoint_path = checkpoint_dir / "epoch_001.pt"
    checkpoint_path.write_bytes(b"checkpoint")

    config_path = config_dir / "experiment.toml"
    config_path.write_text(
        "\n".join(
            [
                "[training]",
                'resume_from = "../checkpoints/epoch_001.pt"',
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.training.resume_from == str(checkpoint_path.resolve())


def test_load_config_rejects_invalid_judgement_fusion_confidence(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.toml"
    config_path.write_text(
        "\n".join(
            [
                "[judgement_fusion]",
                "minimum_model_confidence = 1.5",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="minimum_model_confidence must be in"):
        load_config(config_path)
