from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from uniter.config import AppConfig
from uniter.data.dataset import RegionBatch
from uniter.models.multimodal import ModelOutputs
from uniter.training.trainer import Trainer


class _DummyCollator:
    def __init__(self, **_: object) -> None:
        pass

    def __call__(self, samples) -> RegionBatch:
        region_ids = [sample.region_id for sample in samples]
        metadata = []
        for sample in samples:
            sample_metadata = dict(sample.metadata)
            sample_metadata["selected_image_filenames"] = [path.name for path in sample.image_paths]
            sample_metadata["image_count"] = len(sample.image_paths)
            sample_metadata["current_text_count"] = len(sample.current_texts)
            sample_metadata["historical_text_count"] = len(sample.historical_texts)
            sample_metadata["identity_text_count"] = len(sample.identity_texts)
            metadata.append(sample_metadata)

        region_count = len(samples)
        return RegionBatch(
            region_ids=region_ids,
            pixel_values=torch.ones(region_count, 3, 2, 2),
            segmentation_labels=None,
            image_region_index=torch.arange(region_count, dtype=torch.long),
            image_is_satellite=torch.zeros(region_count, dtype=torch.bool),
            image_point_ids=[f"p{index:03d}" for index in range(region_count)],
            image_view_directions=["north"] * region_count,
            image_longitudes=[108.95 + index * 0.001 for index in range(region_count)],
            image_latitudes=[34.25 for _ in range(region_count)],
            current_input_ids=torch.ones(region_count, 3, dtype=torch.long),
            current_attention_mask=torch.ones(region_count, 3, dtype=torch.long),
            current_region_index=torch.arange(region_count, dtype=torch.long),
            current_sentiment_labels=torch.tensor(
                [sample.targets.sentiment_label or 1 for sample in samples],
                dtype=torch.long,
            ),
            historical_input_ids=torch.ones(region_count, 3, dtype=torch.long),
            historical_attention_mask=torch.ones(region_count, 3, dtype=torch.long),
            historical_region_index=torch.arange(region_count, dtype=torch.long),
            historical_sentiment_labels=torch.tensor(
                [sample.targets.historical_sentiment_label or 2 for sample in samples],
                dtype=torch.long,
            ),
            identity_input_ids=torch.ones(region_count, 3, dtype=torch.long),
            identity_attention_mask=torch.ones(region_count, 3, dtype=torch.long),
            identity_region_index=torch.arange(region_count, dtype=torch.long),
            metadata=metadata,
            targets={
                "lost_space_label": torch.tensor(
                    [sample.targets.lost_space_label or 2 for sample in samples],
                    dtype=torch.long,
                ),
                "sentiment_label": torch.tensor(
                    [sample.targets.sentiment_label or 1 for sample in samples],
                    dtype=torch.long,
                ),
                "historical_sentiment_label": torch.tensor(
                    [sample.targets.historical_sentiment_label or 2 for sample in samples],
                    dtype=torch.long,
                ),
                "ifi": torch.tensor([float("nan")] * region_count),
                "mdi": torch.tensor([float("nan")] * region_count),
                "iai": torch.tensor([float("nan")] * region_count),
            },
        )


class _DummyModel(nn.Module):
    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.current_sentiment_head = nn.Module()
        self.current_sentiment_head.classifier = nn.Sequential(
            nn.Dropout(0.0),
            nn.Linear(4, config.sentiment.num_classes),
        )
        self.historical_sentiment_head = nn.Module()
        self.historical_sentiment_head.classifier = nn.Sequential(
            nn.Dropout(0.0),
            nn.Linear(4, config.sentiment.num_classes),
        )
        self.lost_space_head = nn.Module()
        self.lost_space_head.classifier = nn.Sequential(
            nn.Dropout(0.0),
            nn.Linear(4, config.lost_space.num_classes),
        )

    def forward(self, batch: RegionBatch) -> ModelOutputs:
        region_count = len(batch.region_ids)
        image_count = batch.pixel_values.shape[0]
        base_embedding = torch.stack(
            [
                torch.ones(region_count) * self.weight,
                torch.ones(region_count) * (self.weight + 1),
                torch.ones(region_count) * (self.weight + 2),
                torch.ones(region_count) * (self.weight + 3),
            ],
            dim=-1,
        )
        sentiment_logits = torch.stack(
            [
                torch.ones(region_count) * self.weight,
                torch.ones(region_count) * (self.weight + 0.5),
                torch.ones(region_count) * (self.weight + 1.0),
            ],
            dim=-1,
        )
        historical_sentiment_logits = torch.stack(
            [
                torch.ones(region_count) * self.weight,
                torch.ones(region_count) * (self.weight + 0.2),
                torch.ones(region_count) * (self.weight + 1.2),
            ],
            dim=-1,
        )
        lost_space_logits = torch.stack(
            [
                torch.ones(region_count) * self.weight,
                torch.ones(region_count) * (self.weight + 0.2),
                torch.ones(region_count) * (self.weight + 0.4),
                torch.ones(region_count) * (self.weight + 0.6),
            ],
            dim=-1,
        )
        return ModelOutputs(
            image_embeddings=base_embedding,
            text_embeddings=base_embedding + 0.1,
            image_logits=torch.ones(region_count, 2) * self.weight,
            segmentation_logits=torch.ones(image_count, 2, 2, 2) * self.weight,
            satellite_region_mask=torch.zeros(region_count, dtype=torch.bool),
            current_text_features=base_embedding,
            historical_text_features=base_embedding + 0.2,
            identity_text_features=base_embedding + 0.3,
            current_sentiment_record_logits=sentiment_logits,
            historical_sentiment_record_logits=historical_sentiment_logits,
            sentiment_logits=sentiment_logits,
            historical_sentiment_logits=historical_sentiment_logits,
            historical_region_mask=torch.ones(region_count, dtype=torch.bool),
            mdi_sentiment_mask=torch.ones(region_count, dtype=torch.bool),
            lost_space_logits=lost_space_logits,
            identity_embeddings=base_embedding + 0.4,
            identity_region_mask=torch.ones(region_count, dtype=torch.bool),
            alignment_gap=torch.ones(region_count) * 0.4,
            ifi=torch.ones(region_count) * 0.3,
            mdi=torch.ones(region_count) * 0.2,
            iai=torch.ones(region_count) * 0.5,
        )


def _write_manifest(path: Path, *, extra_train_record: bool = False) -> None:
    rows = [
        {
            "region_id": "region_train_a",
            "split": "train",
            "image_paths": ["image_a.png"],
            "current_texts": ["current a"],
            "historical_texts": ["historical a"],
            "identity_texts": ["identity a"],
            "metadata": {"parent_region_id": "region_base", "bootstrap_view": "train_a"},
            "targets": {
                "lost_space_label": 2,
                "sentiment_label": 1,
                "historical_sentiment_label": 2,
            },
        },
        {
            "region_id": "region_train_b",
            "split": "train",
            "image_paths": ["image_b.png"],
            "current_texts": ["current b"],
            "historical_texts": ["historical b"],
            "identity_texts": ["identity b"],
            "metadata": {"parent_region_id": "region_base", "bootstrap_view": "train_b"},
            "targets": {
                "lost_space_label": 2,
                "sentiment_label": 1,
                "historical_sentiment_label": 2,
            },
        },
        *(
            [
                {
                    "region_id": "region_train_c",
                    "split": "train",
                    "image_paths": ["image_c.png"],
                    "current_texts": ["current c"],
                    "historical_texts": ["historical c"],
                    "identity_texts": ["identity c"],
                    "metadata": {
                        "parent_region_id": "region_base",
                        "bootstrap_view": "train_c",
                    },
                    "targets": {
                        "lost_space_label": 2,
                        "sentiment_label": 1,
                        "historical_sentiment_label": 2,
                    },
                }
            ]
            if extra_train_record
            else []
        ),
        {
            "region_id": "region_val",
            "split": "val",
            "image_paths": ["image_val.png"],
            "current_texts": ["current val"],
            "historical_texts": ["historical val"],
            "identity_texts": ["identity val"],
            "metadata": {"parent_region_id": "region_base", "bootstrap_view": "val"},
            "targets": {
                "lost_space_label": 2,
                "sentiment_label": 1,
                "historical_sentiment_label": 2,
            },
        },
    ]
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_trainer_smoke_creates_output_dirs_and_summaries(monkeypatch, tmp_path: Path) -> None:
    manifest_path = tmp_path / "regions.jsonl"
    _write_manifest(manifest_path)
    for image_name in ("image_a.png", "image_b.png", "image_val.png"):
        (tmp_path / image_name).write_bytes(b"png")

    import uniter.training.trainer as trainer_module

    monkeypatch.setattr(trainer_module, "RegionBatchCollator", _DummyCollator)
    monkeypatch.setattr(trainer_module, "MultimodalRegionModel", _DummyModel)

    config = AppConfig()
    config.experiment.device = "cpu"
    config.experiment.output_dir = str(tmp_path / "run")
    config.data.manifest_path = str(manifest_path)
    config.data.batch_size = 2
    config.data.check_files_on_load = True
    config.training.epochs = 1
    config.training.save_every = 1
    config.training.save_best = True
    config.training.amp = False
    config.training.early_stopping_patience = None
    config.lost_space.enabled = True
    config.identity.enabled = True

    trainer = Trainer(config)
    trainer.train()

    assert (Path(config.experiment.output_dir) / "checkpoints" / "best.pt").exists()
    assert (Path(config.experiment.output_dir) / "summaries" / "epoch_001.json").exists()
    assert (Path(config.experiment.output_dir) / "summaries" / "data_summary.json").exists()
    assert (Path(config.experiment.output_dir) / "training_state.json").exists()


def test_trainer_resets_stale_training_artifacts_and_keeps_last_batch(monkeypatch, tmp_path: Path) -> None:
    manifest_path = tmp_path / "regions.jsonl"
    _write_manifest(manifest_path, extra_train_record=True)
    for image_name in ("image_a.png", "image_b.png", "image_c.png", "image_val.png"):
        (tmp_path / image_name).write_bytes(b"png")

    import uniter.training.trainer as trainer_module

    monkeypatch.setattr(trainer_module, "RegionBatchCollator", _DummyCollator)
    monkeypatch.setattr(trainer_module, "MultimodalRegionModel", _DummyModel)

    output_dir = tmp_path / "run"
    (output_dir / "summaries").mkdir(parents=True)
    (output_dir / "checkpoints").mkdir(parents=True)
    (output_dir / "summaries" / "history.json").write_text(
        json.dumps([{"epoch": 99}], ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "summaries" / "epoch_099.json").write_text("{}", encoding="utf-8")
    (output_dir / "checkpoints" / "epoch_099.pt").write_text("stale", encoding="utf-8")

    config = AppConfig()
    config.experiment.device = "cpu"
    config.experiment.output_dir = str(output_dir)
    config.data.manifest_path = str(manifest_path)
    config.data.batch_size = 2
    config.data.check_files_on_load = True
    config.training.epochs = 1
    config.training.save_every = 1
    config.training.save_best = True
    config.training.amp = False
    config.training.early_stopping_patience = None
    config.lost_space.enabled = True
    config.identity.enabled = True

    trainer = Trainer(config)
    assert trainer.train_loader is not None
    assert trainer.train_loader.drop_last is False
    trainer.train()

    history = json.loads((output_dir / "summaries" / "history.json").read_text(encoding="utf-8"))
    assert len(history) == 1
    assert history[0]["epoch"] == 1
    assert not (output_dir / "summaries" / "epoch_099.json").exists()
    assert not (output_dir / "checkpoints" / "epoch_099.pt").exists()


def test_trainer_initializes_classifier_head_biases_from_train_priors(
    monkeypatch,
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "regions.jsonl"
    _write_manifest(manifest_path)
    for image_name in ("image_a.png", "image_b.png", "image_val.png"):
        (tmp_path / image_name).write_bytes(b"png")

    import uniter.training.trainer as trainer_module

    monkeypatch.setattr(trainer_module, "RegionBatchCollator", _DummyCollator)
    monkeypatch.setattr(trainer_module, "MultimodalRegionModel", _DummyModel)

    config = AppConfig()
    config.experiment.device = "cpu"
    config.experiment.output_dir = str(tmp_path / "run")
    config.data.manifest_path = str(manifest_path)
    config.data.batch_size = 2
    config.data.check_files_on_load = True
    config.training.epochs = 1
    config.training.amp = False
    config.lost_space.enabled = True
    config.identity.enabled = True

    trainer = Trainer(config)

    current_bias = trainer.model.current_sentiment_head.classifier[1].bias.detach().cpu()
    historical_bias = trainer.model.historical_sentiment_head.classifier[1].bias.detach().cpu()
    bias = trainer.model.lost_space_head.classifier[1].bias.detach().cpu()
    assert int(current_bias.argmax().item()) == 1
    assert int(historical_bias.argmax().item()) == 2
    assert int(bias.argmax().item()) == 2


def test_selection_score_uses_loss_as_tie_breaker() -> None:
    trainer = Trainer.__new__(Trainer)

    lower_loss_score = Trainer._compute_selection_score(
        trainer,
        loss=0.5,
        model_lost_space_accuracy=1.0,
        rule_lost_space_accuracy=1.0,
        final_lost_space_accuracy=1.0,
        rule_model_agreement=1.0,
        sentiment_accuracy=1.0,
        historical_sentiment_accuracy=1.0,
        labeled_count=4,
        compared_count=4,
        sentiment_labeled_count=4,
        historical_sentiment_labeled_count=4,
    )
    higher_loss_score = Trainer._compute_selection_score(
        trainer,
        loss=2.0,
        model_lost_space_accuracy=1.0,
        rule_lost_space_accuracy=1.0,
        final_lost_space_accuracy=1.0,
        rule_model_agreement=1.0,
        sentiment_accuracy=1.0,
        historical_sentiment_accuracy=1.0,
        labeled_count=4,
        compared_count=4,
        sentiment_labeled_count=4,
        historical_sentiment_labeled_count=4,
    )

    assert lower_loss_score > higher_loss_score


def test_trainer_marks_training_state_failed_when_epoch_raises(
    monkeypatch,
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "regions.jsonl"
    _write_manifest(manifest_path)
    for image_name in ("image_a.png", "image_b.png", "image_val.png"):
        (tmp_path / image_name).write_bytes(b"png")

    import uniter.training.trainer as trainer_module

    monkeypatch.setattr(trainer_module, "RegionBatchCollator", _DummyCollator)
    monkeypatch.setattr(trainer_module, "MultimodalRegionModel", _DummyModel)

    config = AppConfig()
    config.experiment.device = "cpu"
    config.experiment.output_dir = str(tmp_path / "run")
    config.data.manifest_path = str(manifest_path)
    config.data.batch_size = 2
    config.data.check_files_on_load = True
    config.training.epochs = 2
    config.training.amp = False
    config.lost_space.enabled = True
    config.identity.enabled = True

    trainer = Trainer(config)

    original_run_epoch = trainer._run_epoch
    call_count = {"count": 0}

    def _failing_run_epoch(epoch: int, *, training: bool) -> dict[str, float]:
        call_count["count"] += 1
        if call_count["count"] == 2:
            raise RuntimeError("boom")
        return original_run_epoch(epoch, training=training)

    monkeypatch.setattr(trainer, "_run_epoch", _failing_run_epoch)

    with pytest.raises(RuntimeError, match="boom"):
        trainer.train()

    state = json.loads((Path(config.experiment.output_dir) / "training_state.json").read_text())
    assert state["status"] == "failed"
    assert state["stop_reason"].startswith("failed:RuntimeError:boom")
