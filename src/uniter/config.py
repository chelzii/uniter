from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _merge_dict(defaults: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = defaults.copy()
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


@dataclass(slots=True)
class ExperimentConfig:
    name: str = "lost-space-base"
    seed: int = 42
    device: str = "auto"
    output_dir: str = "outputs/default"


@dataclass(slots=True)
class DataConfig:
    manifest_path: str = "data/regions.jsonl"
    image_root: str | None = None
    batch_size: int = 2
    num_workers: int = 0
    image_size: int = 512
    max_images_per_region: int = 4
    max_current_texts_per_region: int = 8
    max_historical_texts_per_region: int = 4
    max_identity_texts_per_region: int = 6
    check_files_on_load: bool = True


@dataclass(slots=True)
class SpatialModelConfig:
    model_name: str = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
    freeze_encoder: bool = True
    label_groups: dict[str, list[str]] = field(
        default_factory=lambda: {
            "road_surface": ["road", "sidewalk"],
            "enclosure": ["building", "wall", "fence"],
            "vegetation": ["vegetation", "terrain"],
            "sky": ["sky"],
            "mobility": ["car", "truck", "bus", "train", "motorcycle", "bicycle"],
            "pedestrian": ["person", "rider"],
        }
    )


@dataclass(slots=True)
class SpatialSupervisionConfig:
    enabled: bool = False
    ignore_index: int = 255
    label_mapping: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class TextModelConfig:
    model_name: str = "hfl/chinese-macbert-base"
    freeze_encoder: bool = True
    max_length: int = 256
    pooling: str = "mean"


@dataclass(slots=True)
class SentimentConfig:
    enabled: bool = True
    num_classes: int = 3
    class_names: list[str] = field(
        default_factory=lambda: ["negative", "neutral", "positive"]
    )
    dropout: float = 0.1
    ignore_index: int = -100


@dataclass(slots=True)
class LostSpaceConfig:
    enabled: bool = False
    num_classes: int = 4
    class_names: list[str] = field(
        default_factory=lambda: ["none", "light", "moderate", "severe"]
    )
    dropout: float = 0.1
    ignore_index: int = -100


@dataclass(slots=True)
class AlignmentConfig:
    embed_dim: int = 256
    projection_dropout: float = 0.1
    temperature: float = 0.07


@dataclass(slots=True)
class LossConfig:
    alignment_weight: float = 1.0
    sentiment_weight: float = 1.0
    historical_sentiment_weight: float = 1.0
    lost_space_weight: float = 1.0
    segmentation_weight: float = 1.0
    sentiment_label_smoothing: float = 0.0


@dataclass(slots=True)
class TrainingConfig:
    epochs: int = 10
    learning_rate: float = 3e-5
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    log_every: int = 10
    amp: bool = True
    save_every: int = 1
    save_best: bool = True
    monitor_metric: str = "val.loss"
    maximize_monitor_metric: bool = False
    early_stopping_patience: int | None = None
    resume_from: str | None = None


@dataclass(slots=True)
class MetricConfig:
    ifi_target_profile: dict[str, float] = field(
        default_factory=lambda: {
            "road_surface": 0.22,
            "enclosure": 0.28,
            "vegetation": 0.18,
            "sky": 0.20,
            "mobility": 0.08,
            "pedestrian": 0.04,
        }
    )
    ifi_weights: dict[str, float] = field(
        default_factory=lambda: {
            "road_surface": 1.0,
            "enclosure": 1.2,
            "vegetation": 1.1,
            "sky": 0.8,
            "mobility": 1.1,
            "pedestrian": 0.8,
        }
    )
    mdi_normalizer: float = 1.0
    iai_normalizer: float = 1.0


@dataclass(slots=True)
class SeverityThresholdConfig:
    light: float = 0.20
    moderate: float = 0.35
    severe: float = 0.50


@dataclass(slots=True)
class JudgementConfig:
    use_alignment_gap: bool = True
    ifi: SeverityThresholdConfig = field(default_factory=SeverityThresholdConfig)
    mdi: SeverityThresholdConfig = field(default_factory=SeverityThresholdConfig)
    iai: SeverityThresholdConfig = field(default_factory=SeverityThresholdConfig)
    alignment_gap: SeverityThresholdConfig = field(
        default_factory=lambda: SeverityThresholdConfig(
            light=0.25,
            moderate=0.40,
            severe=0.55,
        )
    )


@dataclass(slots=True)
class JudgementFusionConfig:
    enabled: bool = True
    rule_weight: float = 0.65
    model_weight: float = 0.35
    minimum_model_confidence: float = 0.55
    use_iai: bool = False
    iai_weight: float = 0.15


@dataclass(slots=True)
class CalibrationConfig:
    light_quantile: float = 0.50
    moderate_quantile: float = 0.70
    severe_quantile: float = 0.85
    min_samples: int = 10


@dataclass(slots=True)
class IdentityConfig:
    enabled: bool = False
    projection_dropout: float = 0.1


@dataclass(slots=True)
class InferenceConfig:
    require_checkpoint: bool = True


@dataclass(slots=True)
class AppConfig:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    spatial_model: SpatialModelConfig = field(default_factory=SpatialModelConfig)
    spatial_supervision: SpatialSupervisionConfig = field(default_factory=SpatialSupervisionConfig)
    text_model: TextModelConfig = field(default_factory=TextModelConfig)
    sentiment: SentimentConfig = field(default_factory=SentimentConfig)
    lost_space: LostSpaceConfig = field(default_factory=LostSpaceConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    losses: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    metrics: MetricConfig = field(default_factory=MetricConfig)
    judgement: JudgementConfig = field(default_factory=JudgementConfig)
    judgement_fusion: JudgementFusionConfig = field(default_factory=JudgementFusionConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    identity: IdentityConfig = field(default_factory=IdentityConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    @property
    def output_dir(self) -> Path:
        return Path(self.experiment.output_dir)


DEFAULT_CONFIG = AppConfig()


def _resolve_relative_path(
    raw_value: str | None,
    *,
    config_dir: Path,
) -> str | None:
    if raw_value in (None, ""):
        return None

    candidate = Path(raw_value)
    if candidate.is_absolute():
        return str(candidate)
    return str((config_dir / candidate).resolve())


def _validate_config(config: AppConfig) -> AppConfig:
    if config.data.batch_size < 1:
        raise ValueError("data.batch_size must be at least 1.")
    if config.data.num_workers < 0:
        raise ValueError("data.num_workers must be non-negative.")
    if config.data.image_size < 1:
        raise ValueError("data.image_size must be at least 1.")
    for field_name, value in (
        ("data.max_images_per_region", config.data.max_images_per_region),
        ("data.max_current_texts_per_region", config.data.max_current_texts_per_region),
        ("data.max_historical_texts_per_region", config.data.max_historical_texts_per_region),
        ("data.max_identity_texts_per_region", config.data.max_identity_texts_per_region),
    ):
        if value < 1:
            raise ValueError(f"{field_name} must be at least 1.")

    if config.sentiment.num_classes < 2:
        raise ValueError("sentiment.num_classes must be at least 2.")
    if len(config.sentiment.class_names) != config.sentiment.num_classes:
        raise ValueError(
            "sentiment.class_names length must match sentiment.num_classes."
        )
    if config.sentiment.ignore_index >= 0:
        raise ValueError("sentiment.ignore_index must be a negative integer.")
    if not 0.0 <= config.sentiment.dropout < 1.0:
        raise ValueError("sentiment.dropout must be in [0.0, 1.0).")

    if config.lost_space.num_classes not in {2, 4}:
        raise ValueError("lost_space.num_classes must be either 2 or 4.")
    if len(config.lost_space.class_names) != config.lost_space.num_classes:
        raise ValueError(
            "lost_space.class_names length must match lost_space.num_classes."
        )
    if config.lost_space.ignore_index >= 0:
        raise ValueError("lost_space.ignore_index must be a negative integer.")
    if not 0.0 <= config.lost_space.dropout < 1.0:
        raise ValueError("lost_space.dropout must be in [0.0, 1.0).")

    smoothing = config.losses.sentiment_label_smoothing
    if not 0.0 <= smoothing < 1.0:
        raise ValueError("losses.sentiment_label_smoothing must be in [0.0, 1.0).")
    if config.losses.alignment_weight < 0.0:
        raise ValueError("losses.alignment_weight must be non-negative.")
    if config.losses.sentiment_weight < 0.0:
        raise ValueError("losses.sentiment_weight must be non-negative.")
    if config.losses.historical_sentiment_weight < 0.0:
        raise ValueError("losses.historical_sentiment_weight must be non-negative.")
    if config.losses.lost_space_weight < 0.0:
        raise ValueError("losses.lost_space_weight must be non-negative.")
    if config.losses.segmentation_weight < 0.0:
        raise ValueError("losses.segmentation_weight must be non-negative.")
    if config.text_model.pooling not in {"mean", "cls"}:
        raise ValueError("text_model.pooling must be either 'mean' or 'cls'.")
    if config.text_model.max_length < 1:
        raise ValueError("text_model.max_length must be at least 1.")
    if config.training.epochs < 1:
        raise ValueError("training.epochs must be at least 1.")
    if config.training.learning_rate <= 0.0:
        raise ValueError("training.learning_rate must be positive.")
    if config.training.weight_decay < 0.0:
        raise ValueError("training.weight_decay must be non-negative.")
    if config.training.grad_clip_norm <= 0.0:
        raise ValueError("training.grad_clip_norm must be positive.")
    if config.training.log_every < 1:
        raise ValueError("training.log_every must be at least 1.")
    if config.training.save_every < 1:
        raise ValueError("training.save_every must be at least 1.")
    if config.training.monitor_metric not in {"train.loss", "val.loss"}:
        raise ValueError("training.monitor_metric must be either 'train.loss' or 'val.loss'.")
    if (
        config.training.early_stopping_patience is not None
        and config.training.early_stopping_patience < 1
    ):
        raise ValueError("training.early_stopping_patience must be at least 1 when set.")
    if config.calibration.min_samples < 1:
        raise ValueError("calibration.min_samples must be at least 1.")
    if config.metrics.mdi_normalizer <= 0.0:
        raise ValueError("metrics.mdi_normalizer must be positive.")
    if config.metrics.iai_normalizer <= 0.0:
        raise ValueError("metrics.iai_normalizer must be positive.")
    if config.spatial_supervision.ignore_index < 0:
        raise ValueError("spatial_supervision.ignore_index must be non-negative.")
    for raw_label, mapped_label in config.spatial_supervision.label_mapping.items():
        if not raw_label:
            raise ValueError("spatial_supervision.label_mapping keys must be non-empty strings.")
        try:
            int(raw_label)
        except ValueError as exc:
            raise ValueError(
                "spatial_supervision.label_mapping keys must be integer-like strings."
            ) from exc
        if not isinstance(mapped_label, int) or isinstance(mapped_label, bool) or mapped_label < 0:
            raise ValueError(
                "spatial_supervision.label_mapping values must be non-negative integers."
            )
    if not 0.0 <= config.identity.projection_dropout < 1.0:
        raise ValueError("identity.projection_dropout must be in [0.0, 1.0).")
    for section_name, thresholds in (
        ("judgement.ifi", config.judgement.ifi),
        ("judgement.mdi", config.judgement.mdi),
        ("judgement.iai", config.judgement.iai),
        ("judgement.alignment_gap", config.judgement.alignment_gap),
    ):
        if not 0.0 <= thresholds.light <= thresholds.moderate <= thresholds.severe:
            raise ValueError(
                f"{section_name} thresholds must satisfy light <= moderate <= severe."
            )
    if config.judgement_fusion.rule_weight < 0.0:
        raise ValueError("judgement_fusion.rule_weight must be non-negative.")
    if config.judgement_fusion.model_weight < 0.0:
        raise ValueError("judgement_fusion.model_weight must be non-negative.")
    if config.judgement_fusion.iai_weight < 0.0:
        raise ValueError("judgement_fusion.iai_weight must be non-negative.")
    if (
        config.judgement_fusion.rule_weight
        + config.judgement_fusion.model_weight
        + config.judgement_fusion.iai_weight
        <= 0.0
    ):
        raise ValueError("judgement_fusion weights must sum to a positive value.")
    if not 0.0 <= config.judgement_fusion.minimum_model_confidence <= 1.0:
        raise ValueError(
            "judgement_fusion.minimum_model_confidence must be in [0.0, 1.0]."
        )
    if not (
        0.0 < config.calibration.light_quantile
        <= config.calibration.moderate_quantile
        <= config.calibration.severe_quantile
        < 1.0
    ):
        raise ValueError(
            "calibration quantiles must satisfy 0 < light <= moderate <= severe < 1."
        )
    return config


def load_config(config_path: str | Path) -> AppConfig:
    path = Path(config_path)
    with path.open("rb") as handle:
        payload = tomllib.load(handle)

    defaults = asdict(DEFAULT_CONFIG)
    merged = _merge_dict(defaults, payload)
    config_dir = path.resolve().parent
    merged["data"]["manifest_path"] = _resolve_relative_path(
        merged["data"]["manifest_path"],
        config_dir=config_dir,
    )
    merged["data"]["image_root"] = _resolve_relative_path(
        merged["data"].get("image_root"),
        config_dir=config_dir,
    )
    merged["experiment"]["output_dir"] = _resolve_relative_path(
        merged["experiment"]["output_dir"],
        config_dir=config_dir,
    ) or str((config_dir / "outputs/default").resolve())
    merged["training"]["resume_from"] = _resolve_relative_path(
        merged["training"].get("resume_from"),
        config_dir=config_dir,
    )

    config = AppConfig(
        experiment=ExperimentConfig(**merged["experiment"]),
        data=DataConfig(**merged["data"]),
        spatial_model=SpatialModelConfig(**merged["spatial_model"]),
        spatial_supervision=SpatialSupervisionConfig(**merged["spatial_supervision"]),
        text_model=TextModelConfig(**merged["text_model"]),
        sentiment=SentimentConfig(**merged["sentiment"]),
        lost_space=LostSpaceConfig(**merged["lost_space"]),
        alignment=AlignmentConfig(**merged["alignment"]),
        losses=LossConfig(**merged["losses"]),
        training=TrainingConfig(**merged["training"]),
        metrics=MetricConfig(**merged["metrics"]),
        judgement=JudgementConfig(
            use_alignment_gap=merged["judgement"]["use_alignment_gap"],
            ifi=SeverityThresholdConfig(**merged["judgement"]["ifi"]),
            mdi=SeverityThresholdConfig(**merged["judgement"]["mdi"]),
            iai=SeverityThresholdConfig(**merged["judgement"]["iai"]),
            alignment_gap=SeverityThresholdConfig(
                **merged["judgement"]["alignment_gap"]
            ),
        ),
        judgement_fusion=JudgementFusionConfig(**merged["judgement_fusion"]),
        calibration=CalibrationConfig(**merged["calibration"]),
        identity=IdentityConfig(**merged["identity"]),
        inference=InferenceConfig(**merged["inference"]),
    )
    return _validate_config(config)
