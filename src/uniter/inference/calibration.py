from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from uniter.config import (
    AppConfig,
    CalibrationConfig,
    JudgementConfig,
    SeverityThresholdConfig,
)
from uniter.inference.judgement import judge_region_metrics
from uniter.inference.runtime import InferenceRuntime


@dataclass(frozen=True, slots=True)
class CalibratedThresholds:
    ifi: SeverityThresholdConfig
    mdi: SeverityThresholdConfig
    alignment_gap: SeverityThresholdConfig
    sample_count: int


def _clean_values(values: Iterable[float | None]) -> list[float]:
    cleaned: list[float] = []
    for value in values:
        if value is None:
            continue
        numeric = float(value)
        if math.isnan(numeric):
            continue
        cleaned.append(numeric)
    return cleaned


def _percentile(sorted_values: list[float], quantile: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot compute a percentile from an empty series.")
    if len(sorted_values) == 1:
        return sorted_values[0]

    position = quantile * (len(sorted_values) - 1)
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return sorted_values[lower_index]

    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    weight = position - lower_index
    return lower_value + (upper_value - lower_value) * weight


def fit_thresholds(
    values: Iterable[float | None],
    *,
    calibration: CalibrationConfig,
) -> SeverityThresholdConfig:
    cleaned = sorted(_clean_values(values))
    if len(cleaned) < calibration.min_samples:
        raise ValueError(
            "Not enough samples to calibrate thresholds: "
            f"expected at least {calibration.min_samples}, got {len(cleaned)}."
        )

    return SeverityThresholdConfig(
        light=_percentile(cleaned, calibration.light_quantile),
        moderate=_percentile(cleaned, calibration.moderate_quantile),
        severe=_percentile(cleaned, calibration.severe_quantile),
    )


class ThresholdCalibrator:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.runtime = InferenceRuntime(config)

    def calibrate(
        self,
        *,
        checkpoint_path: str | Path | None,
        split: str,
        output_path: str | Path | None,
        allow_random_init: bool = False,
    ) -> Path:
        ifi_values: list[float | None] = []
        mdi_values: list[float | None] = []
        alignment_values: list[float | None] = []
        for batch, outputs in self.runtime.iterate_region_batches(
            split=split,
            checkpoint_path=checkpoint_path,
            allow_random_init=allow_random_init,
        ):
            del batch
            ifi_values.extend(float(value) for value in outputs.ifi.detach().cpu().tolist())
            mdi_values.extend(float(value) for value in outputs.mdi.detach().cpu().tolist())
            alignment_values.extend(
                float(value) for value in outputs.alignment_gap.detach().cpu().tolist()
            )

        calibrated = CalibratedThresholds(
            ifi=fit_thresholds(ifi_values, calibration=self.config.calibration),
            mdi=fit_thresholds(mdi_values, calibration=self.config.calibration),
            alignment_gap=fit_thresholds(
                alignment_values,
                calibration=self.config.calibration,
            ),
            sample_count=len(self.runtime.resolve_records(split)),
        )
        destination = self._resolve_output_path(output_path, split=split)
        preview = self._build_preview(calibrated, ifi_values, mdi_values, alignment_values)
        payload = {
            "split": split,
            "sample_count": calibrated.sample_count,
            "thresholds": asdict(calibrated),
            "judgement_preview": preview,
        }
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return destination

    def _resolve_output_path(self, output_path: str | Path | None, *, split: str) -> Path:
        if output_path is not None:
            return Path(output_path).resolve()
        return (self.config.output_dir / "calibration" / f"thresholds_{split}.json").resolve()

    def _build_preview(
        self,
        calibrated: CalibratedThresholds,
        ifi_values: list[float | None],
        mdi_values: list[float | None],
        alignment_values: list[float | None],
    ) -> dict[str, int]:
        preview_config = AppConfig(
            experiment=self.config.experiment,
            data=self.config.data,
            spatial_model=self.config.spatial_model,
            spatial_supervision=self.config.spatial_supervision,
            text_model=self.config.text_model,
            sentiment=self.config.sentiment,
            lost_space=self.config.lost_space,
            alignment=self.config.alignment,
            losses=self.config.losses,
            training=self.config.training,
            metrics=self.config.metrics,
            judgement=JudgementConfig(
                use_alignment_gap=self.config.judgement.use_alignment_gap,
                ifi=calibrated.ifi,
                mdi=calibrated.mdi,
                iai=self.config.judgement.iai,
                alignment_gap=calibrated.alignment_gap,
            ),
            judgement_fusion=self.config.judgement_fusion,
            calibration=self.config.calibration,
            identity=self.config.identity,
            inference=self.config.inference,
        )
        counts = {"none": 0, "light": 0, "moderate": 0, "severe": 0}
        for ifi, mdi, alignment in zip(ifi_values, mdi_values, alignment_values, strict=True):
            result = judge_region_metrics(
                ifi=ifi,
                mdi=mdi,
                alignment_gap=alignment,
                config=preview_config,
            )
            counts[result.lost_space_level] += 1
        return counts
