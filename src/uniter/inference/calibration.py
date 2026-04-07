from __future__ import annotations

import json
import math
import random
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from uniter.config import (
    AppConfig,
    CalibrationConfig,
    JudgementConfig,
    SeverityThresholdConfig,
)
from uniter.inference.judgement import judge_region_metrics, map_rule_level_to_class_index
from uniter.inference.runtime import InferenceRuntime


@dataclass(frozen=True, slots=True)
class CalibratedThresholds:
    ifi: SeverityThresholdConfig
    mdi: SeverityThresholdConfig
    iai: SeverityThresholdConfig
    alignment_gap: SeverityThresholdConfig
    sample_count: int


DEFAULT_THRESHOLD_SPLIT = "train"


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


def _blend_thresholds(
    estimated: SeverityThresholdConfig,
    fallback: SeverityThresholdConfig,
    *,
    weight: float,
) -> SeverityThresholdConfig:
    clamped_weight = min(max(float(weight), 0.0), 1.0)
    return SeverityThresholdConfig(
        light=fallback.light * (1.0 - clamped_weight) + estimated.light * clamped_weight,
        moderate=fallback.moderate * (1.0 - clamped_weight)
        + estimated.moderate * clamped_weight,
        severe=fallback.severe * (1.0 - clamped_weight) + estimated.severe * clamped_weight,
    )


def _bootstrap_threshold_estimate(
    values: list[float],
    *,
    calibration: CalibrationConfig,
) -> tuple[SeverityThresholdConfig, dict[str, float]]:
    if not values:
        raise ValueError("Cannot bootstrap thresholds from an empty series.")
    if len(values) == 1:
        single = SeverityThresholdConfig(light=values[0], moderate=values[0], severe=values[0])
        return single, {"light": 0.0, "moderate": 0.0, "severe": 0.0}

    rng = random.Random(0)
    sampled_thresholds: list[SeverityThresholdConfig] = []
    relaxed_calibration = CalibrationConfig(
        light_quantile=calibration.light_quantile,
        moderate_quantile=calibration.moderate_quantile,
        severe_quantile=calibration.severe_quantile,
        min_samples=1,
        bootstrap_resamples=calibration.bootstrap_resamples,
        partial_min_weight=calibration.partial_min_weight,
    )
    for _ in range(calibration.bootstrap_resamples):
        sample = sorted(rng.choice(values) for _ in range(len(values)))
        sampled_thresholds.append(fit_thresholds(sample, calibration=relaxed_calibration))

    def _mean_and_std(name: str) -> tuple[float, float]:
        field_values = [float(getattr(item, name)) for item in sampled_thresholds]
        mean_value = sum(field_values) / max(len(field_values), 1)
        variance = sum((value - mean_value) ** 2 for value in field_values) / max(
            len(field_values),
            1,
        )
        return mean_value, math.sqrt(variance)

    light_mean, light_std = _mean_and_std("light")
    moderate_mean, moderate_std = _mean_and_std("moderate")
    severe_mean, severe_std = _mean_and_std("severe")
    return (
        SeverityThresholdConfig(
            light=light_mean,
            moderate=moderate_mean,
            severe=severe_mean,
        ),
        {
            "light": light_std,
            "moderate": moderate_std,
            "severe": severe_std,
        },
    )


def fit_thresholds_or_default(
    values: Iterable[float | None],
    *,
    calibration: CalibrationConfig,
    fallback: SeverityThresholdConfig,
) -> tuple[SeverityThresholdConfig, int, bool, str, float, dict[str, float]]:
    cleaned = _clean_values(values)
    if not cleaned:
        return fallback, 0, False, "fallback", 0.0, {
            "light": 0.0,
            "moderate": 0.0,
            "severe": 0.0,
        }
    if len(cleaned) < calibration.min_samples:
        estimated, uncertainty = _bootstrap_threshold_estimate(
            cleaned,
            calibration=calibration,
        )
        support = len(cleaned) / max(calibration.min_samples, 1)
        effective_weight = max(calibration.partial_min_weight, min(support, 1.0))
        return (
            _blend_thresholds(estimated, fallback, weight=effective_weight),
            len(cleaned),
            True,
            "bootstrap_shrinkage",
            effective_weight,
            uncertainty,
        )
    return (
        fit_thresholds(cleaned, calibration=calibration),
        len(cleaned),
        True,
        "quantile",
        1.0,
        {"light": 0.0, "moderate": 0.0, "severe": 0.0},
    )


def _coerce_thresholds(payload: object) -> SeverityThresholdConfig:
    if not isinstance(payload, dict):
        raise ValueError("Threshold payload must be a JSON object.")
    return SeverityThresholdConfig(
        light=float(payload["light"]),
        moderate=float(payload["moderate"]),
        severe=float(payload["severe"]),
    )


def _is_single_region_bootstrap(records: Iterable[object]) -> bool:
    parent_region_ids: set[str] = set()
    bootstrap_views: set[str] = set()
    count = 0
    for record in records:
        count += 1
        metadata = getattr(record, "metadata", {})
        parent_region_id = str(metadata.get("parent_region_id", "")).strip()
        bootstrap_view = str(metadata.get("bootstrap_view", "")).strip()
        if parent_region_id:
            parent_region_ids.add(parent_region_id)
        if bootstrap_view:
            bootstrap_views.add(bootstrap_view)
    return len(parent_region_ids) == 1 and count > 1 and len(bootstrap_views) >= 1


def _scale_thresholds(
    thresholds: SeverityThresholdConfig,
    *,
    scale: float,
) -> SeverityThresholdConfig:
    values = [
        float(thresholds.light) * scale,
        float(thresholds.moderate) * scale,
        float(thresholds.severe) * scale,
    ]
    values[1] = max(values[1], values[0] + 1e-6)
    values[2] = max(values[2], values[1] + 1e-6)
    return SeverityThresholdConfig(
        light=values[0],
        moderate=values[1],
        severe=values[2],
    )


def _build_judgement_config(
    base_config: AppConfig,
    *,
    ifi: SeverityThresholdConfig,
    mdi: SeverityThresholdConfig,
    iai: SeverityThresholdConfig,
    alignment_gap: SeverityThresholdConfig,
) -> AppConfig:
    return AppConfig(
        experiment=base_config.experiment,
        data=base_config.data,
        spatial_model=base_config.spatial_model,
        spatial_supervision=base_config.spatial_supervision,
        text_model=base_config.text_model,
        sentiment=base_config.sentiment,
        lost_space=base_config.lost_space,
        alignment=base_config.alignment,
        losses=base_config.losses,
        training=base_config.training,
        metrics=base_config.metrics,
        judgement=JudgementConfig(
            rule_mode=base_config.judgement.rule_mode,
            use_alignment_gap=base_config.judgement.use_alignment_gap,
            ifi=ifi,
            mdi=mdi,
            iai=iai,
            alignment_gap=alignment_gap,
        ),
        judgement_fusion=base_config.judgement_fusion,
        calibration=base_config.calibration,
        identity=base_config.identity,
        inference=base_config.inference,
    )


def _label_guided_threshold_search(
    *,
    base_config: AppConfig,
    calibration: CalibrationConfig,
    ifi_thresholds: SeverityThresholdConfig,
    mdi_thresholds: SeverityThresholdConfig,
    iai_thresholds: SeverityThresholdConfig,
    alignment_thresholds: SeverityThresholdConfig,
    ifi_values: list[float | None],
    mdi_values: list[float | None],
    iai_values: list[float | None],
    alignment_values: list[float | None],
    lost_space_targets: list[int | None],
    single_region_bootstrap: bool,
) -> tuple[
    SeverityThresholdConfig,
    SeverityThresholdConfig,
    SeverityThresholdConfig,
    SeverityThresholdConfig,
    dict[str, object] | None,
]:
    labeled_rows = [
        (ifi, mdi, iai, alignment, int(target))
        for ifi, mdi, iai, alignment, target in zip(
            ifi_values,
            mdi_values,
            iai_values,
            alignment_values,
            lost_space_targets,
            strict=True,
        )
        if target is not None
    ]
    if (
        not calibration.label_guided_search
        or len(labeled_rows) < calibration.label_guided_min_labels
        or not calibration.label_guided_scale_candidates
        or not calibration.label_guided_identity_scale_candidates
    ):
        return (
            ifi_thresholds,
            mdi_thresholds,
            iai_thresholds,
            alignment_thresholds,
            None,
        )

    def _objective(
        *,
        anchor_scale: float,
        identity_scale: float,
    ) -> tuple[float, int, int, dict[str, SeverityThresholdConfig]]:
        scaled_ifi = _scale_thresholds(ifi_thresholds, scale=anchor_scale)
        scaled_mdi = _scale_thresholds(mdi_thresholds, scale=anchor_scale)
        scaled_iai = _scale_thresholds(iai_thresholds, scale=identity_scale)
        scaled_alignment = _scale_thresholds(alignment_thresholds, scale=anchor_scale)
        preview_config = _build_judgement_config(
            base_config,
            ifi=scaled_ifi,
            mdi=scaled_mdi,
            iai=scaled_iai,
            alignment_gap=scaled_alignment,
        )
        absolute_error = 0.0
        over_prediction_count = 0
        exact_matches = 0
        for ifi, mdi, iai, alignment, target in labeled_rows:
            result = judge_region_metrics(
                ifi=ifi,
                mdi=mdi,
                alignment_gap=alignment,
                iai=iai,
                config=preview_config,
                single_region_bootstrap=single_region_bootstrap,
            )
            predicted = map_rule_level_to_class_index(
                result.final_level,
                base_config.lost_space.class_names,
            )
            if predicted is None:
                predicted = 0
            absolute_error += abs(predicted - target)
            over_prediction_count += int(predicted > target)
            exact_matches += int(predicted == target)
        regularization = (
            abs(anchor_scale - 1.0) * 0.05 + abs(identity_scale - 1.0) * 0.03
        )
        score = absolute_error + over_prediction_count * 0.75 - exact_matches * 0.01 + regularization
        return (
            score,
            over_prediction_count,
            exact_matches,
            {
                "ifi": scaled_ifi,
                "mdi": scaled_mdi,
                "iai": scaled_iai,
                "alignment_gap": scaled_alignment,
            },
        )

    baseline_score, baseline_over, baseline_matches, _ = _objective(
        anchor_scale=1.0,
        identity_scale=1.0,
    )
    best: tuple[float, int, int, float, float, dict[str, SeverityThresholdConfig]] | None = None
    for anchor_scale in calibration.label_guided_scale_candidates:
        for identity_scale in calibration.label_guided_identity_scale_candidates:
            score, over_count, exact_matches, scaled = _objective(
                anchor_scale=float(anchor_scale),
                identity_scale=float(identity_scale),
            )
            candidate = (
                score,
                over_count,
                -exact_matches,
                abs(float(anchor_scale) - 1.0) + abs(float(identity_scale) - 1.0),
                float(identity_scale),
                scaled,
            )
            if best is None or candidate < best:
                best = candidate

    if best is None or best[0] >= baseline_score:
        return (
            ifi_thresholds,
            mdi_thresholds,
            iai_thresholds,
            alignment_thresholds,
            {
                "enabled": True,
                "applied": False,
                "baseline_score": baseline_score,
                "baseline_over_prediction_count": baseline_over,
                "baseline_exact_match_count": baseline_matches,
                "labeled_count": len(labeled_rows),
            },
        )

    scaled_thresholds = best[-1]
    return (
        scaled_thresholds["ifi"],
        scaled_thresholds["mdi"],
        scaled_thresholds["iai"],
        scaled_thresholds["alignment_gap"],
        {
            "enabled": True,
            "applied": True,
            "baseline_score": baseline_score,
            "optimized_score": best[0],
            "labeled_count": len(labeled_rows),
            "anchor_scale": round(
                scaled_thresholds["ifi"].light / max(ifi_thresholds.light, 1e-6),
                6,
            ),
            "identity_scale": round(
                scaled_thresholds["iai"].light / max(iai_thresholds.light, 1e-6),
                6,
            ),
        },
    )


def resolve_thresholds_path(
    config: AppConfig,
    *,
    explicit_path: str | Path | None = None,
    default_split: str = DEFAULT_THRESHOLD_SPLIT,
) -> Path | None:
    candidates: list[Path] = []
    if explicit_path is not None:
        candidates.append(Path(explicit_path).resolve())
    elif config.inference.thresholds_path:
        candidates.append(Path(config.inference.thresholds_path).resolve())
    else:
        candidates.append(
            (config.output_dir / "calibration" / f"thresholds_{default_split}.json").resolve()
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def apply_calibrated_thresholds(
    config: AppConfig,
    *,
    thresholds_path: str | Path | None = None,
    default_split: str = DEFAULT_THRESHOLD_SPLIT,
) -> Path | None:
    resolved_path = resolve_thresholds_path(
        config,
        explicit_path=thresholds_path,
        default_split=default_split,
    )
    if resolved_path is None:
        return None

    payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    thresholds_payload = payload.get("thresholds")
    if not isinstance(thresholds_payload, dict):
        raise ValueError(
            f"Calibration file does not contain a 'thresholds' object: {resolved_path}"
        )

    config.judgement = JudgementConfig(
        rule_mode=config.judgement.rule_mode,
        use_alignment_gap=config.judgement.use_alignment_gap,
        ifi=_coerce_thresholds(thresholds_payload["ifi"]),
        mdi=_coerce_thresholds(thresholds_payload["mdi"]),
        iai=_coerce_thresholds(thresholds_payload["iai"]),
        alignment_gap=_coerce_thresholds(thresholds_payload["alignment_gap"]),
    )
    config.inference.thresholds_path = str(resolved_path)
    return resolved_path


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
        records = self.runtime.resolve_records(split)
        single_region_bootstrap = _is_single_region_bootstrap(records)
        ifi_values: list[float | None] = []
        mdi_values: list[float | None] = []
        iai_values: list[float | None] = []
        alignment_values: list[float | None] = []
        lost_space_targets: list[int | None] = []
        for batch, outputs in self.runtime.iterate_region_batches(
            split=split,
            checkpoint_path=checkpoint_path,
            allow_random_init=allow_random_init,
        ):
            ifi_values.extend(float(value) for value in outputs.ifi.detach().cpu().tolist())
            mdi_values.extend(float(value) for value in outputs.mdi.detach().cpu().tolist())
            iai_values.extend(float(value) for value in outputs.iai.detach().cpu().tolist())
            alignment_values.extend(
                float(value) for value in outputs.alignment_gap.detach().cpu().tolist()
            )
            if batch is None:
                lost_space_targets.extend([None] * outputs.ifi.shape[0])
            else:
                lost_space_targets.extend(
                    None if int(value) == self.config.lost_space.ignore_index else int(value)
                    for value in batch.targets["lost_space_label"].detach().cpu().tolist()
                )

        (
            ifi_thresholds,
            ifi_count,
            ifi_calibrated,
            ifi_mode,
            ifi_weight,
            ifi_uncertainty,
        ) = fit_thresholds_or_default(
            ifi_values,
            calibration=self.config.calibration,
            fallback=self.config.judgement.ifi,
        )
        (
            mdi_thresholds,
            mdi_count,
            mdi_calibrated,
            mdi_mode,
            mdi_weight,
            mdi_uncertainty,
        ) = fit_thresholds_or_default(
            mdi_values,
            calibration=self.config.calibration,
            fallback=self.config.judgement.mdi,
        )
        (
            iai_thresholds,
            iai_count,
            iai_calibrated,
            iai_mode,
            iai_weight,
            iai_uncertainty,
        ) = fit_thresholds_or_default(
            iai_values,
            calibration=self.config.calibration,
            fallback=self.config.judgement.iai,
        )
        (
            alignment_thresholds,
            alignment_count,
            alignment_calibrated,
            alignment_mode,
            alignment_weight,
            alignment_uncertainty,
        ) = fit_thresholds_or_default(
            alignment_values,
            calibration=self.config.calibration,
            fallback=self.config.judgement.alignment_gap,
        )
        (
            ifi_thresholds,
            mdi_thresholds,
            iai_thresholds,
            alignment_thresholds,
            label_guided_adjustment,
        ) = _label_guided_threshold_search(
            base_config=self.config,
            calibration=self.config.calibration,
            ifi_thresholds=ifi_thresholds,
            mdi_thresholds=mdi_thresholds,
            iai_thresholds=iai_thresholds,
            alignment_thresholds=alignment_thresholds,
            ifi_values=ifi_values,
            mdi_values=mdi_values,
            iai_values=iai_values,
            alignment_values=alignment_values,
            lost_space_targets=lost_space_targets,
            single_region_bootstrap=single_region_bootstrap,
        )

        calibrated = CalibratedThresholds(
            ifi=ifi_thresholds,
            mdi=mdi_thresholds,
            iai=iai_thresholds,
            alignment_gap=alignment_thresholds,
            sample_count=len(records),
        )
        destination = self._resolve_output_path(output_path, split=split)
        preview = self._build_preview(
            calibrated,
            ifi_values,
            mdi_values,
            iai_values,
            alignment_values,
            single_region_bootstrap=single_region_bootstrap,
        )
        payload = {
            "split": split,
            "sample_count": calibrated.sample_count,
            "thresholds": asdict(calibrated),
            "metric_sample_counts": {
                "ifi": ifi_count,
                "mdi": mdi_count,
                "iai": iai_count,
                "alignment_gap": alignment_count,
            },
            "calibrated_metrics": {
                "ifi": ifi_calibrated,
                "mdi": mdi_calibrated,
                "iai": iai_calibrated,
                "alignment_gap": alignment_calibrated,
            },
            "calibration_modes": {
                "ifi": ifi_mode,
                "mdi": mdi_mode,
                "iai": iai_mode,
                "alignment_gap": alignment_mode,
            },
            "effective_weights": {
                "ifi": ifi_weight,
                "mdi": mdi_weight,
                "iai": iai_weight,
                "alignment_gap": alignment_weight,
            },
            "threshold_uncertainty": {
                "ifi": ifi_uncertainty,
                "mdi": mdi_uncertainty,
                "iai": iai_uncertainty,
                "alignment_gap": alignment_uncertainty,
            },
            "bootstrap_context": {
                "single_region_bootstrap": single_region_bootstrap,
                "parent_region_count": len(
                    {
                        str(getattr(record, "metadata", {}).get("parent_region_id", "")).strip()
                        for record in records
                        if str(getattr(record, "metadata", {}).get("parent_region_id", "")).strip()
                    }
                ),
                "bootstrap_view_count": len(
                    {
                        str(getattr(record, "metadata", {}).get("bootstrap_view", "")).strip()
                        for record in records
                        if str(getattr(record, "metadata", {}).get("bootstrap_view", "")).strip()
                    }
                ),
            },
            "label_guided_adjustment": label_guided_adjustment,
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
        iai_values: list[float | None],
        alignment_values: list[float | None],
        *,
        single_region_bootstrap: bool,
    ) -> dict[str, int]:
        preview_config = _build_judgement_config(
            self.config,
            ifi=calibrated.ifi,
            mdi=calibrated.mdi,
            iai=calibrated.iai,
            alignment_gap=calibrated.alignment_gap,
        )
        counts = {"none": 0, "light": 0, "moderate": 0, "severe": 0}
        for ifi, mdi, iai, alignment in zip(
            ifi_values,
            mdi_values,
            iai_values,
            alignment_values,
            strict=True,
        ):
            result = judge_region_metrics(
                ifi=ifi,
                mdi=mdi,
                iai=iai,
                alignment_gap=alignment,
                config=preview_config,
                single_region_bootstrap=single_region_bootstrap,
            )
            counts[result.lost_space_level] += 1
        return counts
