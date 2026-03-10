from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from uniter.config import AppConfig, SeverityThresholdConfig

RULE_LEVELS = ["none", "light", "moderate", "severe"]


@dataclass(frozen=True, slots=True)
class IndicatorJudgement:
    value: float | None
    severity: str


@dataclass(frozen=True, slots=True)
class RegionJudgement:
    ifi: IndicatorJudgement
    mdi: IndicatorJudgement
    alignment_gap: IndicatorJudgement
    iai: IndicatorJudgement
    lost_space_level: str
    lost_space_flag: bool


@dataclass(frozen=True, slots=True)
class FusedRegionJudgement:
    rule_level: str
    rule_flag: bool
    final_level: str
    final_flag: bool
    final_index: int | None
    model_pred_index: int | None
    model_pred_label: str | None
    model_confidence: float | None
    model_expected_index: float | None
    used_model_signal: bool
    used_iai_signal: bool
    agreement: bool | None
    source: str
    fusion_score: float | None


def _normalize_value(value: float | None) -> float | None:
    if value is None:
        return None
    if math.isnan(value):
        return None
    return float(value)


def _level_to_binary_index(level: str) -> int:
    return 0 if level == "none" else 1


def _binary_label_to_flag(index: int) -> bool:
    return index > 0


def _class_index_to_label(index: int, class_names: list[str]) -> str:
    if 0 <= index < len(class_names):
        return class_names[index]
    return str(index)


def map_rule_level_to_class_index(level: str, class_names: list[str]) -> int | None:
    if len(class_names) == 2:
        return _level_to_binary_index(level)
    if level in class_names:
        return class_names.index(level)
    if level in RULE_LEVELS:
        return RULE_LEVELS.index(level)
    return None


def class_index_to_rule_level(index: int, class_names: list[str]) -> str:
    if len(class_names) == 2:
        return "none" if index <= 0 else "light"
    if 0 <= index < len(class_names):
        return class_names[index]
    if 0 <= index < len(RULE_LEVELS):
        return RULE_LEVELS[index]
    return "none"


def class_index_to_flag(index: int, class_names: list[str]) -> bool:
    if len(class_names) == 2:
        return _binary_label_to_flag(index)
    label = _class_index_to_label(index, class_names)
    return label != "none"


def classify_indicator(
    value: float | None,
    thresholds: SeverityThresholdConfig,
) -> IndicatorJudgement:
    normalized = _normalize_value(value)
    if normalized is None:
        return IndicatorJudgement(value=None, severity="unavailable")
    if normalized >= thresholds.severe:
        return IndicatorJudgement(value=normalized, severity="severe")
    if normalized >= thresholds.moderate:
        return IndicatorJudgement(value=normalized, severity="moderate")
    if normalized >= thresholds.light:
        return IndicatorJudgement(value=normalized, severity="light")
    return IndicatorJudgement(value=normalized, severity="none")


def _base_primary_level(ifi: IndicatorJudgement, mdi: IndicatorJudgement) -> str:
    primary_severities = [ifi.severity, mdi.severity]
    if "severe" in primary_severities:
        return "severe"
    moderate_count = sum(severity == "moderate" for severity in primary_severities)
    light_count = sum(severity == "light" for severity in primary_severities)
    if moderate_count == 2:
        return "moderate"
    if moderate_count == 1 or light_count == 2:
        return "light"
    return "none"


def _upgrade_level(level: str) -> str:
    if level == "none":
        return "light"
    if level == "light":
        return "moderate"
    if level == "moderate":
        return "severe"
    return "severe"


def _softmax_confidence(
    logits: torch.Tensor | None,
) -> tuple[int | None, float | None, float | None]:
    if logits is None:
        return None, None, None
    probabilities = torch.softmax(logits.detach().float(), dim=-1)
    prediction = int(probabilities.argmax(dim=-1).item())
    confidence = float(probabilities[prediction].item())
    expected_index = float(
        torch.arange(
            probabilities.shape[-1],
            device=probabilities.device,
            dtype=probabilities.dtype,
        )
        .mul(probabilities)
        .sum()
        .item()
    )
    return prediction, confidence, expected_index


def judge_region_metrics(
    *,
    ifi: float | None,
    mdi: float | None,
    alignment_gap: float | None,
    iai: float | None = None,
    config: AppConfig,
) -> RegionJudgement:
    ifi_result = classify_indicator(ifi, config.judgement.ifi)
    mdi_result = classify_indicator(mdi, config.judgement.mdi)
    alignment_result = classify_indicator(
        alignment_gap,
        config.judgement.alignment_gap,
    )
    iai_result = classify_indicator(iai, config.judgement.iai)

    final_level = _base_primary_level(ifi_result, mdi_result)
    if config.judgement.use_alignment_gap:
        if alignment_result.severity == "severe" and final_level in {"light", "moderate"}:
            final_level = _upgrade_level(final_level)
        elif alignment_result.severity == "moderate" and final_level == "light":
            final_level = "moderate"

    return RegionJudgement(
        ifi=ifi_result,
        mdi=mdi_result,
        alignment_gap=alignment_result,
        iai=iai_result,
        lost_space_level=final_level,
        lost_space_flag=final_level != "none",
    )


def fuse_region_judgement(
    *,
    rule_judgement: RegionJudgement,
    model_logits: torch.Tensor | None,
    iai: float | None,
    class_names: list[str],
    config: AppConfig,
) -> FusedRegionJudgement:
    rule_index = map_rule_level_to_class_index(rule_judgement.lost_space_level, class_names)
    if rule_index is None:
        raise ValueError(f"Unsupported rule level '{rule_judgement.lost_space_level}'.")

    if not config.judgement_fusion.enabled:
        return FusedRegionJudgement(
            rule_level=rule_judgement.lost_space_level,
            rule_flag=rule_judgement.lost_space_flag,
            final_level=_class_index_to_label(rule_index, class_names),
            final_flag=class_index_to_flag(rule_index, class_names),
            final_index=rule_index,
            model_pred_index=None,
            model_pred_label=None,
            model_confidence=None,
            model_expected_index=None,
            used_model_signal=False,
            used_iai_signal=False,
            agreement=None,
            source="rule_only",
            fusion_score=float(rule_index),
        )

    model_pred_index, model_confidence, model_expected_index = _softmax_confidence(model_logits)
    used_model_signal = (
        model_pred_index is not None
        and model_confidence is not None
        and model_confidence >= config.judgement_fusion.minimum_model_confidence
    )
    used_iai_signal = False
    weighted_sum = float(rule_index) * config.judgement_fusion.rule_weight
    total_weight = config.judgement_fusion.rule_weight
    source_parts = ["rule"]

    if used_model_signal and model_expected_index is not None:
        weighted_sum += model_expected_index * config.judgement_fusion.model_weight
        total_weight += config.judgement_fusion.model_weight
        source_parts.append("model")

    if config.judgement_fusion.use_iai:
        iai_indicator = classify_indicator(iai, config.judgement.iai)
        iai_index = map_rule_level_to_class_index(iai_indicator.severity, class_names)
        if iai_index is not None and iai_indicator.value is not None:
            weighted_sum += float(iai_index) * config.judgement_fusion.iai_weight
            total_weight += config.judgement_fusion.iai_weight
            used_iai_signal = True
            source_parts.append("iai")

    fusion_score = weighted_sum / max(total_weight, 1e-6)
    final_index = int(round(fusion_score))
    final_index = min(max(final_index, 0), len(class_names) - 1)
    final_level = _class_index_to_label(final_index, class_names)
    model_pred_label = (
        _class_index_to_label(model_pred_index, class_names)
        if model_pred_index is not None
        else None
    )
    agreement = None
    if model_pred_index is not None:
        agreement = model_pred_index == rule_index

    return FusedRegionJudgement(
        rule_level=rule_judgement.lost_space_level,
        rule_flag=rule_judgement.lost_space_flag,
        final_level=final_level,
        final_flag=class_index_to_flag(final_index, class_names),
        final_index=final_index,
        model_pred_index=model_pred_index,
        model_pred_label=model_pred_label,
        model_confidence=model_confidence,
        model_expected_index=model_expected_index,
        used_model_signal=used_model_signal,
        used_iai_signal=used_iai_signal,
        agreement=agreement,
        source="+".join(source_parts),
        fusion_score=fusion_score,
    )
