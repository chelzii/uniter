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
    primary_level: str
    final_level: str
    risk_score: float
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


SEVERITY_TO_SCORE = {
    "unavailable": 0,
    "none": 0,
    "light": 1,
    "moderate": 2,
    "severe": 3,
}


def _base_primary_level(
    ifi: IndicatorJudgement,
    mdi: IndicatorJudgement,
    iai: IndicatorJudgement,
) -> str:
    anchor_indicators = [ifi, mdi]
    anchor_scores = [
        SEVERITY_TO_SCORE[indicator.severity]
        for indicator in anchor_indicators
        if indicator.value is not None and indicator.severity not in {"unavailable", "none"}
    ]
    support_score = (
        SEVERITY_TO_SCORE[iai.severity]
        if iai.value is not None and iai.severity not in {"unavailable", "none"}
        else 0
    )

    if not anchor_scores and support_score == 0:
        return "none"
    if not anchor_scores:
        return "light"

    anchor_max = max(anchor_scores)
    anchor_sum = sum(anchor_scores)

    if anchor_max >= 3 and anchor_sum >= 5:
        return "severe"
    if anchor_sum >= 4:
        return "moderate"
    if anchor_max >= 2 and support_score >= 2:
        return "moderate"
    if anchor_max >= 3:
        return "moderate"
    if anchor_sum >= 2:
        return "light"
    if anchor_max >= 1 and support_score >= 2:
        return "light"
    if anchor_max >= 1 or support_score >= 1:
        return "light"
    return "none"


def _thesis_primary_level(
    ifi: IndicatorJudgement,
    mdi: IndicatorJudgement,
    iai: IndicatorJudgement,
    *,
    single_region_bootstrap: bool = False,
) -> str:
    if single_region_bootstrap:
        anchor_indicators = (ifi, mdi)
        anchor_scores = [
            SEVERITY_TO_SCORE[indicator.severity]
            for indicator in anchor_indicators
            if indicator.value is not None and indicator.severity not in {"unavailable", "none"}
        ]
        support_score = (
            SEVERITY_TO_SCORE[iai.severity]
            if iai.value is not None and iai.severity not in {"unavailable", "none"}
            else 0
        )
        if not anchor_scores and support_score == 0:
            return "none"
        anchor_moderate_count = sum(score >= SEVERITY_TO_SCORE["moderate"] for score in anchor_scores)
        anchor_severe_count = sum(score >= SEVERITY_TO_SCORE["severe"] for score in anchor_scores)
        support_moderate = support_score >= SEVERITY_TO_SCORE["moderate"]
        support_active = support_score >= SEVERITY_TO_SCORE["light"]
        anchor_max = max(anchor_scores) if anchor_scores else 0

        # Bootstrap子区域共享同一父区域语义时，IAI更适合作为支持信号；
        # severe 需要至少一个锚定指标（IFI/MDI）足够强，moderate 允许锚定+支持共同成立。
        if anchor_severe_count >= 1 and (anchor_moderate_count >= 2 or support_moderate):
            return "severe"
        if anchor_moderate_count >= 2:
            return "moderate"
        if anchor_max >= SEVERITY_TO_SCORE["moderate"] and support_moderate:
            return "moderate"
        if anchor_max >= SEVERITY_TO_SCORE["severe"]:
            return "moderate"
        if anchor_max >= SEVERITY_TO_SCORE["moderate"]:
            return "light"
        if anchor_max >= SEVERITY_TO_SCORE["light"] and support_moderate:
            return "light"
        if anchor_max >= SEVERITY_TO_SCORE["light"] or support_active:
            return "light"
        return "none"

    primary_indicators = (ifi, mdi, iai)
    active_scores = [
        SEVERITY_TO_SCORE[indicator.severity]
        for indicator in primary_indicators
        if indicator.value is not None and indicator.severity not in {"unavailable", "none"}
    ]
    if not active_scores:
        return "none"
    if any(score >= SEVERITY_TO_SCORE["severe"] for score in active_scores):
        return "severe"
    if sum(score >= SEVERITY_TO_SCORE["moderate"] for score in active_scores) >= 2:
        return "moderate"
    return "light"


def _upgrade_level(level: str) -> str:
    if level == "none":
        return "light"
    if level == "light":
        return "moderate"
    if level == "moderate":
        return "severe"
    return "severe"


def _risk_score(
    *,
    ifi: IndicatorJudgement,
    mdi: IndicatorJudgement,
    iai: IndicatorJudgement,
    alignment_gap: IndicatorJudgement,
    use_alignment_gap: bool,
) -> float:
    primary_scores = [
        SEVERITY_TO_SCORE[indicator.severity]
        for indicator in (ifi, mdi, iai)
        if indicator.value is not None and indicator.severity != "unavailable"
    ]
    if not primary_scores:
        base_score = 0.0
    else:
        base_score = sum(primary_scores) / (len(primary_scores) * 3.0)
    alignment_boost = (
        {
            "none": 0.0,
            "light": 0.03,
            "moderate": 0.08,
            "severe": 0.15,
            "unavailable": 0.0,
        }[alignment_gap.severity]
        if use_alignment_gap
        else 0.0
    )
    return min(1.0, base_score + alignment_boost)


def _active_indicator_count(*indicators: IndicatorJudgement) -> int:
    return sum(
        indicator.value is not None and indicator.severity not in {"unavailable", "none"}
        for indicator in indicators
    )


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
    single_region_bootstrap: bool = False,
) -> RegionJudgement:
    ifi_result = classify_indicator(ifi, config.judgement.ifi)
    mdi_result = classify_indicator(mdi, config.judgement.mdi)
    alignment_result = classify_indicator(
        alignment_gap,
        config.judgement.alignment_gap,
    )
    iai_result = classify_indicator(iai, config.judgement.iai)

    if config.judgement.rule_mode == "thesis":
        primary_level = _thesis_primary_level(
            ifi_result,
            mdi_result,
            iai_result,
            single_region_bootstrap=single_region_bootstrap,
        )
    else:
        primary_level = _base_primary_level(ifi_result, mdi_result, iai_result)
    final_level = primary_level
    active_indicator_count = _active_indicator_count(ifi_result, mdi_result, iai_result)
    if config.judgement.rule_mode != "thesis" and config.judgement.use_alignment_gap:
        if (
            alignment_result.severity == "severe"
            and final_level == "moderate"
            and active_indicator_count >= 2
        ):
            final_level = "severe"
        elif alignment_result.severity == "severe" and final_level == "light":
            final_level = _upgrade_level(final_level)
        elif (
            alignment_result.severity == "moderate"
            and final_level == "light"
            and active_indicator_count >= 1
        ):
            final_level = "moderate"
    risk_score = _risk_score(
        ifi=ifi_result,
        mdi=mdi_result,
        iai=iai_result,
        alignment_gap=alignment_result,
        use_alignment_gap=config.judgement.use_alignment_gap,
    )

    return RegionJudgement(
        ifi=ifi_result,
        mdi=mdi_result,
        alignment_gap=alignment_result,
        iai=iai_result,
        primary_level=primary_level,
        final_level=final_level,
        risk_score=risk_score,
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
    rule_index = map_rule_level_to_class_index(rule_judgement.final_level, class_names)
    if rule_index is None:
        raise ValueError(f"Unsupported rule level '{rule_judgement.final_level}'.")

    if not config.judgement_fusion.enabled:
        return FusedRegionJudgement(
            rule_level=rule_judgement.final_level,
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
    disagreement_gap = 0.0

    if used_model_signal and model_expected_index is not None:
        weighted_sum += model_expected_index * config.judgement_fusion.model_weight
        total_weight += config.judgement_fusion.model_weight
        disagreement_gap = abs(float(rule_index) - float(model_expected_index))
        source_parts.append("model")

    fusion_score = weighted_sum / max(total_weight, 1e-6)
    if config.judgement_fusion.use_iai:
        iai_indicator = classify_indicator(iai, config.judgement.iai)
        iai_bonus = {
            "none": 0.0,
            "light": 0.08,
            "moderate": 0.18,
            "severe": 0.28,
            "unavailable": 0.0,
        }[iai_indicator.severity]
        if iai_indicator.value is not None and iai_bonus > 0.0 and fusion_score >= 1.0:
            if disagreement_gap >= 1.5:
                iai_bonus *= 0.5
            fusion_score += iai_bonus * max(config.judgement_fusion.iai_weight, 0.0)
            used_iai_signal = True
            source_parts.append("iai")

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
        rule_level=rule_judgement.final_level,
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
