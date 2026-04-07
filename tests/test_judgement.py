from __future__ import annotations

import torch

from uniter.config import AppConfig
from uniter.inference.judgement import (
    classify_indicator,
    fuse_region_judgement,
    judge_region_metrics,
)


def test_classify_indicator_returns_expected_severity() -> None:
    config = AppConfig()

    assert classify_indicator(0.10, config.judgement.ifi).severity == "none"
    assert classify_indicator(0.25, config.judgement.ifi).severity == "light"
    assert classify_indicator(0.40, config.judgement.ifi).severity == "moderate"
    assert classify_indicator(0.60, config.judgement.ifi).severity == "severe"


def test_judge_region_uses_alignment_gap_as_supporting_upgrade() -> None:
    config = AppConfig()

    result = judge_region_metrics(
        ifi=0.36,
        mdi=0.10,
        alignment_gap=0.45,
        iai=0.10,
        config=config,
    )

    assert result.ifi.severity == "moderate"
    assert result.mdi.severity == "none"
    assert result.alignment_gap.severity == "moderate"
    assert result.primary_level == "light"
    assert result.lost_space_level == "moderate"
    assert result.lost_space_flag is True


def test_judge_region_does_not_allow_alignment_gap_to_act_alone() -> None:
    config = AppConfig()

    result = judge_region_metrics(
        ifi=0.10,
        mdi=0.10,
        alignment_gap=0.70,
        config=config,
    )

    assert result.alignment_gap.severity == "severe"
    assert result.lost_space_level == "none"
    assert result.lost_space_flag is False


def test_judge_region_uses_iai_in_primary_rule() -> None:
    config = AppConfig()

    result = judge_region_metrics(
        ifi=0.36,
        mdi=0.10,
        alignment_gap=0.10,
        iai=0.40,
        config=config,
    )

    assert result.ifi.severity == "moderate"
    assert result.iai.severity == "moderate"
    assert result.primary_level == "moderate"
    assert result.lost_space_level == "moderate"
    assert result.risk_score > 0.0


def test_judge_region_limits_identity_only_signal_to_light() -> None:
    config = AppConfig()

    result = judge_region_metrics(
        ifi=0.10,
        mdi=0.10,
        alignment_gap=0.10,
        iai=0.60,
        config=config,
    )

    assert result.iai.severity == "severe"
    assert result.primary_level == "light"
    assert result.lost_space_level == "light"


def test_judge_region_supports_thesis_rule_mode() -> None:
    config = AppConfig()
    config.judgement.rule_mode = "thesis"
    config.judgement.use_alignment_gap = False

    result = judge_region_metrics(
        ifi=0.36,
        mdi=0.38,
        alignment_gap=0.70,
        iai=0.10,
        config=config,
    )

    assert result.ifi.severity == "moderate"
    assert result.mdi.severity == "moderate"
    assert result.alignment_gap.severity == "severe"
    assert result.primary_level == "moderate"
    assert result.lost_space_level == "moderate"
    assert result.lost_space_flag is True


def test_judge_region_thesis_rule_ignores_alignment_gap_for_final_decision() -> None:
    config = AppConfig()
    config.judgement.rule_mode = "thesis"
    config.judgement.use_alignment_gap = True

    result = judge_region_metrics(
        ifi=0.10,
        mdi=0.10,
        alignment_gap=0.70,
        iai=0.10,
        config=config,
    )

    assert result.alignment_gap.severity == "severe"
    assert result.lost_space_level == "none"
    assert result.lost_space_flag is False


def test_judge_region_thesis_rule_is_more_conservative_for_single_region_bootstrap() -> None:
    config = AppConfig()
    config.judgement.rule_mode = "thesis"
    config.judgement.use_alignment_gap = False

    result = judge_region_metrics(
        ifi=0.10,
        mdi=0.10,
        alignment_gap=0.10,
        iai=0.60,
        config=config,
        single_region_bootstrap=True,
    )

    assert result.iai.severity == "severe"
    assert result.primary_level == "light"
    assert result.lost_space_level == "light"


def test_judge_region_single_region_bootstrap_allows_anchor_plus_identity_to_reach_moderate() -> None:
    config = AppConfig()
    config.judgement.rule_mode = "thesis"
    config.judgement.use_alignment_gap = False

    result = judge_region_metrics(
        ifi=0.36,
        mdi=0.10,
        alignment_gap=0.10,
        iai=0.40,
        config=config,
        single_region_bootstrap=True,
    )

    assert result.ifi.severity == "moderate"
    assert result.iai.severity == "moderate"
    assert result.primary_level == "moderate"
    assert result.lost_space_level == "moderate"


def test_fuse_region_judgement_combines_rule_and_model() -> None:
    config = AppConfig()
    rule = judge_region_metrics(
        ifi=0.52,
        mdi=0.37,
        alignment_gap=0.56,
        iai=0.52,
        config=config,
    )

    fused = fuse_region_judgement(
        rule_judgement=rule,
        model_logits=torch.tensor([0.1, 4.8, 0.2, 0.0]),
        iai=0.52,
        class_names=["none", "light", "moderate", "severe"],
        config=config,
    )

    assert fused.final_level == "moderate"
    assert fused.model_pred_label == "light"
    assert fused.used_model_signal is True
    assert fused.used_iai_signal is True
