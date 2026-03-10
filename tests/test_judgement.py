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
        config=config,
    )

    assert result.ifi.severity == "moderate"
    assert result.mdi.severity == "none"
    assert result.alignment_gap.severity == "moderate"
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


def test_fuse_region_judgement_combines_rule_and_model() -> None:
    config = AppConfig()
    rule = judge_region_metrics(
        ifi=0.36,
        mdi=0.10,
        alignment_gap=0.45,
        iai=None,
        config=config,
    )

    fused = fuse_region_judgement(
        rule_judgement=rule,
        model_logits=torch.tensor([0.1, 0.2, 4.7, 0.0]),
        iai=None,
        class_names=["none", "light", "moderate", "severe"],
        config=config,
    )

    assert fused.final_level in {"light", "moderate", "severe"}
    assert fused.model_pred_label == "moderate"
    assert fused.used_model_signal is True
