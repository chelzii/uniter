from __future__ import annotations

from uniter.inference.evaluator import summarize_region_rows


def test_summarize_region_rows_aggregates_metrics_and_labels() -> None:
    summary = summarize_region_rows(
        [
            {
                "ifi": 0.4,
                "mdi": 0.2,
                "iai": None,
                "iai_target": None,
                "alignment_gap": 0.3,
                "ifi_components_json": (
                    '{"enclosure":{"actual":0.3,"target":0.2,'
                    '"abs_delta":0.1,"weighted_delta":0.12}}'
                ),
                "sentiment_pred": 2,
                "sentiment_target": 2,
                "historical_sentiment_pred": 0,
                "historical_sentiment_target": 1,
                "lost_space_level": "moderate",
                "lost_space_flag": True,
                "rule_lost_space_level": "moderate",
                "rule_lost_space_flag": True,
                "lost_space_fusion_source": "rule+model",
                "lost_space_rule_model_agreement": True,
                "identity_available": False,
                "lost_space_model_pred": 2,
                "lost_space_target": 1,
            },
            {
                "ifi": 0.1,
                "mdi": None,
                "iai": None,
                "iai_target": None,
                "alignment_gap": 0.2,
                "ifi_components_json": (
                    '{"sky":{"actual":0.1,"target":0.2,'
                    '"abs_delta":0.1,"weighted_delta":0.08}}'
                ),
                "sentiment_pred": 0,
                "sentiment_target": None,
                "historical_sentiment_pred": None,
                "historical_sentiment_target": None,
                "lost_space_level": "none",
                "lost_space_flag": False,
                "rule_lost_space_level": "none",
                "rule_lost_space_flag": False,
                "lost_space_fusion_source": "rule",
                "lost_space_rule_model_agreement": False,
                "identity_available": True,
                "lost_space_model_pred": 0,
                "lost_space_target": 0,
            },
        ],
        lost_space_class_names=["none", "light", "moderate", "severe"],
    )

    assert summary["region_count"] == 2
    assert summary["metrics"]["ifi"]["count"] == 2
    assert summary["metrics"]["mdi"]["count"] == 1
    assert summary["current_sentiment"]["accuracy"] == 1.0
    assert summary["historical_sentiment"]["accuracy"] == 0.0
    assert summary["lost_space"]["level_counts"]["moderate"] == 1
    assert summary["lost_space"]["binary_rule_metrics"]["accuracy"] == 1.0
    assert summary["lost_space"]["final_metrics"]["label_count"] == 2
    assert summary["lost_space"]["model_metrics"]["label_count"] == 2
    assert summary["lost_space"]["model_metrics"]["macro_f1"] is not None
    assert summary["metrics"]["ifi_components"]["per_group"][0]["group"] in {"enclosure", "sky"}
    assert summary["identity"]["available_count"] == 1
