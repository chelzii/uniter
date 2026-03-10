from __future__ import annotations

from uniter.inference.visualization import (
    build_region_report_markdown,
    build_training_curves_svg,
)


def test_build_region_report_markdown_contains_metrics_and_images() -> None:
    markdown = build_region_report_markdown(
        row={
            "region_id": "region_a",
            "split": "train",
            "lost_space_flag": True,
            "lost_space_level": "moderate",
            "rule_lost_space_level": "light",
            "lost_space_model_pred_label": "moderate",
            "lost_space_fusion_source": "rule+model",
            "lost_space_model_confidence": 0.88,
            "lost_space_fusion_score": 1.6,
            "decision_summary": "final=moderate; rule=light",
            "mdi_source": "sentiment_drift",
            "ifi": 0.42,
            "ifi_severity": "moderate",
            "ifi_top_groups": (
                '[{"group":"enclosure","actual":0.3,'
                '"target":0.2,"weighted_delta":0.12}]'
            ),
            "mdi": 0.18,
            "mdi_severity": "none",
            "iai": None,
            "iai_severity": "unavailable",
            "iai_target": None,
            "iai_error": None,
            "identity_available": False,
            "alignment_gap": 0.44,
            "alignment_gap_severity": "moderate",
            "sentiment_pred_label": "positive",
            "historical_sentiment_pred_label": "negative",
            "metadata_json": '{"city":"西安"}',
        },
        image_filenames=["overlay_01.png"],
    )

    assert "# Region Report: region_a" in markdown
    assert "Lost space level: `moderate`" in markdown
    assert "Rule lost space level: `light`" in markdown
    assert "## IFI Components" in markdown
    assert "![overlay_01.png](overlay_01.png)" in markdown


def test_build_training_curves_svg_returns_svg_markup() -> None:
    svg = build_training_curves_svg(
        [
            {"epoch": 1, "train": {"loss": 1.0, "ifi": 0.5, "mdi": 0.2, "alignment_gap": 0.4}},
            {
                "epoch": 2,
                "train": {"loss": 0.8, "ifi": 0.45, "mdi": 0.18, "alignment_gap": 0.35},
                "val": {"loss": 0.9, "ifi": 0.5, "mdi": 0.19, "alignment_gap": 0.36},
            },
        ]
    )

    assert svg.startswith('<svg xmlns="http://www.w3.org/2000/svg"')
    assert "Loss" in svg
    assert "Alignment Gap" in svg
