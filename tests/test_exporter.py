from __future__ import annotations

import json

import torch

from uniter.config import AppConfig
from uniter.data.dataset import RegionBatch
from uniter.inference.exporter import build_region_metric_rows
from uniter.models.multimodal import ModelOutputs


def test_build_region_metric_rows_serializes_predictions_and_masks_missing_history() -> None:
    config = AppConfig()
    config.inference.thresholds_path = "/tmp/thresholds_train.json"
    batch = RegionBatch(
        region_ids=["region_a", "region_b"],
        pixel_values=None,
        segmentation_labels=None,
        image_region_index=torch.tensor([0, 1]),
        image_is_satellite=torch.tensor([False, True]),
        image_point_ids=["p001", "p001"],
        image_view_directions=["north", "top"],
        image_longitudes=[108.95, 108.95],
        image_latitudes=[34.25, 34.25],
        current_input_ids=None,
        current_attention_mask=None,
        current_region_index=None,
        current_sentiment_labels=None,
        historical_input_ids=None,
        historical_attention_mask=None,
        historical_region_index=None,
        historical_sentiment_labels=None,
        identity_input_ids=None,
        identity_attention_mask=None,
        identity_region_index=None,
        metadata=[{"city": "西安"}, {"city": "咸阳"}],
        targets={
            "lost_space_label": torch.tensor([1, -100]),
            "sentiment_label": torch.tensor([2, -100]),
            "historical_sentiment_label": torch.tensor([0, -100]),
            "ifi": torch.tensor([float("nan"), float("nan")]),
            "mdi": torch.tensor([float("nan"), float("nan")]),
            "iai": torch.tensor([float("nan"), float("nan")]),
        },
    )
    batch.metadata[0]["_selected_identity_texts"] = [
        "宗教遗产侧翼线性界面",
        "游客支线通达与驻留节点",
    ]
    outputs = ModelOutputs(
        image_embeddings=torch.zeros(2, 2),
        text_embeddings=torch.zeros(2, 2),
        image_logits=torch.zeros(2, 2),
        segmentation_logits=torch.zeros(2, 2, 1, 1),
        satellite_region_mask=torch.tensor([False, True]),
        current_text_features=torch.zeros(2, 2),
        historical_text_features=torch.zeros(2, 2),
        identity_text_features=None,
        current_sentiment_record_logits=None,
        historical_sentiment_record_logits=None,
        sentiment_logits=torch.tensor([[0.1, 0.2, 0.9], [0.7, 0.2, 0.1]]),
        historical_sentiment_logits=torch.tensor([[0.8, 0.1, 0.1], [0.2, 0.6, 0.2]]),
        historical_region_mask=torch.tensor([True, False]),
        mdi_sentiment_mask=torch.tensor([True, False]),
        lost_space_logits=torch.tensor([[0.1, 0.8, 0.1, 0.0], [0.8, 0.1, 0.1, 0.0]]),
        identity_embeddings=None,
        identity_region_mask=torch.tensor([False, False]),
        alignment_gap=torch.tensor([0.2, 0.4]),
        ifi=torch.tensor([0.3, 0.5]),
        mdi=torch.tensor([0.8, float("nan")]),
        iai=torch.tensor([float("nan"), float("nan")]),
    )

    rows = build_region_metric_rows(
        batch,
        outputs,
        split_by_region={"region_a": "train", "region_b": "val"},
        single_region_bootstrap=False,
        sentiment_class_names=["negative", "neutral", "positive"],
        sentiment_ignore_index=-100,
        lost_space_class_names=["none", "light", "moderate", "severe"],
        lost_space_ignore_index=-100,
        spatial_id2label={0: "road", 1: "building"},
        config=config,
    )

    assert rows[0]["region_id"] == "region_a"
    assert rows[0]["split"] == "train"
    assert rows[0]["mdi_source"] == "sentiment_plus_narrative_drift"
    assert rows[0]["mdi_mode"] == "thesis"
    assert rows[0]["mdi_sentiment_gap_target"] == 1.0
    assert rows[0]["ifi_target_mode"] == "historical_plan_multiview_profile"
    assert rows[0]["ifi_geometry_mode"] == "lonlat_distance_path_geometry"
    assert rows[0]["primary_level"] in {"light", "moderate", "severe"}
    assert rows[0]["final_level"] in {"light", "moderate", "severe"}
    assert rows[0]["risk_score"] is not None
    assert rows[0]["lost_space_level"] in {"light", "moderate", "severe"}
    assert rows[0]["rule_lost_space_level"] in {"light", "moderate", "severe"}
    assert rows[0]["ifi_severity"] in {"none", "light", "moderate", "severe"}
    assert rows[0]["sentiment_pred"] == 2
    assert rows[0]["sentiment_pred_label"] == "positive"
    assert rows[0]["historical_sentiment_pred"] == 0
    assert rows[0]["historical_sentiment_pred_label"] == "negative"
    assert rows[0]["lost_space_model_pred"] == 1
    assert rows[0]["lost_space_model_pred_label"] == "light"
    assert rows[0]["lost_space_model_confidence"] is not None
    assert rows[0]["lost_space_fusion_source"] is not None
    assert rows[0]["decision_summary"]
    assert json.loads(rows[0]["ifi_components_json"])
    assert json.loads(rows[0]["ifi_top_groups"])
    assert json.loads(rows[0]["sentiment_logits"]) == [0.1, 0.2, 0.9]
    assert json.loads(rows[0]["historical_sentiment_logits"]) == [0.8, 0.1, 0.1]
    assert json.loads(rows[0]["lost_space_model_logits"]) == [0.1, 0.8, 0.1, 0.0]
    assert rows[0]["has_sentiment_labels"] is True
    assert rows[0]["has_historical_sentiment_labels"] is True
    assert rows[0]["used_image_count"] is not None
    assert rows[0]["parent_current_text_source_platforms"] is not None
    assert rows[0]["thresholds_path"] == "/tmp/thresholds_train.json"
    assert rows[0]["spatial_proxy_score"] is None
    assert rows[0]["identity_vector_mode"] == "structured_identity_attribute_vector"
    assert json.loads(rows[0]["identity_attribute_vector_json"])["function_religious_heritage"] > 0.0
    assert json.loads(rows[0]["identity_function_profile_json"])["religious_heritage"] > 0.0
    assert rows[1]["mdi_source"] == "unavailable"
    assert rows[1]["historical_sentiment_logits"] is None
    assert rows[1]["historical_sentiment_pred"] is None
    assert "_selected_current_texts" not in json.loads(rows[0]["metadata_json"])
