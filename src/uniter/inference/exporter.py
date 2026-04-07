from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import torch

from uniter.config import AppConfig
from uniter.data.dataset import RegionBatch
from uniter.inference.calibration import apply_calibrated_thresholds
from uniter.inference.explanations import (
    build_region_decision_summary,
    serialize_ifi_components,
    top_ifi_groups,
)
from uniter.inference.judgement import fuse_region_judgement, judge_region_metrics
from uniter.inference.runtime import InferenceRuntime
from uniter.metrics.identity import build_identity_attribute_vector
from uniter.metrics.spatial import (
    build_adaptive_target_profiles,
    build_historical_plan_targets,
    compute_class_ratios,
    compute_ifi_components,
    reduce_to_label_groups,
)
from uniter.models.multimodal import ModelOutputs
from uniter.reporting.context import build_region_context


def _optional_float(value: torch.Tensor, index: int) -> float | None:
    scalar = float(value[index].detach().cpu().item())
    if math.isnan(scalar):
        return None
    return scalar


def _optional_int_target(
    value: torch.Tensor,
    index: int,
    *,
    ignore_index: int,
) -> int | None:
    scalar = int(value[index].detach().cpu().item())
    if scalar == ignore_index:
        return None
    return scalar


def _serialize_logits(logits: torch.Tensor | None) -> str | None:
    if logits is None:
        return None
    values = [round(float(item), 6) for item in logits.detach().cpu().tolist()]
    return json.dumps(values, ensure_ascii=False)


def _serialize_json_value(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _public_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metadata.items()
        if not str(key).startswith("_")
    }


def _prediction_from_logits(
    logits: torch.Tensor | None,
    *,
    class_names: list[str],
) -> tuple[int | None, str | None, float | None, float | None]:
    if logits is None:
        return None, None, None, None

    probabilities = torch.softmax(logits.detach().float(), dim=-1)
    prediction = int(probabilities.argmax(dim=-1).detach().cpu().item())
    label = class_names[prediction] if prediction < len(class_names) else str(prediction)
    confidence = float(probabilities[prediction].detach().cpu().item())
    expected_index = float(
        (
            probabilities
            * torch.arange(
                probabilities.shape[-1],
                device=probabilities.device,
                dtype=probabilities.dtype,
            )
        )
        .sum()
        .detach()
        .cpu()
        .item()
    )
    return prediction, label, confidence, expected_index


def _resolve_mdi_source(
    outputs: ModelOutputs,
    region_index: int,
    *,
    mdi_mode: str,
) -> str:
    if not bool(outputs.historical_region_mask[region_index].detach().cpu().item()):
        return "unavailable"
    has_embedding_drift = outputs.historical_text_features is not None
    has_distributional_sentiment = (
        bool(outputs.mdi_sentiment_mask[region_index].detach().cpu().item())
        and outputs.current_sentiment_record_logits is not None
        and outputs.historical_sentiment_record_logits is not None
    )
    has_sentiment_drift = bool(outputs.mdi_sentiment_mask[region_index].detach().cpu().item())
    if mdi_mode == "thesis":
        if has_distributional_sentiment and has_embedding_drift:
            return "sentiment_distribution_plus_narrative_drift"
        if has_sentiment_drift and has_embedding_drift:
            return "sentiment_plus_narrative_drift"
        if has_sentiment_drift:
            return "sentiment_drift"
        if has_embedding_drift:
            return "embedding_drift"
        return "unavailable"
    if (
        has_distributional_sentiment
    ):
        return "sentiment_distribution_drift"
    if has_sentiment_drift:
        return "sentiment_drift"
    if has_embedding_drift:
        return "embedding_drift"
    return "unavailable"


def _identity_vector_payload(
    metadata: dict[str, Any],
) -> tuple[dict[str, float] | None, dict[str, float], dict[str, float], dict[str, float], dict[str, float]]:
    attribute_vector = build_identity_attribute_vector(metadata)
    if attribute_vector is None:
        return None, {}, {}, {}, {}
    function_profile = {
        key.removeprefix("function_"): value
        for key, value in attribute_vector.items()
        if key.startswith("function_")
    }
    interface_profile = {
        key.removeprefix("interface_"): value
        for key, value in attribute_vector.items()
        if key.startswith("interface_")
    }
    structure_profile = {
        key.removeprefix("structure_"): value
        for key, value in attribute_vector.items()
        if key.startswith("structure_")
    }
    summary_profile = {
        key: value
        for key, value in attribute_vector.items()
        if not (
            key.startswith("function_")
            or key.startswith("interface_")
            or key.startswith("structure_")
        )
    }
    return (
        attribute_vector,
        function_profile,
        interface_profile,
        structure_profile,
        summary_profile,
    )


def build_region_metric_rows(
    batch: RegionBatch,
    outputs: ModelOutputs,
    *,
    split_by_region: dict[str, str],
    single_region_bootstrap: bool,
    sentiment_class_names: list[str],
    sentiment_ignore_index: int,
    lost_space_class_names: list[str],
    lost_space_ignore_index: int,
    spatial_id2label: dict[int, str],
    config: AppConfig,
    manifest_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    street_image_mask = ~batch.image_is_satellite
    satellite_image_mask = batch.image_is_satellite
    street_class_ratios = compute_class_ratios(
        outputs.segmentation_logits[street_image_mask],
        batch.image_region_index[street_image_mask],
        len(batch.region_ids),
    )
    per_image_region_index = torch.arange(
        outputs.segmentation_logits.shape[0],
        device=batch.image_region_index.device,
        dtype=batch.image_region_index.dtype,
    )
    per_image_class_ratios = compute_class_ratios(
        outputs.segmentation_logits,
        per_image_region_index,
        outputs.segmentation_logits.shape[0],
    )
    street_group_ratios = reduce_to_label_groups(
        street_class_ratios,
        id2label=spatial_id2label,
        label_groups=config.spatial_model.label_groups,
    )
    per_image_group_ratios = reduce_to_label_groups(
        per_image_class_ratios,
        id2label=spatial_id2label,
        label_groups=config.spatial_model.label_groups,
    )
    street_per_image_group_ratios = {
        name: values[street_image_mask]
        for name, values in per_image_group_ratios.items()
    }
    satellite_group_ratios = None
    satellite_per_image_group_ratios = None
    if torch.any(satellite_image_mask):
        satellite_class_ratios = compute_class_ratios(
            outputs.segmentation_logits[satellite_image_mask],
            batch.image_region_index[satellite_image_mask],
            len(batch.region_ids),
        )
        satellite_group_ratios = reduce_to_label_groups(
            satellite_class_ratios,
            id2label=spatial_id2label,
            label_groups=config.spatial_model.label_groups,
        )
        satellite_per_image_group_ratios = {
            name: values[satellite_image_mask]
            for name, values in per_image_group_ratios.items()
        }
    ifi_components = compute_ifi_components(
        street_group_ratios=street_group_ratios,
        street_target_profile=build_adaptive_target_profiles(
            base_profile=config.metrics.ifi_street_target_profile,
            metadata=batch.metadata,
            device=outputs.segmentation_logits.device,
            image_type="street",
        ),
        street_weights=config.metrics.ifi_street_weights,
        satellite_group_ratios=satellite_group_ratios,
        satellite_target_profile=build_adaptive_target_profiles(
            base_profile=config.metrics.ifi_satellite_target_profile,
            metadata=batch.metadata,
            device=outputs.segmentation_logits.device,
            image_type="satellite",
        ),
        satellite_weights=config.metrics.ifi_satellite_weights,
        cross_view_weights=config.metrics.ifi_cross_view_weights,
        per_image_street_group_ratios=street_per_image_group_ratios,
        street_image_region_index=batch.image_region_index[street_image_mask],
        street_image_point_ids=[
            point_id
            for point_id, is_satellite in zip(
                batch.image_point_ids,
                batch.image_is_satellite.tolist(),
                strict=True,
            )
            if not is_satellite
        ],
        street_image_view_directions=[
            direction
            for direction, is_satellite in zip(
                batch.image_view_directions,
                batch.image_is_satellite.tolist(),
                strict=True,
            )
            if not is_satellite
        ],
        street_image_longitudes=[
            longitude
            for longitude, is_satellite in zip(
                batch.image_longitudes,
                batch.image_is_satellite.tolist(),
                strict=True,
            )
            if not is_satellite
        ],
        street_image_latitudes=[
            latitude
            for latitude, is_satellite in zip(
                batch.image_latitudes,
                batch.image_is_satellite.tolist(),
                strict=True,
            )
            if not is_satellite
        ],
        per_image_satellite_group_ratios=satellite_per_image_group_ratios,
        satellite_image_region_index=(
            batch.image_region_index[satellite_image_mask]
            if torch.any(satellite_image_mask)
            else None
        ),
        historical_plan_targets=build_historical_plan_targets(
            metadata=batch.metadata,
            device=outputs.segmentation_logits.device,
        ),
    )

    rows: list[dict[str, Any]] = []
    for region_index, region_id in enumerate(batch.region_ids):
        region_metadata = batch.metadata[region_index]
        region_context = build_region_context(
            region_id=region_id,
            metadata=region_metadata,
            manifest_path=manifest_path,
        )
        current_logits = (
            outputs.sentiment_logits[region_index]
            if outputs.sentiment_logits is not None
            else None
        )
        historical_available = bool(
            outputs.historical_region_mask[region_index].detach().cpu().item()
        )
        historical_logits = (
            outputs.historical_sentiment_logits[region_index]
            if outputs.historical_sentiment_logits is not None and historical_available
            else None
        )
        sentiment_pred, sentiment_pred_label, _, _ = _prediction_from_logits(
            current_logits,
            class_names=sentiment_class_names,
        )
        mdi_source = _resolve_mdi_source(
            outputs,
            region_index,
            mdi_mode=config.metrics.mdi_mode,
        )
        (
            identity_attribute_vector,
            identity_function_profile,
            identity_interface_profile,
            identity_structure_profile,
            identity_summary_profile,
        ) = _identity_vector_payload(region_metadata)
        historical_sentiment_pred, historical_sentiment_pred_label, _, _ = _prediction_from_logits(
            historical_logits,
            class_names=sentiment_class_names,
        )
        lost_space_logits = (
            outputs.lost_space_logits[region_index]
            if outputs.lost_space_logits is not None
            else None
        )
        (
            lost_space_model_pred,
            lost_space_model_pred_label,
            lost_space_model_confidence,
            lost_space_model_expected_index,
        ) = _prediction_from_logits(
            lost_space_logits,
            class_names=lost_space_class_names,
        )
        ifi = _optional_float(outputs.ifi, region_index)
        mdi = _optional_float(outputs.mdi, region_index)
        iai = _optional_float(outputs.iai, region_index)
        iai_target = _optional_float(batch.targets["iai"], region_index)
        alignment_gap = _optional_float(outputs.alignment_gap, region_index)
        rule_judgement = judge_region_metrics(
            ifi=ifi,
            mdi=mdi,
            alignment_gap=alignment_gap,
            iai=iai,
            config=config,
            single_region_bootstrap=single_region_bootstrap,
        )
        fused_judgement = fuse_region_judgement(
            rule_judgement=rule_judgement,
            model_logits=lost_space_logits,
            iai=iai,
            class_names=lost_space_class_names,
            config=config,
        )
        sentiment_target = _optional_int_target(
            batch.targets["sentiment_label"],
            region_index,
            ignore_index=sentiment_ignore_index,
        )
        historical_sentiment_target = _optional_int_target(
            batch.targets["historical_sentiment_label"],
            region_index,
            ignore_index=sentiment_ignore_index,
        )
        mdi_sentiment_gap_target = None
        if sentiment_target is not None and historical_sentiment_target is not None:
            mdi_sentiment_gap_target = abs(sentiment_target - historical_sentiment_target) / max(
                len(sentiment_class_names) - 1,
                1,
            )
        lost_space_target = _optional_int_target(
            batch.targets["lost_space_label"],
            region_index,
            ignore_index=lost_space_ignore_index,
        )
        serialized_components = serialize_ifi_components(
            ifi_components,
            region_index=region_index,
        )
        ranked_groups = top_ifi_groups(serialized_components)
        row = {
            "region_id": region_id,
            "split": split_by_region[region_id],
            "ifi": ifi,
            "ifi_severity": rule_judgement.ifi.severity,
            "ifi_components_json": json.dumps(
                serialized_components,
                ensure_ascii=False,
                sort_keys=True,
            ),
            "ifi_top_groups": json.dumps(ranked_groups, ensure_ascii=False),
            "mdi": mdi,
            "mdi_severity": rule_judgement.mdi.severity,
            "mdi_source": mdi_source,
            "mdi_mode": config.metrics.mdi_mode,
            "mdi_sentiment_gap_target": mdi_sentiment_gap_target,
            "ifi_target_mode": "historical_plan_multiview_profile",
            "ifi_geometry_mode": "lonlat_distance_path_geometry",
            "iai": iai,
            "iai_severity": rule_judgement.iai.severity,
            "iai_target": iai_target,
            "iai_error": (
                abs(iai - iai_target)
                if iai is not None and iai_target is not None
                else None
            ),
            "identity_available": bool(
                outputs.identity_region_mask[region_index].detach().cpu().item()
            ),
            "spatial_proxy_score": _optional_float(
                outputs.spatial_proxy_score,
                region_index,
            )
            if outputs.spatial_proxy_score is not None
            else None,
            "identity_vector_mode": "structured_identity_attribute_vector",
            "identity_attribute_vector_json": _serialize_json_value(
                identity_attribute_vector
            ),
            "identity_function_profile_json": _serialize_json_value(
                identity_function_profile
            ),
            "identity_interface_profile_json": _serialize_json_value(
                identity_interface_profile
            ),
            "identity_structure_profile_json": _serialize_json_value(
                identity_structure_profile
            ),
            "identity_summary_profile_json": _serialize_json_value(
                identity_summary_profile
            ),
            "alignment_gap": alignment_gap,
            "alignment_gap_severity": rule_judgement.alignment_gap.severity,
            "primary_level": rule_judgement.primary_level,
            "rule_final_level": rule_judgement.final_level,
            "risk_score": rule_judgement.risk_score,
            "rule_lost_space_flag": rule_judgement.lost_space_flag,
            "rule_lost_space_level": rule_judgement.lost_space_level,
            "final_level": fused_judgement.final_level,
            "lost_space_flag": fused_judgement.final_flag,
            "lost_space_level": fused_judgement.final_level,
            "lost_space_fusion_source": fused_judgement.source,
            "lost_space_fusion_score": fused_judgement.fusion_score,
            "lost_space_rule_model_agreement": fused_judgement.agreement,
            "lost_space_model_logits": _serialize_logits(lost_space_logits),
            "lost_space_model_pred": lost_space_model_pred,
            "lost_space_model_pred_label": lost_space_model_pred_label,
            "lost_space_model_confidence": lost_space_model_confidence,
            "lost_space_model_expected_index": lost_space_model_expected_index,
            "sentiment_logits": _serialize_logits(current_logits),
            "sentiment_pred": sentiment_pred,
            "sentiment_pred_label": sentiment_pred_label,
            "historical_sentiment_logits": _serialize_logits(historical_logits),
            "historical_sentiment_pred": historical_sentiment_pred,
            "historical_sentiment_pred_label": historical_sentiment_pred_label,
            "sentiment_target": sentiment_target,
            "historical_sentiment_target": historical_sentiment_target,
            "lost_space_target": lost_space_target,
            "has_sentiment_labels": sentiment_target is not None,
            "has_historical_sentiment_labels": historical_sentiment_target is not None,
            "has_lost_space_label": lost_space_target is not None,
            "identity_enabled": config.identity.enabled,
            "parent_region_id": region_context["lookup_region_id"],
            "single_region_bootstrap": single_region_bootstrap,
            "region_name": region_context["region_name"],
            "site_name": region_context["site_name"],
            "city": region_context["city"],
            "district": region_context["district"],
            "region_type": region_context["region_type"],
            "image_count": region_context["image_count"],
            "current_text_count": region_context["current_text_count"],
            "historical_text_count": region_context["historical_text_count"],
            "identity_text_count": region_context["identity_text_count"],
            "used_image_count": region_context["used_image_count"],
            "selected_street_image_count": region_metadata.get("selected_street_image_count"),
            "selected_satellite_image_count": region_metadata.get(
                "selected_satellite_image_count"
            ),
            "parent_image_count": region_context["parent_image_count"],
            "raw_current_text_count": region_context["raw_current_text_count"],
            "cleaned_current_text_count": region_context["cleaned_current_text_count"],
            "parent_cleaned_current_text_count": region_context[
                "parent_cleaned_current_text_count"
            ],
            "data_sources": _serialize_json_value(region_context["data_sources"]),
            "time_range": _serialize_json_value(region_context["time_range"]),
            "current_text_source_platforms": _serialize_json_value(
                region_context["current_text_source_platforms"]
            ),
            "parent_current_text_source_platforms": _serialize_json_value(
                region_context["parent_current_text_source_platforms"]
            ),
            "image_source_platforms": _serialize_json_value(
                region_context["image_source_platforms"]
            ),
            "street_image_count": region_context["street_image_count"],
            "satellite_image_count": region_context["satellite_image_count"],
            "has_view_direction": region_context["has_view_direction"],
            "has_spatial_points": region_context["has_spatial_points"],
            "point_count": region_context["point_count"],
            "bad_image_skip_count": region_context["bad_image_skip_count"],
            "had_bad_image_skip": region_context["had_bad_image_skip"],
            "metadata_json": json.dumps(
                _public_metadata(region_metadata),
                ensure_ascii=False,
                sort_keys=True,
            ),
        }
        row["decision_summary"] = build_region_decision_summary(row)
        row["thresholds_path"] = config.inference.thresholds_path

        rows.append(row)
    return rows


class RegionMetricExporter:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.thresholds_path = apply_calibrated_thresholds(config)
        self.runtime = InferenceRuntime(config)
        self.output_dir = config.output_dir
        parent_region_ids = {
            str(record.metadata.get("parent_region_id", "")).strip()
            for record in self.runtime.records
            if str(record.metadata.get("parent_region_id", "")).strip()
        }
        bootstrap_views = {
            str(record.metadata.get("bootstrap_view", "")).strip()
            for record in self.runtime.records
            if str(record.metadata.get("bootstrap_view", "")).strip()
        }
        self.single_region_bootstrap = (
            len(parent_region_ids) == 1
            and len(self.runtime.records) > 1
            and len(bootstrap_views) >= 1
        )

    def export(
        self,
        *,
        output_path: str | Path | None,
        checkpoint_path: str | Path | None,
        split: str,
        allow_random_init: bool = False,
    ) -> Path:
        rows = self.collect_rows(
            checkpoint_path=checkpoint_path,
            split=split,
            allow_random_init=allow_random_init,
        )
        destination = self._resolve_output_path(output_path, split=split)
        self._write_rows(destination, rows)
        self.runtime.logger.info("Exported %d region rows to %s", len(rows), destination)
        return destination

    def collect_rows(
        self,
        *,
        checkpoint_path: str | Path | None,
        split: str,
        allow_random_init: bool = False,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for batch, outputs in self.runtime.iterate_region_batches(
            split=split,
            checkpoint_path=checkpoint_path,
            allow_random_init=allow_random_init,
        ):
            rows.extend(
                build_region_metric_rows(
                    batch,
                    outputs,
                    split_by_region=self.runtime.split_by_region,
                    single_region_bootstrap=self.single_region_bootstrap,
                    sentiment_class_names=self.config.sentiment.class_names,
                    sentiment_ignore_index=self.config.sentiment.ignore_index,
                    lost_space_class_names=self.config.lost_space.class_names,
                    lost_space_ignore_index=self.config.lost_space.ignore_index,
                    spatial_id2label=self.runtime.model.spatial_encoder.id2label,
                    config=self.config,
                    manifest_path=self.config.data.manifest_path,
                )
            )
        return rows

    def _resolve_output_path(self, output_path: str | Path | None, *, split: str) -> Path:
        if output_path is not None:
            return Path(output_path).resolve()
        return (self.output_dir / "exports" / f"region_metrics_{split}.csv").resolve()

    def _write_rows(self, output_path: Path, rows: list[dict[str, Any]]) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "region_id",
            "split",
            "parent_region_id",
            "single_region_bootstrap",
            "region_name",
            "site_name",
            "city",
            "district",
            "region_type",
            "image_count",
            "current_text_count",
            "historical_text_count",
            "identity_text_count",
            "used_image_count",
            "selected_street_image_count",
            "selected_satellite_image_count",
            "parent_image_count",
            "raw_current_text_count",
            "cleaned_current_text_count",
            "parent_cleaned_current_text_count",
            "ifi",
            "ifi_severity",
            "ifi_components_json",
            "ifi_top_groups",
            "mdi",
            "mdi_severity",
            "mdi_source",
            "mdi_mode",
            "mdi_sentiment_gap_target",
            "ifi_target_mode",
            "ifi_geometry_mode",
            "iai",
            "iai_severity",
            "iai_target",
            "iai_error",
            "identity_available",
            "identity_enabled",
            "spatial_proxy_score",
            "identity_vector_mode",
            "identity_attribute_vector_json",
            "identity_function_profile_json",
            "identity_interface_profile_json",
            "identity_structure_profile_json",
            "identity_summary_profile_json",
            "alignment_gap",
            "alignment_gap_severity",
            "primary_level",
            "rule_final_level",
            "final_level",
            "risk_score",
            "rule_lost_space_flag",
            "rule_lost_space_level",
            "lost_space_flag",
            "lost_space_level",
            "lost_space_fusion_source",
            "lost_space_fusion_score",
            "lost_space_rule_model_agreement",
            "lost_space_model_logits",
            "lost_space_model_pred",
            "lost_space_model_pred_label",
            "lost_space_model_confidence",
            "lost_space_model_expected_index",
            "sentiment_logits",
            "sentiment_pred",
            "sentiment_pred_label",
            "historical_sentiment_logits",
            "historical_sentiment_pred",
            "historical_sentiment_pred_label",
            "sentiment_target",
            "historical_sentiment_target",
            "lost_space_target",
            "has_sentiment_labels",
            "has_historical_sentiment_labels",
            "has_lost_space_label",
            "data_sources",
            "time_range",
            "current_text_source_platforms",
            "parent_current_text_source_platforms",
            "image_source_platforms",
            "street_image_count",
            "satellite_image_count",
            "has_view_direction",
            "has_spatial_points",
            "point_count",
            "bad_image_skip_count",
            "had_bad_image_skip",
            "decision_summary",
            "metadata_json",
            "thresholds_path",
        ]
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
