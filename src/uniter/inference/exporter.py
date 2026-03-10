from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import torch

from uniter.config import AppConfig
from uniter.data.dataset import RegionBatch
from uniter.inference.explanations import (
    build_region_decision_summary,
    serialize_ifi_components,
    top_ifi_groups,
)
from uniter.inference.judgement import fuse_region_judgement, judge_region_metrics
from uniter.inference.runtime import InferenceRuntime
from uniter.metrics.spatial import (
    compute_class_ratios,
    compute_ifi_components,
    reduce_to_label_groups,
)
from uniter.models.multimodal import ModelOutputs


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


def _resolve_mdi_source(outputs: ModelOutputs, region_index: int) -> str:
    if not bool(outputs.historical_region_mask[region_index].detach().cpu().item()):
        return "unavailable"
    if outputs.sentiment_logits is not None and outputs.historical_sentiment_logits is not None:
        return "sentiment_drift"
    if outputs.historical_text_features is not None:
        return "embedding_drift"
    return "unavailable"


def build_region_metric_rows(
    batch: RegionBatch,
    outputs: ModelOutputs,
    *,
    split_by_region: dict[str, str],
    sentiment_class_names: list[str],
    sentiment_ignore_index: int,
    lost_space_class_names: list[str],
    lost_space_ignore_index: int,
    spatial_id2label: dict[int, str],
    config: AppConfig,
) -> list[dict[str, Any]]:
    class_ratios = compute_class_ratios(
        outputs.segmentation_logits,
        batch.image_region_index,
        len(batch.region_ids),
    )
    group_ratios = reduce_to_label_groups(
        class_ratios,
        id2label=spatial_id2label,
        label_groups=config.spatial_model.label_groups,
    )
    ifi_components = compute_ifi_components(
        group_ratios,
        target_profile=config.metrics.ifi_target_profile,
        weights=config.metrics.ifi_weights,
    )

    rows: list[dict[str, Any]] = []
    for region_index, region_id in enumerate(batch.region_ids):
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
        )
        fused_judgement = fuse_region_judgement(
            rule_judgement=rule_judgement,
            model_logits=lost_space_logits,
            iai=iai,
            class_names=lost_space_class_names,
            config=config,
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
            "mdi_source": _resolve_mdi_source(outputs, region_index),
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
            "alignment_gap": alignment_gap,
            "alignment_gap_severity": rule_judgement.alignment_gap.severity,
            "rule_lost_space_flag": rule_judgement.lost_space_flag,
            "rule_lost_space_level": rule_judgement.lost_space_level,
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
            "sentiment_target": _optional_int_target(
                batch.targets["sentiment_label"],
                region_index,
                ignore_index=sentiment_ignore_index,
            ),
            "historical_sentiment_target": _optional_int_target(
                batch.targets["historical_sentiment_label"],
                region_index,
                ignore_index=sentiment_ignore_index,
            ),
            "lost_space_target": _optional_int_target(
                batch.targets["lost_space_label"],
                region_index,
                ignore_index=lost_space_ignore_index,
            ),
            "metadata_json": json.dumps(
                batch.metadata[region_index],
                ensure_ascii=False,
                sort_keys=True,
            ),
        }
        row["decision_summary"] = build_region_decision_summary(row)

        rows.append(row)
    return rows


class RegionMetricExporter:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.runtime = InferenceRuntime(config)
        self.output_dir = config.output_dir

    def export(
        self,
        *,
        output_path: str | Path | None,
        checkpoint_path: str | Path | None,
        split: str,
        allow_random_init: bool = False,
    ) -> Path:
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
                    sentiment_class_names=self.config.sentiment.class_names,
                    sentiment_ignore_index=self.config.sentiment.ignore_index,
                    lost_space_class_names=self.config.lost_space.class_names,
                    lost_space_ignore_index=self.config.lost_space.ignore_index,
                    spatial_id2label=self.runtime.model.spatial_encoder.id2label,
                    config=self.config,
                )
            )

        destination = self._resolve_output_path(output_path, split=split)
        self._write_rows(destination, rows)
        self.runtime.logger.info("Exported %d region rows to %s", len(rows), destination)
        return destination

    def _resolve_output_path(self, output_path: str | Path | None, *, split: str) -> Path:
        if output_path is not None:
            return Path(output_path).resolve()
        return (self.output_dir / "exports" / f"region_metrics_{split}.csv").resolve()

    def _write_rows(self, output_path: Path, rows: list[dict[str, Any]]) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "region_id",
            "split",
            "ifi",
            "ifi_severity",
            "ifi_components_json",
            "ifi_top_groups",
            "mdi",
            "mdi_severity",
            "mdi_source",
            "iai",
            "iai_severity",
            "iai_target",
            "iai_error",
            "identity_available",
            "alignment_gap",
            "alignment_gap_severity",
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
            "decision_summary",
            "metadata_json",
        ]
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
