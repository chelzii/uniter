from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from uniter.config import AppConfig
from uniter.inference.calibration import apply_calibrated_thresholds
from uniter.inference.exporter import build_region_metric_rows
from uniter.inference.judgement import map_rule_level_to_class_index
from uniter.inference.runtime import InferenceRuntime
from uniter.metrics.segmentation import SegmentationAccumulator

DEFAULT_SENTIMENT_CLASS_NAMES = ["negative", "neutral", "positive"]
DEFAULT_LOST_SPACE_CLASS_NAMES = ["none", "light", "moderate", "severe"]


def _numeric_summary(values: list[float | None]) -> dict[str, float | int | None]:
    cleaned = [float(value) for value in values if value is not None and not math.isnan(value)]
    if not cleaned:
        return {
            "count": 0,
            "mean": None,
            "min": None,
            "max": None,
        }
    return {
        "count": len(cleaned),
        "mean": sum(cleaned) / len(cleaned),
        "min": min(cleaned),
        "max": max(cleaned),
    }


def _classification_summary(
    rows: list[dict[str, Any]],
    *,
    prediction_key: str,
    target_key: str,
    class_names: list[str],
) -> dict[str, Any]:
    num_classes = len(class_names)
    confusion = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    labeled = []
    for row in rows:
        prediction = row.get(prediction_key)
        target = row.get(target_key)
        if prediction is None or target is None:
            continue
        if not isinstance(prediction, int) or not isinstance(target, int):
            continue
        if prediction < 0 or prediction >= num_classes or target < 0 or target >= num_classes:
            continue
        confusion[target][prediction] += 1
        labeled.append((prediction, target))

    if not labeled:
        return {
            "label_count": 0,
            "accuracy": None,
            "macro_f1": None,
            "confusion_matrix": None,
            "per_class": [],
        }

    correct = sum(int(prediction == target) for prediction, target in labeled)
    per_class: list[dict[str, float | int | str | None]] = []
    macro_f1 = 0.0
    for index, class_name in enumerate(class_names):
        true_positive = confusion[index][index]
        false_positive = sum(confusion[row][index] for row in range(num_classes)) - true_positive
        false_negative = sum(confusion[index][col] for col in range(num_classes)) - true_positive
        support = sum(confusion[index])
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive > 0
            else None
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative > 0
            else None
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall > 0
            else 0.0
        )
        macro_f1 += f1
        per_class.append(
            {
                "label": class_name,
                "support": support,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    return {
        "label_count": len(labeled),
        "accuracy": correct / len(labeled),
        "macro_f1": macro_f1 / max(num_classes, 1),
        "confusion_matrix": {
            "labels": class_names,
            "matrix": confusion,
        },
        "per_class": per_class,
    }


def _regression_summary(
    rows: list[dict[str, Any]],
    *,
    prediction_key: str,
    target_key: str,
) -> dict[str, float | int | None]:
    pairs: list[tuple[float, float]] = []
    for row in rows:
        prediction = row.get(prediction_key)
        target = row.get(target_key)
        if prediction is None or target is None:
            continue
        if math.isnan(float(prediction)) or math.isnan(float(target)):
            continue
        pairs.append((float(prediction), float(target)))
    if not pairs:
        return {
            "label_count": 0,
            "mae": None,
            "rmse": None,
        }
    absolute_errors = [abs(prediction - target) for prediction, target in pairs]
    squared_errors = [(prediction - target) ** 2 for prediction, target in pairs]
    return {
        "label_count": len(pairs),
        "mae": sum(absolute_errors) / len(absolute_errors),
        "rmse": math.sqrt(sum(squared_errors) / len(squared_errors)),
    }


def _average_ranks(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(ordered):
        next_index = index + 1
        while next_index < len(ordered) and ordered[next_index][1] == ordered[index][1]:
            next_index += 1
        average_rank = (index + next_index - 1) / 2.0 + 1.0
        for position in range(index, next_index):
            ranks[ordered[position][0]] = average_rank
        index = next_index
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    centered_x = [value - mean_x for value in xs]
    centered_y = [value - mean_y for value in ys]
    denominator = math.sqrt(
        sum(value * value for value in centered_x) * sum(value * value for value in centered_y)
    )
    if denominator <= 0.0:
        return None
    return sum(left * right for left, right in zip(centered_x, centered_y, strict=True)) / denominator


def _rank_correlation_summary(
    rows: list[dict[str, Any]],
    *,
    prediction_key: str,
    target_key: str,
) -> dict[str, float | int | None]:
    predictions: list[float] = []
    targets: list[float] = []
    for row in rows:
        prediction = row.get(prediction_key)
        target = row.get(target_key)
        if prediction is None or target is None:
            continue
        prediction_value = float(prediction)
        target_value = float(target)
        if math.isnan(prediction_value) or math.isnan(target_value):
            continue
        predictions.append(prediction_value)
        targets.append(target_value)
    if len(predictions) < 2:
        return {"label_count": len(predictions), "spearman": None}
    return {
        "label_count": len(predictions),
        "spearman": _pearson(_average_ranks(predictions), _average_ranks(targets)),
    }


def _binary_lost_space_summary(
    rows: list[dict[str, Any]],
    *,
    flag_key: str,
) -> dict[str, float | int | None]:
    labeled = [row for row in rows if row["lost_space_target"] in {0, 1}]
    if not labeled:
        return {"label_count": 0, "accuracy": None, "precision": None, "recall": None, "f1": None}

    true_positive = 0
    false_positive = 0
    false_negative = 0
    correct = 0
    for row in labeled:
        predicted = int(bool(row[flag_key]))
        target = int(row["lost_space_target"])
        correct += int(predicted == target)
        true_positive += int(predicted == 1 and target == 1)
        false_positive += int(predicted == 1 and target == 0)
        false_negative += int(predicted == 0 and target == 1)

    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive > 0
        else None
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative > 0
        else None
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall > 0
        else None
    )
    return {
        "label_count": len(labeled),
        "accuracy": correct / len(labeled),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _label_based_summary(
    rows: list[dict[str, Any]],
    *,
    prediction_label_key: str,
    target_key: str,
    class_names: list[str],
) -> dict[str, Any]:
    normalized_rows = []
    label_to_index = {label: index for index, label in enumerate(class_names)}
    for row in rows:
        target = row.get(target_key)
        prediction_label = row.get(prediction_label_key)
        if target is None:
            continue
        if not isinstance(prediction_label, str):
            continue
        if prediction_label not in label_to_index:
            mapped = map_rule_level_to_class_index(prediction_label, class_names)
            if mapped is None:
                continue
        else:
            mapped = label_to_index[prediction_label]
        normalized_rows.append(
            {
                "prediction": mapped,
                target_key: target,
            }
        )
    return _classification_summary(
        normalized_rows,
        prediction_key="prediction",
        target_key=target_key,
        class_names=class_names,
    )


def _ifi_component_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, dict[str, float]] = {}
    counts: dict[str, int] = {}
    for row in rows:
        raw_payload = row.get("ifi_components_json")
        if not isinstance(raw_payload, str) or not raw_payload:
            continue
        try:
            components = json.loads(raw_payload)
        except json.JSONDecodeError:
            continue
        if not isinstance(components, dict):
            continue
        for group_name, values in components.items():
            if not isinstance(values, dict):
                continue
            bucket = aggregate.setdefault(
                group_name,
                {
                    "actual": 0.0,
                    "target": 0.0,
                    "abs_delta": 0.0,
                    "weighted_delta": 0.0,
                },
            )
            bucket["actual"] += float(values.get("actual", 0.0))
            bucket["target"] += float(values.get("target", 0.0))
            bucket["abs_delta"] += float(values.get("abs_delta", 0.0))
            bucket["weighted_delta"] += float(values.get("weighted_delta", 0.0))
            counts[group_name] = counts.get(group_name, 0) + 1

    if not aggregate:
        return {"count": 0, "per_group": []}

    per_group = []
    for group_name, totals in aggregate.items():
        count = counts[group_name]
        per_group.append(
            {
                "group": group_name,
                "count": count,
                "mean_actual": totals["actual"] / count,
                "mean_target": totals["target"] / count,
                "mean_abs_delta": totals["abs_delta"] / count,
                "mean_weighted_delta": totals["weighted_delta"] / count,
            }
        )
    per_group.sort(key=lambda item: item["mean_weighted_delta"], reverse=True)
    return {
        "count": sum(counts.values()),
        "per_group": per_group,
    }


def _ifi_validation_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    bucket_totals = {
        "profile": [],
        "historical_plan": [],
        "geometry": [],
    }
    for row in rows:
        raw_payload = row.get("ifi_components_json")
        if not isinstance(raw_payload, str) or not raw_payload:
            continue
        try:
            components = json.loads(raw_payload)
        except json.JSONDecodeError:
            continue
        if not isinstance(components, dict):
            continue
        profile_values: list[float] = []
        plan_values: list[float] = []
        geometry_values: list[float] = []
        for group_name, values in components.items():
            if not isinstance(values, dict):
                continue
            weighted_delta = float(values.get("weighted_delta", 0.0))
            if group_name.startswith("historical_plan_"):
                plan_values.append(weighted_delta)
            elif "geometry" in group_name or "continuity" in group_name or "consistency" in group_name:
                geometry_values.append(weighted_delta)
            else:
                profile_values.append(weighted_delta)
        if profile_values:
            bucket_totals["profile"].append(sum(profile_values) / len(profile_values))
        if plan_values:
            bucket_totals["historical_plan"].append(sum(plan_values) / len(plan_values))
        if geometry_values:
            bucket_totals["geometry"].append(sum(geometry_values) / len(geometry_values))

    return {
        "proxy_score": _numeric_summary([row.get("spatial_proxy_score") for row in rows]),
        "lost_space_monotonicity": _rank_correlation_summary(
            rows,
            prediction_key="ifi",
            target_key="lost_space_target",
        ),
        "component_fit": {
            bucket: _numeric_summary(values)
            for bucket, values in bucket_totals.items()
        },
    }


def _mdi_validation_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    narrative_modes: dict[str, int] = {}
    for row in rows:
        source = row.get("mdi_source")
        if not isinstance(source, str):
            continue
        if "narrative" in source or "embedding" in source:
            narrative_modes[source] = narrative_modes.get(source, 0) + 1
    return {
        "sentiment_gap_regression": _regression_summary(
            rows,
            prediction_key="mdi",
            target_key="mdi_sentiment_gap_target",
        ),
        "sentiment_gap_monotonicity": _rank_correlation_summary(
            rows,
            prediction_key="mdi",
            target_key="mdi_sentiment_gap_target",
        ),
        "lost_space_monotonicity": _rank_correlation_summary(
            rows,
            prediction_key="mdi",
            target_key="lost_space_target",
        ),
        "narrative_mode_counts": narrative_modes,
    }


def _bootstrap_stability_summary(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        parent_region_id = row.get("parent_region_id")
        if not isinstance(parent_region_id, str) or not parent_region_id:
            continue
        grouped.setdefault(parent_region_id, []).append(row)
    if len(grouped) != 1:
        return None
    parent_region_id, group_rows = next(iter(grouped.items()))
    if len(group_rows) <= 1:
        return None

    def _spread(key: str) -> dict[str, float | int | None]:
        cleaned = [
            float(value)
            for value in (row.get(key) for row in group_rows)
            if value is not None and not math.isnan(float(value))
        ]
        if not cleaned:
            return {"count": 0, "mean": None, "range": None, "std": None}
        mean_value = sum(cleaned) / len(cleaned)
        variance = sum((value - mean_value) ** 2 for value in cleaned) / len(cleaned)
        return {
            "count": len(cleaned),
            "mean": mean_value,
            "range": max(cleaned) - min(cleaned),
            "std": math.sqrt(variance),
        }

    level_counts: dict[str, int] = {}
    for row in group_rows:
        level = str(row.get("lost_space_level"))
        level_counts[level] = level_counts.get(level, 0) + 1

    return {
        "single_region_bootstrap": True,
        "parent_region_id": parent_region_id,
        "view_count": len(group_rows),
        "level_counts": level_counts,
        "metrics": {
            "ifi": _spread("ifi"),
            "mdi": _spread("mdi"),
            "iai": _spread("iai"),
            "alignment_gap": _spread("alignment_gap"),
            "risk_score": _spread("risk_score"),
        },
    }


def summarize_region_rows(
    rows: list[dict[str, Any]],
    *,
    sentiment_class_names: list[str] | None = None,
    lost_space_class_names: list[str] | None = None,
    segmentation_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sentiment_names = sentiment_class_names or DEFAULT_SENTIMENT_CLASS_NAMES
    lost_space_names = lost_space_class_names or DEFAULT_LOST_SPACE_CLASS_NAMES
    level_counts: dict[str, int] = {}
    fusion_source_counts: dict[str, int] = {}
    mdi_mode_counts: dict[str, int] = {}
    identity_available_count = 0
    agreement_count = 0
    disagreement_count = 0
    sentiment_label_count = 0
    historical_sentiment_label_count = 0
    lost_space_label_count = 0

    for row in rows:
        level = str(row["lost_space_level"])
        level_counts[level] = level_counts.get(level, 0) + 1
        source = row.get("lost_space_fusion_source")
        if isinstance(source, str):
            fusion_source_counts[source] = fusion_source_counts.get(source, 0) + 1
        mdi_mode = row.get("mdi_mode")
        if isinstance(mdi_mode, str):
            mdi_mode_counts[mdi_mode] = mdi_mode_counts.get(mdi_mode, 0) + 1
        if bool(row.get("identity_available")):
            identity_available_count += 1
        if bool(row.get("has_sentiment_labels")):
            sentiment_label_count += 1
        if bool(row.get("has_historical_sentiment_labels")):
            historical_sentiment_label_count += 1
        if bool(row.get("has_lost_space_label")):
            lost_space_label_count += 1
        agreement = row.get("lost_space_rule_model_agreement")
        if agreement is True:
            agreement_count += 1
        elif agreement is False:
            disagreement_count += 1

    return {
        "proxy_score": _numeric_summary([row.get("spatial_proxy_score") for row in rows]),
        "region_count": len(rows),
        "bootstrap_stability": _bootstrap_stability_summary(rows),
        "metrics": {
            "ifi": _numeric_summary([row["ifi"] for row in rows]),
            "mdi": _numeric_summary([row["mdi"] for row in rows]),
            "iai": _numeric_summary([row.get("iai") for row in rows]),
            "alignment_gap": _numeric_summary([row["alignment_gap"] for row in rows]),
            "ifi_components": _ifi_component_summary(rows),
        },
        "indicator_validation": {
            "ifi": _ifi_validation_summary(rows),
            "mdi": _mdi_validation_summary(rows),
        },
        "identity": {
            "available_count": identity_available_count,
            "coverage_rate": identity_available_count / max(len(rows), 1),
            "iai_regression": _regression_summary(
                rows,
                prediction_key="iai",
                target_key="iai_target",
            ),
        },
        "data_capability": {
            "mdi_mode_counts": mdi_mode_counts,
            "sentiment_label_count": sentiment_label_count,
            "historical_sentiment_label_count": historical_sentiment_label_count,
            "lost_space_label_count": lost_space_label_count,
        },
        "current_sentiment": _classification_summary(
            rows,
            prediction_key="sentiment_pred",
            target_key="sentiment_target",
            class_names=sentiment_names,
        ),
        "historical_sentiment": _classification_summary(
            rows,
            prediction_key="historical_sentiment_pred",
            target_key="historical_sentiment_target",
            class_names=sentiment_names,
        ),
        "lost_space": {
            "level_counts": level_counts,
            "positive_rate": sum(int(row["lost_space_flag"]) for row in rows) / max(len(rows), 1),
            "fusion_source_counts": fusion_source_counts,
            "rule_model_agreement": {
                "agreement_count": agreement_count,
                "disagreement_count": disagreement_count,
            },
            "binary_final_metrics": _binary_lost_space_summary(rows, flag_key="lost_space_flag"),
            "binary_rule_metrics": _binary_lost_space_summary(
                rows,
                flag_key="rule_lost_space_flag",
            ),
            "final_metrics": _label_based_summary(
                rows,
                prediction_label_key="lost_space_level",
                target_key="lost_space_target",
                class_names=lost_space_names,
            ),
            "rule_level_metrics": _label_based_summary(
                rows,
                prediction_label_key="rule_lost_space_level",
                target_key="lost_space_target",
                class_names=lost_space_names,
            ),
            "model_metrics": _classification_summary(
                rows,
                prediction_key="lost_space_model_pred",
                target_key="lost_space_target",
                class_names=lost_space_names,
            ),
        },
        "segmentation": (
            {
                **(segmentation_summary or {}),
                "proxy_score": _numeric_summary([row.get("spatial_proxy_score") for row in rows]),
            }
            if segmentation_summary is not None
            else {
                "label_count": 0,
                "labeled_images": 0,
                "labeled_pixels": 0,
                "pixel_accuracy": None,
                "mIoU": None,
                "mean_dice": None,
                "proxy_score": _numeric_summary([row.get("spatial_proxy_score") for row in rows]),
                "per_class": [],
                "confusion_matrix": None,
            }
        ),
    }


class RegionEvaluator:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.thresholds_path = apply_calibrated_thresholds(config)
        self.runtime = InferenceRuntime(config)
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

    def evaluate(
        self,
        *,
        checkpoint_path: str | Path | None,
        split: str,
        output_path: str | Path | None,
        allow_random_init: bool = False,
    ) -> Path:
        rows: list[dict[str, Any]] = []
        class_names = [
            self.runtime.model.spatial_encoder.id2label[index]
            for index in sorted(self.runtime.model.spatial_encoder.id2label)
        ]
        segmentation = SegmentationAccumulator(
            num_classes=len(class_names),
            ignore_index=self.config.spatial_supervision.ignore_index,
        )
        for batch, outputs in self.runtime.iterate_region_batches(
            split=split,
            checkpoint_path=checkpoint_path,
            allow_random_init=allow_random_init,
        ):
            segmentation.update(outputs.segmentation_logits, batch.segmentation_labels)
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

        payload = {
            "split": split,
            "thresholds_path": str(self.thresholds_path) if self.thresholds_path is not None else None,
            "summary": summarize_region_rows(
                rows,
                sentiment_class_names=self.config.sentiment.class_names,
                lost_space_class_names=self.config.lost_space.class_names,
                segmentation_summary=segmentation.summarize(class_names=class_names),
            ),
        }
        destination = self._resolve_output_path(output_path, split=split)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return destination

    def _resolve_output_path(self, output_path: str | Path | None, *, split: str) -> Path:
        if output_path is not None:
            return Path(output_path).resolve()
        return (self.config.output_dir / "evaluations" / f"evaluation_{split}.json").resolve()
