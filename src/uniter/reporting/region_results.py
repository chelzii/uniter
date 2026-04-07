from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def _decode_json_value(value: object) -> Any:
    if not isinstance(value, str) or not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _lost_space_label_text(value: object) -> str | None:
    if value is None or value == "":
        return None
    mapping = {0: "none", 1: "light", 2: "moderate", 3: "severe"}
    try:
        return mapping.get(int(value), str(value))
    except (TypeError, ValueError):
        return str(value)


def build_region_results_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result_rows: list[dict[str, Any]] = []
    for row in rows:
        result_rows.append(
            {
                "region_id": row["region_id"],
                "parent_region_id": row.get("parent_region_id"),
                "single_region_bootstrap": row.get("single_region_bootstrap"),
                "split": row["split"],
                "region_name": row.get("region_name"),
                "site_name": row.get("site_name"),
                "city": row.get("city"),
                "district": row.get("district"),
                "region_type": row.get("region_type"),
                "image_count": row.get("image_count"),
                "used_image_count": row.get("used_image_count"),
                "parent_image_count": row.get("parent_image_count"),
                "current_text_count": row.get("current_text_count"),
                "historical_text_count": row.get("historical_text_count"),
                "identity_text_count": row.get("identity_text_count"),
                "IFI": row.get("ifi"),
                "MDI": row.get("mdi"),
                "IAI": row.get("iai"),
                "alignment_gap": row.get("alignment_gap"),
                "selected_street_image_count": row.get("selected_street_image_count"),
                "selected_satellite_image_count": row.get("selected_satellite_image_count"),
                "primary_level": row.get("primary_level"),
                "rule_final_level": row.get("rule_final_level"),
                "final_level": row.get("final_level", row.get("lost_space_level")),
                "risk_score": row.get("risk_score"),
                "lost_space_target": row.get("lost_space_target"),
                "lost_space_target_label": _lost_space_label_text(row.get("lost_space_target")),
                "mdi_mode": row.get("mdi_mode"),
                "ifi_target_mode": row.get("ifi_target_mode"),
                "ifi_geometry_mode": row.get("ifi_geometry_mode"),
                "identity_enabled": row.get("identity_enabled"),
                "identity_available": row.get("identity_available"),
                "has_sentiment_labels": row.get("has_sentiment_labels"),
                "has_historical_sentiment_labels": row.get("has_historical_sentiment_labels"),
                "has_lost_space_label": row.get("has_lost_space_label"),
                "data_sources": row.get("data_sources"),
                "time_range": row.get("time_range"),
                "current_text_source_platforms": row.get("current_text_source_platforms"),
                "parent_current_text_source_platforms": row.get(
                    "parent_current_text_source_platforms"
                ),
                "street_image_count": row.get("street_image_count"),
                "satellite_image_count": row.get("satellite_image_count"),
                "has_view_direction": row.get("has_view_direction"),
                "has_spatial_points": row.get("has_spatial_points"),
                "point_count": row.get("point_count"),
                "cleaned_current_text_count": row.get("cleaned_current_text_count"),
                "parent_cleaned_current_text_count": row.get(
                    "parent_cleaned_current_text_count"
                ),
                "raw_current_text_count": row.get("raw_current_text_count"),
                "bad_image_skip_count": row.get("bad_image_skip_count"),
                "had_bad_image_skip": row.get("had_bad_image_skip"),
                "decision_summary": row.get("decision_summary"),
            }
        )
    return result_rows


def _markdown_summary(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Single Region Result Summary",
        "",
        f"- Region rows: `{len(rows)}`",
        "",
        "| Region | Split | Final | Risk | MDI Mode | Data Sources |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        data_sources = _decode_json_value(row.get("data_sources")) or []
        if isinstance(data_sources, list):
            sources_text = ", ".join(str(item) for item in data_sources)
        else:
            sources_text = str(data_sources)
        risk_score = row.get("risk_score")
        risk_text = "N/A" if risk_score is None else f"{float(risk_score):.4f}"
        lines.append(
            "| "
            f"{row['region_id']} | {row['split']} | {row.get('final_level', 'N/A')} | "
            f"{risk_text} | {row.get('mdi_mode', 'N/A')} | {sources_text} |"
        )
    return "\n".join(lines).strip() + "\n"


def export_region_results_table(
    *,
    rows: list[dict[str, Any]],
    output_path: str | Path,
    markdown_path: str | Path | None = None,
) -> tuple[Path, Path | None]:
    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    result_rows = build_region_results_rows(rows)
    fieldnames = list(result_rows[0].keys()) if result_rows else [
        "region_id",
        "parent_region_id",
        "single_region_bootstrap",
        "split",
        "region_name",
        "site_name",
        "city",
        "district",
        "region_type",
        "image_count",
        "used_image_count",
        "parent_image_count",
        "current_text_count",
        "historical_text_count",
        "identity_text_count",
        "IFI",
        "MDI",
        "IAI",
        "alignment_gap",
        "selected_street_image_count",
        "selected_satellite_image_count",
        "primary_level",
        "final_level",
        "rule_final_level",
        "risk_score",
        "lost_space_target",
        "lost_space_target_label",
        "mdi_mode",
        "ifi_target_mode",
        "ifi_geometry_mode",
        "identity_enabled",
        "identity_available",
        "has_sentiment_labels",
        "has_historical_sentiment_labels",
        "has_lost_space_label",
        "data_sources",
        "time_range",
        "current_text_source_platforms",
        "parent_current_text_source_platforms",
        "street_image_count",
        "satellite_image_count",
        "has_view_direction",
        "has_spatial_points",
        "point_count",
        "cleaned_current_text_count",
        "parent_cleaned_current_text_count",
        "raw_current_text_count",
        "bad_image_skip_count",
        "had_bad_image_skip",
        "decision_summary",
    ]
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result_rows)

    final_markdown_path = None
    if markdown_path is not None:
        final_markdown_path = Path(markdown_path).resolve()
        final_markdown_path.parent.mkdir(parents=True, exist_ok=True)
        final_markdown_path.write_text(_markdown_summary(result_rows), encoding="utf-8")

    return destination, final_markdown_path
