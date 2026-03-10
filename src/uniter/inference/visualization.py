from __future__ import annotations

import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image

from uniter.config import AppConfig
from uniter.inference.exporter import build_region_metric_rows
from uniter.inference.runtime import InferenceRuntime
from uniter.utils.device import move_region_batch_to_device


def _palette_color(index: int) -> tuple[int, int, int, int]:
    if index == 0:
        return (0, 0, 0, 0)
    red = (53 * index + 67) % 256
    green = (97 * index + 31) % 256
    blue = (193 * index + 19) % 256
    return (red, green, blue, 120)


def _mask_to_rgba(mask: torch.Tensor, *, num_labels: int) -> Image.Image:
    flat_mask = mask.reshape(-1).detach().cpu().tolist()
    palette = [_palette_color(index) for index in range(num_labels)]
    overlay = Image.new("RGBA", (mask.shape[1], mask.shape[0]))
    overlay.putdata([palette[int(label)] for label in flat_mask])
    return overlay


def _open_resized_image(path: Path, *, image_size: int) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB").resize((image_size, image_size))


def _render_segmentation_overlay(
    image: Image.Image,
    segmentation_logits: torch.Tensor,
    *,
    num_labels: int,
) -> Image.Image:
    resized_logits = F.interpolate(
        segmentation_logits.unsqueeze(0),
        size=(image.height, image.width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    mask = resized_logits.argmax(dim=0)
    overlay = _mask_to_rgba(mask, num_labels=num_labels)
    composed = image.convert("RGBA")
    composed.alpha_composite(overlay)
    return composed.convert("RGB")


def _markdown_metric_line(label: str, value: float | None, severity: str) -> str:
    numeric = "N/A" if value is None or math.isnan(value) else f"{value:.4f}"
    return f"- {label}: {numeric} (`{severity}`)"


def _optional_label(value: str | None) -> str:
    return value if value is not None else "N/A"


def _format_index_value(value: float | None) -> str:
    if value is None or math.isnan(value):
        return "N/A"
    return f"{value:.4f}"


def build_region_report_markdown(
    *,
    row: dict[str, Any],
    image_filenames: list[str],
) -> str:
    metadata = json.loads(row["metadata_json"])
    ifi_groups = []
    try:
        ifi_groups = json.loads(row.get("ifi_top_groups", "[]"))
    except json.JSONDecodeError:
        ifi_groups = []
    lines = [
        f"# Region Report: {row['region_id']}",
        "",
        f"- Split: `{row['split']}`",
        f"- Lost space flag: `{row['lost_space_flag']}`",
        f"- Lost space level: `{row['lost_space_level']}`",
        f"- Rule lost space level: `{row['rule_lost_space_level']}`",
        f"- Lost space model prediction: `{_optional_label(row['lost_space_model_pred_label'])}`",
        f"- Lost space fusion source: `{_optional_label(row.get('lost_space_fusion_source'))}`",
        f"- MDI source: `{row['mdi_source']}`",
        _markdown_metric_line("IFI", row["ifi"], row["ifi_severity"]),
        _markdown_metric_line("MDI", row["mdi"], row["mdi_severity"]),
        _markdown_metric_line("IAI", row["iai"], row["iai_severity"]),
        _markdown_metric_line(
            "Alignment gap",
            row["alignment_gap"],
            row["alignment_gap_severity"],
        ),
        "",
        "## Decision Summary",
        "",
        f"- Summary: `{row.get('decision_summary', 'N/A')}`",
        f"- Model confidence: `{_format_index_value(row.get('lost_space_model_confidence'))}`",
        f"- Fusion score: `{_format_index_value(row.get('lost_space_fusion_score'))}`",
        "",
        "## IFI Components",
        "",
    ]
    for item in ifi_groups[:3]:
        lines.append(
            "- "
            f"{item['group']}: actual={item['actual']:.4f} target={item['target']:.4f} "
            f"weighted_delta={item['weighted_delta']:.4f}"
        )

    lines.extend(
        [
            "",
        "## Sentiment",
        "",
        f"- Current prediction: `{_optional_label(row['sentiment_pred_label'])}`",
        f"- Historical prediction: `{_optional_label(row['historical_sentiment_pred_label'])}`",
        "",
        "## Identity",
        "",
        f"- Identity available: `{row['identity_available']}`",
        f"- IAI target: `{_format_index_value(row.get('iai_target'))}`",
        f"- IAI error: `{_format_index_value(row.get('iai_error'))}`",
        "",
        "## Metadata",
        "",
        ]
    )
    for key, value in metadata.items():
        lines.append(f"- {key}: `{value}`")

    lines.extend(["", "## Overlays", ""])
    for filename in image_filenames:
        lines.append(f"![{filename}]({filename})")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def build_index_markdown(rows: Iterable[dict[str, Any]]) -> str:
    lines = [
        "# Region Visualization Index",
        "",
        "| Region | Split | Level | IFI | MDI | Alignment | Report |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['region_id']} | {row['split']} | {row['lost_space_level']} | "
            f"{_format_index_value(row['ifi'])} | "
            f"{_format_index_value(row['mdi'])} | "
            f"{_format_index_value(row['alignment_gap'])} | "
            f"[report]({row['region_id']}/report.md) |"
        )
    return "\n".join(lines).strip() + "\n"


def _load_training_summaries(summary_dir: Path) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for path in sorted(summary_dir.glob("epoch_*.json")):
        summaries.append(json.loads(path.read_text(encoding="utf-8")))
    return summaries


def build_training_curves_svg(summaries: list[dict[str, Any]]) -> str:
    if not summaries:
        return ""

    metrics = [
        ("loss", "Loss"),
        ("ifi", "IFI"),
        ("mdi", "MDI"),
        ("alignment_gap", "Alignment Gap"),
    ]
    panel_width = 320
    panel_height = 220
    padding = 36
    width = panel_width * len(metrics)
    height = panel_height
    parts = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">'
        ),
        '<rect width="100%" height="100%" fill="#ffffff" />',
    ]

    epochs = [summary["epoch"] for summary in summaries]
    for metric_index, (metric_key, label) in enumerate(metrics):
        panel_x = metric_index * panel_width
        train_values = [summary["train"].get(metric_key) for summary in summaries]
        val_values = [
            summary["val"].get(metric_key) if summary.get("val") else None
            for summary in summaries
        ]
        series_values = [
            float(value)
            for value in train_values + val_values
            if value is not None and not math.isnan(float(value))
        ]
        if not series_values:
            continue

        minimum = min(series_values)
        maximum = max(series_values)
        span = max(maximum - minimum, 1e-6)
        epoch_min = min(epochs)
        epoch_denominator = max(len(epochs) - 1, 1)
        chart_left = panel_x + padding
        chart_top = 24
        chart_width = panel_width - padding * 1.5
        chart_height = panel_height - padding * 1.75

        parts.append(
            f'<text x="{chart_left}" y="18" font-size="14" fill="#111827">{label}</text>'
        )
        parts.append(
            f'<rect x="{chart_left}" y="{chart_top}" width="{chart_width}" '
            f'height="{chart_height}" fill="none" stroke="#d1d5db" stroke-width="1" />'
        )

        def build_polyline(
            values: list[float | None],
            color: str,
            *,
            chart_left: float = chart_left,
            chart_top: float = chart_top,
            chart_width: float = chart_width,
            chart_height: float = chart_height,
            minimum: float = minimum,
            span: float = span,
            epoch_min: int = epoch_min,
            epoch_denominator: int = epoch_denominator,
        ) -> None:
            coordinates: list[str] = []
            for epoch, value in zip(epochs, values, strict=True):
                if value is None:
                    continue
                numeric = float(value)
                x = chart_left + ((epoch - epoch_min) / epoch_denominator) * chart_width
                y = chart_top + chart_height - ((numeric - minimum) / span) * chart_height
                coordinates.append(f"{x:.2f},{y:.2f}")
            if len(coordinates) >= 2:
                parts.append(
                    f'<polyline fill="none" stroke="{color}" stroke-width="2" '
                    f'points="{" ".join(coordinates)}" />'
                )

        build_polyline(train_values, "#2563eb")
        build_polyline(val_values, "#dc2626")
        parts.append(
            f'<text x="{chart_left}" y="{panel_height - 12}" font-size="11" fill="#6b7280">'
            "blue=train red=val"
            "</text>"
        )

    parts.append("</svg>")
    return "\n".join(parts)


class VisualizationExporter:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.runtime = InferenceRuntime(config)

    def export(
        self,
        *,
        checkpoint_path: str | Path | None,
        split: str,
        output_dir: str | Path | None,
        allow_random_init: bool = False,
    ) -> Path:
        records = self.runtime.resolve_records(split)
        if not records:
            raise RuntimeError(f"No records found for split '{split}'.")
        self.runtime.prepare_model(
            checkpoint_path=checkpoint_path,
            allow_random_init=allow_random_init,
        )

        destination = self._resolve_output_dir(output_dir, split=split)
        destination.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, Any]] = []
        loader = self.runtime.build_loader(records, batch_size=1)

        self.runtime.model.eval()
        with torch.no_grad():
            for record, batch in zip(records, loader, strict=True):
                batch = move_region_batch_to_device(batch, self.runtime.device)
                outputs = self.runtime.model(batch)
                row = build_region_metric_rows(
                    batch,
                    outputs,
                    split_by_region=self.runtime.split_by_region,
                    sentiment_class_names=self.config.sentiment.class_names,
                    sentiment_ignore_index=self.config.sentiment.ignore_index,
                    lost_space_class_names=self.config.lost_space.class_names,
                    lost_space_ignore_index=self.config.lost_space.ignore_index,
                    spatial_id2label=self.runtime.model.spatial_encoder.id2label,
                    config=self.config,
                )[0]
                rows.append(row)

                region_dir = destination / record.region_id
                region_dir.mkdir(parents=True, exist_ok=True)
                image_paths = record.image_paths[: self.config.data.max_images_per_region]
                segmentation_logits = outputs.segmentation_logits.detach().cpu()
                image_filenames: list[str] = []
                for image_index, image_path in enumerate(image_paths):
                    base_image = _open_resized_image(
                        image_path,
                        image_size=self.config.data.image_size,
                    )
                    overlay = _render_segmentation_overlay(
                        base_image,
                        segmentation_logits[image_index],
                        num_labels=self.runtime.model.spatial_encoder.num_labels,
                    )
                    filename = f"overlay_{image_index + 1:02d}.png"
                    overlay.save(region_dir / filename)
                    image_filenames.append(filename)

                report = build_region_report_markdown(row=row, image_filenames=image_filenames)
                (region_dir / "report.md").write_text(report, encoding="utf-8")

        (destination / "index.md").write_text(build_index_markdown(rows), encoding="utf-8")
        self._write_training_curves(destination)
        return destination

    def _resolve_output_dir(self, output_dir: str | Path | None, *, split: str) -> Path:
        if output_dir is not None:
            return Path(output_dir).resolve()
        return (self.config.output_dir / "visualizations" / split).resolve()

    def _write_training_curves(self, destination: Path) -> None:
        summary_dir = self.config.output_dir / "summaries"
        summaries = _load_training_summaries(summary_dir)
        if not summaries:
            return
        svg = build_training_curves_svg(summaries)
        if svg:
            (destination / "training_curves.svg").write_text(svg, encoding="utf-8")
