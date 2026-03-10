from __future__ import annotations

import json
from typing import Any


def serialize_ifi_components(
    components: dict[str, dict[str, object]],
    *,
    region_index: int,
) -> dict[str, dict[str, float]]:
    serialized: dict[str, dict[str, float]] = {}
    for group_name, component in components.items():
        actual = float(component["actual"][region_index].detach().cpu().item())  # type: ignore[index]
        abs_delta = float(component["abs_delta"][region_index].detach().cpu().item())  # type: ignore[index]
        weighted_delta = float(
            component["weighted_delta"][region_index].detach().cpu().item()  # type: ignore[index]
        )
        serialized[group_name] = {
            "actual": actual,
            "target": float(component["target"]),
            "abs_delta": abs_delta,
            "weighted_delta": weighted_delta,
            "weight": float(component["weight"]),
        }
    return serialized


def top_ifi_groups(
    serialized_components: dict[str, dict[str, float]],
    *,
    top_k: int = 3,
) -> list[dict[str, float | str]]:
    ranked = sorted(
        serialized_components.items(),
        key=lambda item: item[1]["weighted_delta"],
        reverse=True,
    )
    return [
        {
            "group": name,
            "actual": values["actual"],
            "target": values["target"],
            "weighted_delta": values["weighted_delta"],
        }
        for name, values in ranked[:top_k]
    ]


def build_region_decision_summary(row: dict[str, Any]) -> str:
    parts = [
        f"final={row['lost_space_level']}",
        f"rule={row['rule_lost_space_level']}",
    ]
    model_label = row.get("lost_space_model_pred_label")
    if model_label:
        parts.append(f"model={model_label}")
    if row.get("lost_space_fusion_source"):
        parts.append(f"source={row['lost_space_fusion_source']}")
    confidence = row.get("lost_space_model_confidence")
    if isinstance(confidence, (int, float)):
        parts.append(f"confidence={confidence:.3f}")

    top_groups_raw = row.get("ifi_top_groups")
    if isinstance(top_groups_raw, str) and top_groups_raw:
        try:
            top_groups = json.loads(top_groups_raw)
        except json.JSONDecodeError:
            top_groups = []
        if top_groups:
            group_names = ",".join(str(item["group"]) for item in top_groups[:3])
            parts.append(f"ifi_top={group_names}")
    return "; ".join(parts)
