from __future__ import annotations

from collections.abc import Iterable
import re
from typing import Any

import torch
import torch.nn.functional as F

STREET_COMPONENT_NAMES = {
    "road_surface": "street_road_width_proxy",
    "enclosure": "street_enclosure_strength",
    "vegetation": "street_green_coverage",
    "sky": "street_openness",
    "mobility": "street_traffic_intrusion",
    "pedestrian": "street_pedestrian_activity",
}

SATELLITE_COMPONENT_NAMES = {
    "road_surface": "satellite_paved_corridor",
    "enclosure": "satellite_built_footprint",
    "vegetation": "satellite_green_coverage",
    "sky": "satellite_open_background",
    "mobility": "satellite_vehicle_exposure",
    "pedestrian": "satellite_pedestrian_activity",
}

HERITAGE_KEYWORDS = (
    "历史",
    "传统",
    "风貌",
    "遗产",
    "寺",
    "碑",
    "街区",
    "文化",
)
CONNECTIVITY_KEYWORDS = (
    "打通",
    "连通",
    "通行",
    "连接",
    "环线",
    "消防",
    "断头路",
    "道路宽度",
    "断面",
)
SLOW_TRAFFIC_KEYWORDS = (
    "步行",
    "慢行",
    "步行街",
    "石板路",
    "街巷",
    "慢行系统",
)
GREENING_KEYWORDS = ("景观", "绿化", "树", "院落", "环境", "清幽")
STONE_PAVING_KEYWORDS = ("石板路", "石板路面", "铺装", "沥青路面改造")
SHOWCASE_KEYWORDS = ("展示", "展现", "游线", "导览", "展示界面", "展示空间")
EMERGENCY_ACCESS_KEYWORDS = ("消防", "消防通行", "消防环线", "应急", "救援")
RESIDENT_SERVICE_KEYWORDS = ("居民生活", "便利居民生活", "便民", "生活服务")
STYLE_KEYWORDS = ("传统街巷风貌", "传统风貌", "历史风貌", "街巷风貌", "风貌延续", "风貌保护")
METRIC_WIDTH_PATTERN = re.compile(r"(\d+)\s*[—-]\s*(\d+)\s*米")


def compute_class_ratios(
    segmentation_logits: torch.Tensor,
    image_region_index: torch.Tensor,
    num_regions: int,
) -> torch.Tensor:
    if segmentation_logits.shape[0] == 0:
        return torch.zeros(
            num_regions,
            segmentation_logits.shape[1],
            device=segmentation_logits.device,
            dtype=torch.float32,
        )
    predicted = segmentation_logits.argmax(dim=1)
    num_labels = segmentation_logits.shape[1]
    counts_dtype = torch.float32
    per_image_counts = []
    for label_map in predicted:
        counts = torch.bincount(label_map.reshape(-1), minlength=num_labels).to(counts_dtype)
        per_image_counts.append(counts)
    image_counts = torch.stack(per_image_counts, dim=0)

    region_counts = torch.zeros(
        num_regions,
        num_labels,
        device=segmentation_logits.device,
        dtype=counts_dtype,
    )
    region_counts.index_add_(0, image_region_index, image_counts.to(segmentation_logits.device))
    return region_counts / region_counts.sum(dim=1, keepdim=True).clamp_min(1.0)


def compute_soft_class_ratios(
    segmentation_logits: torch.Tensor,
    image_region_index: torch.Tensor,
    num_regions: int,
) -> torch.Tensor:
    if segmentation_logits.shape[0] == 0:
        return torch.zeros(
            num_regions,
            segmentation_logits.shape[1],
            device=segmentation_logits.device,
            dtype=segmentation_logits.dtype,
        )
    probabilities = F.softmax(segmentation_logits, dim=1)
    per_image_class_mass = probabilities.mean(dim=(-2, -1))
    region_mass = torch.zeros(
        num_regions,
        per_image_class_mass.shape[-1],
        device=segmentation_logits.device,
        dtype=per_image_class_mass.dtype,
    )
    counts = torch.bincount(image_region_index, minlength=num_regions).to(
        device=segmentation_logits.device,
        dtype=per_image_class_mass.dtype,
    )
    region_mass.index_add_(0, image_region_index, per_image_class_mass)
    return region_mass / counts.clamp_min(1.0).unsqueeze(-1)


def reduce_to_label_groups(
    class_ratios: torch.Tensor,
    *,
    id2label: dict[int, str],
    label_groups: dict[str, list[str]],
) -> dict[str, torch.Tensor]:
    label_to_id = {label: idx for idx, label in id2label.items()}
    group_values: dict[str, torch.Tensor] = {}
    for group_name, labels in label_groups.items():
        indices = [label_to_id[label] for label in labels if label in label_to_id]
        if not indices:
            group_values[group_name] = torch.zeros(
                class_ratios.shape[0],
                device=class_ratios.device,
            )
            continue
        group_values[group_name] = class_ratios[:, indices].sum(dim=1)
    return group_values


def _component_weight(weights: dict[str, float], *names: str, default: float = 1.0) -> float:
    selected = [float(weights[name]) for name in names if name in weights]
    if not selected:
        return default
    return sum(selected) / len(selected)


def _region_variability(
    per_image_group_ratios: dict[str, torch.Tensor],
    *,
    image_region_index: torch.Tensor,
    num_regions: int,
    groups: tuple[str, ...],
) -> torch.Tensor:
    device = image_region_index.device
    accumulated = torch.zeros(num_regions, device=device, dtype=torch.float32)
    valid_group_count = 0
    ones = torch.ones_like(image_region_index, dtype=torch.float32)
    image_counts = torch.zeros(num_regions, device=device, dtype=torch.float32)
    image_counts.index_add_(0, image_region_index, ones)
    image_counts = image_counts.clamp_min(1.0)

    for group_name in groups:
        values = per_image_group_ratios.get(group_name)
        if values is None:
            continue
        values = values.to(device=device, dtype=torch.float32)
        region_sum = torch.zeros(num_regions, device=device, dtype=torch.float32)
        region_sum.index_add_(0, image_region_index, values)
        region_mean = region_sum / image_counts
        deviation = torch.abs(values - region_mean[image_region_index])
        region_deviation = torch.zeros(num_regions, device=device, dtype=torch.float32)
        region_deviation.index_add_(0, image_region_index, deviation)
        accumulated += region_deviation / image_counts
        valid_group_count += 1

    if valid_group_count == 0:
        return accumulated
    return accumulated / valid_group_count


def _iter_historical_texts(metadata: dict[str, Any]) -> Iterable[str]:
    for key in ("_selected_historical_texts", "historical_texts"):
        raw_value = metadata.get(key)
        if not isinstance(raw_value, list):
            continue
        for item in raw_value:
            text = str(item).strip()
            if text:
                yield text


def _parse_historical_spatial_intent(metadata: dict[str, Any]) -> dict[str, float | bool | None]:
    texts = list(_iter_historical_texts(metadata))
    if not texts:
        return {
            "lane_width_meters": None,
            "pedestrian_priority": False,
            "connectivity_priority": False,
            "heritage_priority": False,
            "greening_priority": False,
            "stone_paving_priority": False,
            "showcase_priority": False,
            "emergency_access_priority": False,
            "resident_service_priority": False,
            "streetscape_style_priority": False,
        }

    text_blob = " ".join(texts)
    width_matches = [
        (int(match.group(1)), int(match.group(2)))
        for match in METRIC_WIDTH_PATTERN.finditer(text_blob)
    ]
    width_values = [(left + right) / 2.0 for left, right in width_matches]
    lane_width_meters = (
        sum(width_values) / max(len(width_values), 1) if width_values else None
    )
    return {
        "lane_width_meters": lane_width_meters,
        "pedestrian_priority": any(keyword in text_blob for keyword in SLOW_TRAFFIC_KEYWORDS),
        "connectivity_priority": any(keyword in text_blob for keyword in CONNECTIVITY_KEYWORDS),
        "heritage_priority": any(keyword in text_blob for keyword in HERITAGE_KEYWORDS),
        "greening_priority": any(keyword in text_blob for keyword in GREENING_KEYWORDS),
        "stone_paving_priority": any(keyword in text_blob for keyword in STONE_PAVING_KEYWORDS),
        "showcase_priority": any(keyword in text_blob for keyword in SHOWCASE_KEYWORDS),
        "emergency_access_priority": any(
            keyword in text_blob for keyword in EMERGENCY_ACCESS_KEYWORDS
        ),
        "resident_service_priority": any(
            keyword in text_blob for keyword in RESIDENT_SERVICE_KEYWORDS
        ),
        "streetscape_style_priority": any(keyword in text_blob for keyword in STYLE_KEYWORDS),
    }


def _clip_profile(profile: dict[str, float]) -> dict[str, float]:
    clipped = {key: max(0.0, value) for key, value in profile.items()}
    total = sum(clipped.values())
    if total <= 0.0:
        return clipped
    return {key: value / total for key, value in clipped.items()}


def _adjust_profile_for_historical_text(
    *,
    base_profile: dict[str, float],
    metadata: dict[str, Any],
    image_type: str,
) -> dict[str, float]:
    adjusted = {key: float(value) for key, value in base_profile.items()}
    intent = _parse_historical_spatial_intent(metadata)
    if not any(
        value
        for key, value in intent.items()
        if key != "lane_width_meters"
    ) and intent.get("lane_width_meters") is None:
        return adjusted

    lane_width_meters = intent.get("lane_width_meters")
    if isinstance(lane_width_meters, (int, float)):
        if float(lane_width_meters) <= 6.0:
            adjusted["road_surface"] = 0.18 if image_type == "street" else 0.16
            adjusted["enclosure"] = 0.31 if image_type == "street" else 0.35
            adjusted["sky"] = 0.17 if image_type == "street" else adjusted.get("sky", 0.0)
        else:
            adjusted["road_surface"] = 0.24 if image_type == "street" else 0.22
            adjusted["enclosure"] = 0.24 if image_type == "street" else 0.30
            adjusted["sky"] = 0.22 if image_type == "street" else adjusted.get("sky", 0.0)

    if bool(intent.get("heritage_priority")):
        adjusted["enclosure"] = max(adjusted.get("enclosure", 0.0), 0.30 if image_type == "street" else 0.36)
        adjusted["vegetation"] = max(adjusted.get("vegetation", 0.0), 0.18 if image_type == "street" else 0.20)
        adjusted["mobility"] = min(adjusted.get("mobility", 0.0), 0.07 if image_type == "street" else 0.05)
    if bool(intent.get("connectivity_priority")):
        adjusted["road_surface"] = max(adjusted.get("road_surface", 0.0), 0.20 if image_type == "street" else 0.18)
        adjusted["enclosure"] = min(adjusted.get("enclosure", 0.0), 0.28 if image_type == "street" else 0.33)
    if bool(intent.get("pedestrian_priority")):
        adjusted["pedestrian"] = max(adjusted.get("pedestrian", 0.0), 0.07 if image_type == "street" else 0.02)
        adjusted["mobility"] = min(adjusted.get("mobility", 0.0), 0.05 if image_type == "street" else 0.04)
    if bool(intent.get("stone_paving_priority")):
        adjusted["road_surface"] = max(adjusted.get("road_surface", 0.0), 0.20 if image_type == "street" else 0.17)
        adjusted["mobility"] = min(adjusted.get("mobility", 0.0), 0.05 if image_type == "street" else 0.04)
    if bool(intent.get("greening_priority")):
        adjusted["vegetation"] = max(adjusted.get("vegetation", 0.0), 0.20 if image_type == "street" else 0.22)
        adjusted["sky"] = max(adjusted.get("sky", 0.0), 0.21 if image_type == "street" else adjusted.get("sky", 0.0))
    if bool(intent.get("showcase_priority")):
        adjusted["enclosure"] = max(adjusted.get("enclosure", 0.0), 0.32 if image_type == "street" else 0.37)
        adjusted["pedestrian"] = max(adjusted.get("pedestrian", 0.0), 0.08 if image_type == "street" else 0.02)
        adjusted["mobility"] = min(adjusted.get("mobility", 0.0), 0.05 if image_type == "street" else 0.04)
    if bool(intent.get("emergency_access_priority")):
        adjusted["road_surface"] = max(adjusted.get("road_surface", 0.0), 0.21 if image_type == "street" else 0.19)
        adjusted["mobility"] = min(adjusted.get("mobility", 0.0), 0.07 if image_type == "street" else 0.05)
    if bool(intent.get("resident_service_priority")):
        adjusted["road_surface"] = max(adjusted.get("road_surface", 0.0), 0.19 if image_type == "street" else 0.18)
        adjusted["pedestrian"] = max(adjusted.get("pedestrian", 0.0), 0.06 if image_type == "street" else 0.02)
        adjusted["mobility"] = min(adjusted.get("mobility", 0.0), 0.06 if image_type == "street" else 0.05)
    if bool(intent.get("streetscape_style_priority")):
        adjusted["enclosure"] = max(adjusted.get("enclosure", 0.0), 0.31 if image_type == "street" else 0.36)
        adjusted["vegetation"] = max(adjusted.get("vegetation", 0.0), 0.19 if image_type == "street" else 0.21)
        adjusted["mobility"] = min(adjusted.get("mobility", 0.0), 0.05 if image_type == "street" else 0.04)

    return _clip_profile(adjusted)


def build_adaptive_target_profiles(
    *,
    base_profile: dict[str, float],
    metadata: list[dict[str, Any]],
    device: torch.device,
    image_type: str,
) -> dict[str, torch.Tensor]:
    per_group_targets = {
        group_name: torch.zeros(len(metadata), device=device, dtype=torch.float32)
        for group_name in base_profile
    }
    for region_index, region_metadata in enumerate(metadata):
        adjusted_profile = _adjust_profile_for_historical_text(
            base_profile=base_profile,
            metadata=region_metadata,
            image_type=image_type,
        )
        for group_name, value in adjusted_profile.items():
            per_group_targets[group_name][region_index] = float(value)
    return per_group_targets


def build_historical_plan_targets(
    *,
    metadata: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    targets = {
        "width_road_surface_target": torch.full(
            (len(metadata),),
            torch.nan,
            device=device,
            dtype=torch.float32,
        ),
        "walkability_target": torch.full(
            (len(metadata),),
            torch.nan,
            device=device,
            dtype=torch.float32,
        ),
        "connectivity_target": torch.full(
            (len(metadata),),
            torch.nan,
            device=device,
            dtype=torch.float32,
        ),
        "heritage_target": torch.full(
            (len(metadata),),
            torch.nan,
            device=device,
            dtype=torch.float32,
        ),
        "satellite_width_target": torch.full(
            (len(metadata),),
            torch.nan,
            device=device,
            dtype=torch.float32,
        ),
        "satellite_connectivity_target": torch.full(
            (len(metadata),),
            torch.nan,
            device=device,
            dtype=torch.float32,
        ),
        "satellite_heritage_target": torch.full(
            (len(metadata),),
            torch.nan,
            device=device,
            dtype=torch.float32,
        ),
        "satellite_greening_target": torch.full(
            (len(metadata),),
            torch.nan,
            device=device,
            dtype=torch.float32,
        ),
        "service_access_target": torch.full(
            (len(metadata),),
            torch.nan,
            device=device,
            dtype=torch.float32,
        ),
        "emergency_access_target": torch.full(
            (len(metadata),),
            torch.nan,
            device=device,
            dtype=torch.float32,
        ),
        "showcase_target": torch.full(
            (len(metadata),),
            torch.nan,
            device=device,
            dtype=torch.float32,
        ),
        "streetscape_style_target": torch.full(
            (len(metadata),),
            torch.nan,
            device=device,
            dtype=torch.float32,
        ),
        "satellite_service_access_target": torch.full(
            (len(metadata),),
            torch.nan,
            device=device,
            dtype=torch.float32,
        ),
        "satellite_showcase_target": torch.full(
            (len(metadata),),
            torch.nan,
            device=device,
            dtype=torch.float32,
        ),
    }
    for region_index, region_metadata in enumerate(metadata):
        intent = _parse_historical_spatial_intent(region_metadata)
        lane_width_meters = intent.get("lane_width_meters")
        if isinstance(lane_width_meters, (int, float)):
            targets["width_road_surface_target"][region_index] = (
                0.18 if float(lane_width_meters) <= 6.0 else 0.24
            )
            targets["satellite_width_target"][region_index] = (
                0.16 if float(lane_width_meters) <= 6.0 else 0.21
            )
        if bool(intent.get("pedestrian_priority")) or bool(intent.get("stone_paving_priority")):
            targets["walkability_target"][region_index] = 0.62
        if bool(intent.get("connectivity_priority")):
            targets["connectivity_target"][region_index] = 0.75
            targets["satellite_connectivity_target"][region_index] = 0.72
        if bool(intent.get("heritage_priority")):
            targets["heritage_target"][region_index] = 0.60
            targets["satellite_heritage_target"][region_index] = 0.63
        if bool(intent.get("greening_priority")) or bool(intent.get("heritage_priority")):
            targets["satellite_greening_target"][region_index] = 0.22
        if bool(intent.get("resident_service_priority")) or bool(intent.get("connectivity_priority")):
            targets["service_access_target"][region_index] = 0.68
            targets["satellite_service_access_target"][region_index] = 0.70
        if bool(intent.get("emergency_access_priority")):
            targets["emergency_access_target"][region_index] = 0.74
            targets["satellite_service_access_target"][region_index] = 0.74
        if bool(intent.get("showcase_priority")):
            targets["showcase_target"][region_index] = 0.70
            targets["satellite_showcase_target"][region_index] = 0.66
        if (
            bool(intent.get("streetscape_style_priority"))
            or bool(intent.get("heritage_priority"))
            or bool(intent.get("stone_paving_priority"))
        ):
            targets["streetscape_style_target"][region_index] = 0.66
            targets["satellite_showcase_target"][region_index] = 0.68
    return targets


def _resolve_target_tensor(
    target_value: torch.Tensor | float | int,
    *,
    reference: torch.Tensor,
) -> torch.Tensor:
    if isinstance(target_value, torch.Tensor):
        return target_value.to(device=reference.device, dtype=reference.dtype)
    return torch.full_like(reference, float(target_value))


def _optional_target_tensor(
    targets: dict[str, torch.Tensor],
    key: str,
    *,
    reference: torch.Tensor,
) -> torch.Tensor:
    target = targets.get(key)
    if target is None:
        return torch.full_like(reference, torch.nan)
    return target.to(device=reference.device, dtype=reference.dtype)


def _add_profile_components(
    components: dict[str, dict[str, torch.Tensor | float]],
    *,
    prefix_names: dict[str, str],
    group_ratios: dict[str, torch.Tensor],
    target_profile: dict[str, torch.Tensor | float],
    weights: dict[str, float],
) -> None:
    for group_name, actual in group_ratios.items():
        target_tensor = _resolve_target_tensor(target_profile.get(group_name, 0.0), reference=actual)
        weight = float(weights.get(group_name, 1.0))
        component_name = prefix_names.get(group_name, group_name)
        abs_delta = torch.abs(actual - target_tensor)
        components[component_name] = {
            "actual": actual,
            "target": target_tensor,
            "abs_delta": abs_delta,
            "weighted_delta": abs_delta * weight,
            "weight": weight,
        }


def _cross_view_gap(
    street_group_ratios: dict[str, torch.Tensor],
    satellite_group_ratios: dict[str, torch.Tensor] | None,
    *,
    groups: tuple[str, ...],
) -> torch.Tensor | None:
    if satellite_group_ratios is None:
        return None
    deltas = []
    for group_name in groups:
        street_values = street_group_ratios.get(group_name)
        satellite_values = satellite_group_ratios.get(group_name)
        if street_values is None or satellite_values is None:
            continue
        deltas.append(torch.abs(street_values - satellite_values))
    if not deltas:
        return None
    return torch.stack(deltas, dim=0).mean(dim=0)


def _ordered_point_ids(
    point_directions: dict[str, dict[str, dict[str, float]]],
    point_coordinates: dict[str, tuple[float, float]],
) -> list[str]:
    coordinate_point_ids = [point_id for point_id in point_directions if point_id in point_coordinates]
    if len(coordinate_point_ids) >= 2:
        longitudes = [point_coordinates[point_id][0] for point_id in coordinate_point_ids]
        latitudes = [point_coordinates[point_id][1] for point_id in coordinate_point_ids]
        longitude_mean = sum(longitudes) / max(len(longitudes), 1)
        latitude_mean = sum(latitudes) / max(len(latitudes), 1)
        covariance_xx = sum((value - longitude_mean) ** 2 for value in longitudes)
        covariance_yy = sum((value - latitude_mean) ** 2 for value in latitudes)
        covariance_xy = sum(
            (longitude - longitude_mean) * (latitude - latitude_mean)
            for longitude, latitude in zip(longitudes, latitudes, strict=True)
        )
        trace = covariance_xx + covariance_yy
        determinant = covariance_xx * covariance_yy - covariance_xy * covariance_xy
        discriminant = max(trace * trace / 4.0 - determinant, 0.0) ** 0.5
        largest_eigenvalue = trace / 2.0 + discriminant
        if abs(covariance_xy) > 1e-8:
            axis_vector = (largest_eigenvalue - covariance_yy, covariance_xy)
        elif covariance_xx >= covariance_yy:
            axis_vector = (1.0, 0.0)
        else:
            axis_vector = (0.0, 1.0)
        axis_norm = max((axis_vector[0] ** 2 + axis_vector[1] ** 2) ** 0.5, 1e-6)
        ordered = sorted(
            coordinate_point_ids,
            key=lambda point_id: (
                (point_coordinates[point_id][0] - longitude_mean) * axis_vector[0]
                + (point_coordinates[point_id][1] - latitude_mean) * axis_vector[1]
            )
            / axis_norm,
        )
        trailing = sorted(set(point_directions) - set(ordered))
        return ordered + trailing
    return sorted(point_directions.keys())


def _coordinate_distance(
    left_coordinate: tuple[float, float],
    right_coordinate: tuple[float, float],
) -> float:
    return (
        (left_coordinate[0] - right_coordinate[0]) ** 2
        + (left_coordinate[1] - right_coordinate[1]) ** 2
    ) ** 0.5


def _turn_angle_score(
    left_coordinate: tuple[float, float],
    center_coordinate: tuple[float, float],
    right_coordinate: tuple[float, float],
) -> float:
    left_vector = (
        center_coordinate[0] - left_coordinate[0],
        center_coordinate[1] - left_coordinate[1],
    )
    right_vector = (
        right_coordinate[0] - center_coordinate[0],
        right_coordinate[1] - center_coordinate[1],
    )
    left_norm = max((left_vector[0] ** 2 + left_vector[1] ** 2) ** 0.5, 1e-6)
    right_norm = max((right_vector[0] ** 2 + right_vector[1] ** 2) ** 0.5, 1e-6)
    cosine = (
        left_vector[0] * right_vector[0] + left_vector[1] * right_vector[1]
    ) / (left_norm * right_norm)
    cosine = max(min(cosine, 1.0), -1.0)
    return float(torch.arccos(torch.tensor(cosine)).item() / torch.pi)


def _median_distance(values: list[float]) -> float | None:
    positives = sorted(value for value in values if value > 0.0)
    if not positives:
        return None
    middle = len(positives) // 2
    if len(positives) % 2 == 1:
        return positives[middle]
    return (positives[middle - 1] + positives[middle]) / 2.0


def _directional_geometry_gap(
    *,
    per_image_group_ratios: dict[str, torch.Tensor],
    image_region_index: torch.Tensor,
    image_point_ids: list[str] | None,
    image_view_directions: list[str] | None,
    image_longitudes: list[float | None] | None,
    image_latitudes: list[float | None] | None,
    num_regions: int,
) -> torch.Tensor:
    device = image_region_index.device
    result = torch.zeros(num_regions, device=device, dtype=torch.float32)
    if (
        image_point_ids is None
        or image_view_directions is None
        or len(image_point_ids) != len(image_view_directions)
        or len(image_point_ids) != int(image_region_index.shape[0])
    ):
        return result
    if (
        image_longitudes is None
        or image_latitudes is None
        or len(image_longitudes) != len(image_point_ids)
        or len(image_latitudes) != len(image_point_ids)
    ):
        image_longitudes = [None] * len(image_point_ids)
        image_latitudes = [None] * len(image_point_ids)

    geometry_groups = ("road_surface", "enclosure", "vegetation", "sky", "mobility")
    regions: dict[int, dict[str, dict[str, dict[str, float]]]] = {}
    region_coordinates: dict[int, dict[str, tuple[float, float]]] = {}
    for image_index, region_index_value in enumerate(image_region_index.tolist()):
        point_id = image_point_ids[image_index]
        direction = image_view_directions[image_index].strip().lower()
        if not point_id or direction in {"", "unknown", "satellite"}:
            continue
        point_bucket = regions.setdefault(region_index_value, {}).setdefault(point_id, {})
        point_bucket[direction] = {
            group_name: float(per_image_group_ratios[group_name][image_index].detach().cpu().item())
            for group_name in geometry_groups
            if group_name in per_image_group_ratios
        }
        longitude = image_longitudes[image_index]
        latitude = image_latitudes[image_index]
        if longitude is not None and latitude is not None:
            region_coordinates.setdefault(region_index_value, {})[point_id] = (
                float(longitude),
                float(latitude),
            )

    opposite_pairs = (("north", "south"), ("east", "west"))
    for region_index_value, region_points in regions.items():
        pair_scores: list[float] = []
        sorted_point_ids = _ordered_point_ids(
            region_points,
            region_coordinates.get(region_index_value, {}),
        )
        adjacent_coordinates: list[tuple[tuple[float, float], tuple[float, float]]] = []
        for point_id in sorted_point_ids:
            direction_map = region_points[point_id]
            for left, right in opposite_pairs:
                if left not in direction_map or right not in direction_map:
                    continue
                deltas = [
                    abs(direction_map[left].get(group_name, 0.0) - direction_map[right].get(group_name, 0.0))
                    for group_name in ("enclosure", "sky", "vegetation", "road_surface")
                ]
                pair_scores.append(sum(deltas) / max(len(deltas), 1))

        for left_point, right_point in zip(sorted_point_ids, sorted_point_ids[1:], strict=False):
            left_views = region_points[left_point]
            right_views = region_points[right_point]
            common_directions = sorted(set(left_views) & set(right_views))
            left_coordinate = region_coordinates.get(region_index_value, {}).get(left_point)
            right_coordinate = region_coordinates.get(region_index_value, {}).get(right_point)
            if left_coordinate is not None and right_coordinate is not None:
                adjacent_coordinates.append((left_coordinate, right_coordinate))
        median_distance = _median_distance(
            [
                _coordinate_distance(left_coordinate, right_coordinate)
                for left_coordinate, right_coordinate in adjacent_coordinates
            ]
        )

        for left_point, right_point in zip(sorted_point_ids, sorted_point_ids[1:], strict=False):
            left_views = region_points[left_point]
            right_views = region_points[right_point]
            common_directions = sorted(set(left_views) & set(right_views))
            left_coordinate = region_coordinates.get(region_index_value, {}).get(left_point)
            right_coordinate = region_coordinates.get(region_index_value, {}).get(right_point)
            distance_weight = 1.0
            if (
                median_distance is not None
                and left_coordinate is not None
                and right_coordinate is not None
            ):
                distance_ratio = _coordinate_distance(left_coordinate, right_coordinate) / max(
                    median_distance,
                    1e-6,
                )
                distance_weight = 1.0 / min(max(distance_ratio, 0.75), 1.75)
            for direction in common_directions:
                deltas = [
                    abs(left_views[direction].get(group_name, 0.0) - right_views[direction].get(group_name, 0.0))
                    for group_name in ("road_surface", "enclosure", "mobility")
                ]
                pair_scores.append(sum(deltas) / max(len(deltas), 1) * distance_weight)

        if median_distance is not None:
            spacing_penalty = sum(
                abs(_coordinate_distance(left_coordinate, right_coordinate) - median_distance)
                / max(median_distance, 1e-6)
                for left_coordinate, right_coordinate in adjacent_coordinates
            ) / max(len(adjacent_coordinates), 1)
            pair_scores.append(min(spacing_penalty, 1.0))

        ordered_coordinates = [
            region_coordinates[region_index_value][point_id]
            for point_id in sorted_point_ids
            if point_id in region_coordinates.get(region_index_value, {})
        ]
        if len(ordered_coordinates) >= 3:
            turn_scores = [
                _turn_angle_score(left_coordinate, center_coordinate, right_coordinate)
                for left_coordinate, center_coordinate, right_coordinate in zip(
                    ordered_coordinates,
                    ordered_coordinates[1:],
                    ordered_coordinates[2:],
                    strict=False,
                )
            ]
            if turn_scores:
                pair_scores.append(sum(turn_scores) / max(len(turn_scores), 1))

        if pair_scores:
            result[region_index_value] = float(sum(pair_scores) / len(pair_scores))
    return result.clamp(min=0.0, max=1.0)


def compute_ifi(
    *,
    street_group_ratios: dict[str, torch.Tensor],
    street_target_profile: dict[str, torch.Tensor | float],
    street_weights: dict[str, float],
    satellite_group_ratios: dict[str, torch.Tensor] | None = None,
    satellite_target_profile: dict[str, torch.Tensor | float] | None = None,
    satellite_weights: dict[str, float] | None = None,
    cross_view_weights: dict[str, float] | None = None,
    per_image_street_group_ratios: dict[str, torch.Tensor] | None = None,
    street_image_region_index: torch.Tensor | None = None,
    street_image_point_ids: list[str] | None = None,
    street_image_view_directions: list[str] | None = None,
    street_image_longitudes: list[float | None] | None = None,
    street_image_latitudes: list[float | None] | None = None,
    per_image_satellite_group_ratios: dict[str, torch.Tensor] | None = None,
    satellite_image_region_index: torch.Tensor | None = None,
    historical_plan_targets: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    components = compute_ifi_components(
        street_group_ratios=street_group_ratios,
        street_target_profile=street_target_profile,
        street_weights=street_weights,
        satellite_group_ratios=satellite_group_ratios,
        satellite_target_profile=satellite_target_profile,
        satellite_weights=satellite_weights,
        cross_view_weights=cross_view_weights,
        per_image_street_group_ratios=per_image_street_group_ratios,
        street_image_region_index=street_image_region_index,
        street_image_point_ids=street_image_point_ids,
        street_image_view_directions=street_image_view_directions,
        street_image_longitudes=street_image_longitudes,
        street_image_latitudes=street_image_latitudes,
        per_image_satellite_group_ratios=per_image_satellite_group_ratios,
        satellite_image_region_index=satellite_image_region_index,
        historical_plan_targets=historical_plan_targets,
    )
    numerator = None
    denominator = 0.0
    for component in components.values():
        weighted_delta = component["weighted_delta"]
        numerator = weighted_delta if numerator is None else numerator + weighted_delta
        denominator += float(component["weight"])
    if numerator is None:
        raise ValueError("No spatial morphology components were provided for IFI computation.")
    return numerator / max(denominator, 1e-6)


def compute_ifi_components(
    *,
    street_group_ratios: dict[str, torch.Tensor],
    street_target_profile: dict[str, torch.Tensor | float],
    street_weights: dict[str, float],
    satellite_group_ratios: dict[str, torch.Tensor] | None = None,
    satellite_target_profile: dict[str, torch.Tensor | float] | None = None,
    satellite_weights: dict[str, float] | None = None,
    cross_view_weights: dict[str, float] | None = None,
    per_image_street_group_ratios: dict[str, torch.Tensor] | None = None,
    street_image_region_index: torch.Tensor | None = None,
    street_image_point_ids: list[str] | None = None,
    street_image_view_directions: list[str] | None = None,
    street_image_longitudes: list[float | None] | None = None,
    street_image_latitudes: list[float | None] | None = None,
    per_image_satellite_group_ratios: dict[str, torch.Tensor] | None = None,
    satellite_image_region_index: torch.Tensor | None = None,
    historical_plan_targets: dict[str, torch.Tensor] | None = None,
) -> dict[str, dict[str, torch.Tensor | float]]:
    components: dict[str, dict[str, torch.Tensor | float]] = {}
    _add_profile_components(
        components,
        prefix_names=STREET_COMPONENT_NAMES,
        group_ratios=street_group_ratios,
        target_profile=street_target_profile,
        weights=street_weights,
    )

    if satellite_group_ratios is not None:
        _add_profile_components(
            components,
            prefix_names=SATELLITE_COMPONENT_NAMES,
            group_ratios=satellite_group_ratios,
            target_profile=satellite_target_profile or {},
            weights=satellite_weights or {},
        )

    resolved_cross_view_weights = cross_view_weights or {}
    if per_image_street_group_ratios is not None and street_image_region_index is not None:
        variability = _region_variability(
            per_image_street_group_ratios,
            image_region_index=street_image_region_index,
            num_regions=next(iter(street_group_ratios.values())).shape[0],
            groups=("road_surface", "enclosure", "vegetation", "mobility"),
        )
        continuity_weight = float(
            resolved_cross_view_weights.get(
                "street_boundary_continuity",
                _component_weight(
                    street_weights,
                    "road_surface",
                    "enclosure",
                    "vegetation",
                    "mobility",
                    default=1.1,
                ),
            )
        )
        components["street_boundary_continuity"] = {
            "actual": variability,
            "target": torch.zeros_like(variability),
            "abs_delta": variability,
            "weighted_delta": variability * continuity_weight,
            "weight": continuity_weight,
        }

        geometry_gap = _directional_geometry_gap(
            per_image_group_ratios=per_image_street_group_ratios,
            image_region_index=street_image_region_index,
            image_point_ids=street_image_point_ids,
            image_view_directions=street_image_view_directions,
            image_longitudes=street_image_longitudes,
            image_latitudes=street_image_latitudes,
            num_regions=next(iter(street_group_ratios.values())).shape[0],
        )
        geometry_weight = float(
            resolved_cross_view_weights.get(
                "street_view_geometry",
                _component_weight(
                    street_weights,
                    "road_surface",
                    "enclosure",
                    "sky",
                    default=1.0,
                ),
            )
        )
        components["street_view_geometry"] = {
            "actual": geometry_gap,
            "target": torch.zeros_like(geometry_gap),
            "abs_delta": geometry_gap,
            "weighted_delta": geometry_gap * geometry_weight,
            "weight": geometry_weight,
        }

        if historical_plan_targets is not None:
            width_target = _optional_target_tensor(
                historical_plan_targets,
                "width_road_surface_target",
                reference=street_group_ratios["road_surface"],
            )
            width_valid = ~torch.isnan(width_target)
            width_target_resolved = torch.where(width_valid, width_target, street_group_ratios["road_surface"])
            width_delta = torch.where(
                width_valid,
                torch.abs(street_group_ratios["road_surface"] - width_target_resolved),
                torch.zeros_like(width_target_resolved),
            )
            components["historical_plan_width"] = {
                "actual": street_group_ratios["road_surface"],
                "target": width_target_resolved,
                "abs_delta": width_delta,
                "weighted_delta": width_delta * 1.15,
                "weight": 1.15,
            }

            walkability_actual = (
                street_group_ratios.get("pedestrian", torch.zeros_like(geometry_gap))
                + street_group_ratios["road_surface"] * 0.35
                + street_group_ratios["enclosure"] * 0.15
                - street_group_ratios["mobility"] * 0.50
            ).clamp(min=0.0, max=1.0)
            walkability_target = _optional_target_tensor(
                historical_plan_targets,
                "walkability_target",
                reference=walkability_actual,
            )
            walkability_valid = ~torch.isnan(walkability_target)
            walkability_target_resolved = torch.where(
                walkability_valid,
                walkability_target,
                walkability_actual,
            )
            walkability_delta = torch.where(
                walkability_valid,
                torch.abs(walkability_actual - walkability_target_resolved),
                torch.zeros_like(walkability_actual),
            )
            components["historical_plan_walkability"] = {
                "actual": walkability_actual,
                "target": walkability_target_resolved,
                "abs_delta": walkability_delta,
                "weighted_delta": walkability_delta * 1.10,
                "weight": 1.10,
            }

            continuity_actual = (1.0 - geometry_gap).clamp(min=0.0, max=1.0)
            connectivity_target = _optional_target_tensor(
                historical_plan_targets,
                "connectivity_target",
                reference=continuity_actual,
            )
            connectivity_valid = ~torch.isnan(connectivity_target)
            connectivity_target_resolved = torch.where(
                connectivity_valid,
                connectivity_target,
                continuity_actual,
            )
            connectivity_delta = torch.where(
                connectivity_valid,
                torch.abs(continuity_actual - connectivity_target_resolved),
                torch.zeros_like(continuity_actual),
            )
            components["historical_plan_connectivity"] = {
                "actual": continuity_actual,
                "target": connectivity_target_resolved,
                "abs_delta": connectivity_delta,
                "weighted_delta": connectivity_delta * 1.05,
                "weight": 1.05,
            }

            heritage_actual = (
                street_group_ratios["enclosure"] * 0.45
                + street_group_ratios["vegetation"] * 0.25
                + street_group_ratios.get("pedestrian", torch.zeros_like(geometry_gap)) * 0.10
                - street_group_ratios["mobility"] * 0.20
            ).clamp(min=0.0, max=1.0)
            heritage_target = _optional_target_tensor(
                historical_plan_targets,
                "heritage_target",
                reference=heritage_actual,
            )
            heritage_valid = ~torch.isnan(heritage_target)
            heritage_target_resolved = torch.where(
                heritage_valid,
                heritage_target,
                heritage_actual,
            )
            heritage_delta = torch.where(
                heritage_valid,
                torch.abs(heritage_actual - heritage_target_resolved),
                torch.zeros_like(heritage_actual),
            )
            components["historical_plan_heritage"] = {
                "actual": heritage_actual,
                "target": heritage_target_resolved,
                "abs_delta": heritage_delta,
                "weighted_delta": heritage_delta * 1.00,
                "weight": 1.00,
            }

            service_access_actual = (
                street_group_ratios["road_surface"] * 0.30
                + street_group_ratios.get("pedestrian", torch.zeros_like(geometry_gap)) * 0.20
                + (1.0 - street_group_ratios["mobility"]).clamp(min=0.0, max=1.0) * 0.25
                + continuity_actual * 0.15
                + street_group_ratios["enclosure"] * 0.10
            ).clamp(min=0.0, max=1.0)
            service_access_target = _optional_target_tensor(
                historical_plan_targets,
                "service_access_target",
                reference=service_access_actual,
            )
            service_access_valid = ~torch.isnan(service_access_target)
            service_access_target_resolved = torch.where(
                service_access_valid,
                service_access_target,
                service_access_actual,
            )
            service_access_delta = torch.where(
                service_access_valid,
                torch.abs(service_access_actual - service_access_target_resolved),
                torch.zeros_like(service_access_actual),
            )
            components["historical_plan_service_access"] = {
                "actual": service_access_actual,
                "target": service_access_target_resolved,
                "abs_delta": service_access_delta,
                "weighted_delta": service_access_delta * 0.95,
                "weight": 0.95,
            }

            emergency_access_actual = (
                street_group_ratios["road_surface"] * 0.35
                + continuity_actual * 0.35
                + (1.0 - street_group_ratios["mobility"]).clamp(min=0.0, max=1.0) * 0.20
                + street_group_ratios["sky"] * 0.10
            ).clamp(min=0.0, max=1.0)
            emergency_access_target = _optional_target_tensor(
                historical_plan_targets,
                "emergency_access_target",
                reference=emergency_access_actual,
            )
            emergency_access_valid = ~torch.isnan(emergency_access_target)
            emergency_access_target_resolved = torch.where(
                emergency_access_valid,
                emergency_access_target,
                emergency_access_actual,
            )
            emergency_access_delta = torch.where(
                emergency_access_valid,
                torch.abs(emergency_access_actual - emergency_access_target_resolved),
                torch.zeros_like(emergency_access_actual),
            )
            components["historical_plan_emergency_access"] = {
                "actual": emergency_access_actual,
                "target": emergency_access_target_resolved,
                "abs_delta": emergency_access_delta,
                "weighted_delta": emergency_access_delta * 0.95,
                "weight": 0.95,
            }

            showcase_actual = (
                heritage_actual * 0.50
                + walkability_actual * 0.30
                + street_group_ratios["enclosure"] * 0.10
                + street_group_ratios["vegetation"] * 0.10
            ).clamp(min=0.0, max=1.0)
            showcase_target = _optional_target_tensor(
                historical_plan_targets,
                "showcase_target",
                reference=showcase_actual,
            )
            showcase_valid = ~torch.isnan(showcase_target)
            showcase_target_resolved = torch.where(
                showcase_valid,
                showcase_target,
                showcase_actual,
            )
            showcase_delta = torch.where(
                showcase_valid,
                torch.abs(showcase_actual - showcase_target_resolved),
                torch.zeros_like(showcase_actual),
            )
            components["historical_plan_showcase"] = {
                "actual": showcase_actual,
                "target": showcase_target_resolved,
                "abs_delta": showcase_delta,
                "weighted_delta": showcase_delta * 0.90,
                "weight": 0.90,
            }

            streetscape_style_actual = (
                heritage_actual * 0.45
                + street_group_ratios["enclosure"] * 0.20
                + street_group_ratios["vegetation"] * 0.20
                + (1.0 - street_group_ratios["mobility"]).clamp(min=0.0, max=1.0) * 0.15
            ).clamp(min=0.0, max=1.0)
            streetscape_style_target = _optional_target_tensor(
                historical_plan_targets,
                "streetscape_style_target",
                reference=streetscape_style_actual,
            )
            streetscape_style_valid = ~torch.isnan(streetscape_style_target)
            streetscape_style_target_resolved = torch.where(
                streetscape_style_valid,
                streetscape_style_target,
                streetscape_style_actual,
            )
            streetscape_style_delta = torch.where(
                streetscape_style_valid,
                torch.abs(streetscape_style_actual - streetscape_style_target_resolved),
                torch.zeros_like(streetscape_style_actual),
            )
            components["historical_plan_streetscape_style"] = {
                "actual": streetscape_style_actual,
                "target": streetscape_style_target_resolved,
                "abs_delta": streetscape_style_delta,
                "weighted_delta": streetscape_style_delta * 0.90,
                "weight": 0.90,
            }

    satellite_variability = None
    if (
        satellite_group_ratios is not None
        and per_image_satellite_group_ratios is not None
        and satellite_image_region_index is not None
    ):
        variability = _region_variability(
            per_image_satellite_group_ratios,
            image_region_index=satellite_image_region_index,
            num_regions=next(iter(satellite_group_ratios.values())).shape[0],
            groups=("road_surface", "enclosure", "vegetation"),
        )
        satellite_variability = variability
        continuity_weight = float(
            resolved_cross_view_weights.get(
                "satellite_boundary_continuity",
                _component_weight(
                    satellite_weights or {},
                    "road_surface",
                    "enclosure",
                    "vegetation",
                    default=0.8,
                ),
            )
        )
        components["satellite_boundary_continuity"] = {
            "actual": variability,
            "target": torch.zeros_like(variability),
            "abs_delta": variability,
            "weighted_delta": variability * continuity_weight,
            "weight": continuity_weight,
        }

    if satellite_group_ratios is not None and historical_plan_targets is not None:
        if satellite_variability is None:
            satellite_variability = torch.zeros_like(satellite_group_ratios["road_surface"])
        variability = satellite_variability
        if historical_plan_targets is not None:
            satellite_width_target = _optional_target_tensor(
                historical_plan_targets,
                "satellite_width_target",
                reference=satellite_group_ratios["road_surface"],
            )
            satellite_width_valid = ~torch.isnan(satellite_width_target)
            satellite_width_target_resolved = torch.where(
                satellite_width_valid,
                satellite_width_target,
                satellite_group_ratios["road_surface"],
            )
            satellite_width_delta = torch.where(
                satellite_width_valid,
                torch.abs(
                    satellite_group_ratios["road_surface"] - satellite_width_target_resolved
                ),
                torch.zeros_like(satellite_width_target_resolved),
            )
            components["historical_plan_satellite_width"] = {
                "actual": satellite_group_ratios["road_surface"],
                "target": satellite_width_target_resolved,
                "abs_delta": satellite_width_delta,
                "weighted_delta": satellite_width_delta * 1.05,
                "weight": 1.05,
            }

            satellite_connectivity_actual = (
                satellite_group_ratios["road_surface"] * 0.45
                + (1.0 - variability).clamp(min=0.0, max=1.0) * 0.45
                + (1.0 - satellite_group_ratios["mobility"]).clamp(min=0.0, max=1.0) * 0.10
            ).clamp(min=0.0, max=1.0)
            satellite_connectivity_target = _optional_target_tensor(
                historical_plan_targets,
                "satellite_connectivity_target",
                reference=satellite_connectivity_actual,
            )
            satellite_connectivity_valid = ~torch.isnan(satellite_connectivity_target)
            satellite_connectivity_target_resolved = torch.where(
                satellite_connectivity_valid,
                satellite_connectivity_target,
                satellite_connectivity_actual,
            )
            satellite_connectivity_delta = torch.where(
                satellite_connectivity_valid,
                torch.abs(
                    satellite_connectivity_actual - satellite_connectivity_target_resolved
                ),
                torch.zeros_like(satellite_connectivity_actual),
            )
            components["historical_plan_satellite_connectivity"] = {
                "actual": satellite_connectivity_actual,
                "target": satellite_connectivity_target_resolved,
                "abs_delta": satellite_connectivity_delta,
                "weighted_delta": satellite_connectivity_delta * 1.00,
                "weight": 1.00,
            }

            satellite_heritage_actual = (
                satellite_group_ratios["enclosure"] * 0.50
                + satellite_group_ratios["vegetation"] * 0.30
                + (1.0 - satellite_group_ratios["mobility"]).clamp(min=0.0, max=1.0) * 0.20
            ).clamp(min=0.0, max=1.0)
            satellite_heritage_target = _optional_target_tensor(
                historical_plan_targets,
                "satellite_heritage_target",
                reference=satellite_heritage_actual,
            )
            satellite_heritage_valid = ~torch.isnan(satellite_heritage_target)
            satellite_heritage_target_resolved = torch.where(
                satellite_heritage_valid,
                satellite_heritage_target,
                satellite_heritage_actual,
            )
            satellite_heritage_delta = torch.where(
                satellite_heritage_valid,
                torch.abs(satellite_heritage_actual - satellite_heritage_target_resolved),
                torch.zeros_like(satellite_heritage_actual),
            )
            components["historical_plan_satellite_heritage"] = {
                "actual": satellite_heritage_actual,
                "target": satellite_heritage_target_resolved,
                "abs_delta": satellite_heritage_delta,
                "weighted_delta": satellite_heritage_delta * 0.95,
                "weight": 0.95,
            }

            satellite_greening_target = _optional_target_tensor(
                historical_plan_targets,
                "satellite_greening_target",
                reference=satellite_group_ratios["vegetation"],
            )
            satellite_greening_valid = ~torch.isnan(satellite_greening_target)
            satellite_greening_target_resolved = torch.where(
                satellite_greening_valid,
                satellite_greening_target,
                satellite_group_ratios["vegetation"],
            )
            satellite_greening_delta = torch.where(
                satellite_greening_valid,
                torch.abs(
                    satellite_group_ratios["vegetation"] - satellite_greening_target_resolved
                ),
                torch.zeros_like(satellite_greening_target_resolved),
            )
            components["historical_plan_satellite_greening"] = {
                "actual": satellite_group_ratios["vegetation"],
                "target": satellite_greening_target_resolved,
                "abs_delta": satellite_greening_delta,
                "weighted_delta": satellite_greening_delta * 0.90,
                "weight": 0.90,
            }

            satellite_service_access_actual = (
                satellite_group_ratios["road_surface"] * 0.35
                + satellite_connectivity_actual * 0.45
                + (1.0 - satellite_group_ratios["mobility"]).clamp(min=0.0, max=1.0) * 0.20
            ).clamp(min=0.0, max=1.0)
            satellite_service_access_target = _optional_target_tensor(
                historical_plan_targets,
                "satellite_service_access_target",
                reference=satellite_service_access_actual,
            )
            satellite_service_access_valid = ~torch.isnan(satellite_service_access_target)
            satellite_service_access_target_resolved = torch.where(
                satellite_service_access_valid,
                satellite_service_access_target,
                satellite_service_access_actual,
            )
            satellite_service_access_delta = torch.where(
                satellite_service_access_valid,
                torch.abs(
                    satellite_service_access_actual
                    - satellite_service_access_target_resolved
                ),
                torch.zeros_like(satellite_service_access_actual),
            )
            components["historical_plan_satellite_service_access"] = {
                "actual": satellite_service_access_actual,
                "target": satellite_service_access_target_resolved,
                "abs_delta": satellite_service_access_delta,
                "weighted_delta": satellite_service_access_delta * 0.90,
                "weight": 0.90,
            }

            satellite_showcase_actual = (
                satellite_heritage_actual * 0.70
                + satellite_group_ratios["vegetation"] * 0.20
                + (1.0 - satellite_group_ratios["mobility"]).clamp(min=0.0, max=1.0) * 0.10
            ).clamp(min=0.0, max=1.0)
            satellite_showcase_target = _optional_target_tensor(
                historical_plan_targets,
                "satellite_showcase_target",
                reference=satellite_showcase_actual,
            )
            satellite_showcase_valid = ~torch.isnan(satellite_showcase_target)
            satellite_showcase_target_resolved = torch.where(
                satellite_showcase_valid,
                satellite_showcase_target,
                satellite_showcase_actual,
            )
            satellite_showcase_delta = torch.where(
                satellite_showcase_valid,
                torch.abs(satellite_showcase_actual - satellite_showcase_target_resolved),
                torch.zeros_like(satellite_showcase_actual),
            )
            components["historical_plan_satellite_showcase"] = {
                "actual": satellite_showcase_actual,
                "target": satellite_showcase_target_resolved,
                "abs_delta": satellite_showcase_delta,
                "weighted_delta": satellite_showcase_delta * 0.85,
                "weight": 0.85,
            }

    view_gap = _cross_view_gap(
        street_group_ratios,
        satellite_group_ratios,
        groups=("road_surface", "enclosure", "vegetation", "mobility"),
    )
    if view_gap is not None:
        consistency_weight = float(
            resolved_cross_view_weights.get("street_satellite_consistency", 1.0)
        )
        components["street_satellite_consistency"] = {
            "actual": view_gap,
            "target": torch.zeros_like(view_gap),
            "abs_delta": view_gap,
            "weighted_delta": view_gap * consistency_weight,
            "weight": consistency_weight,
        }

    return components


def compute_spatial_proxy_score(
    *,
    street_group_ratios: dict[str, torch.Tensor],
    street_target_profile: dict[str, torch.Tensor | float],
    street_weights: dict[str, float],
    satellite_group_ratios: dict[str, torch.Tensor] | None = None,
    satellite_target_profile: dict[str, torch.Tensor | float] | None = None,
    satellite_weights: dict[str, float] | None = None,
    cross_view_weights: dict[str, float] | None = None,
    per_image_street_group_ratios: dict[str, torch.Tensor] | None = None,
    street_image_region_index: torch.Tensor | None = None,
    per_image_satellite_group_ratios: dict[str, torch.Tensor] | None = None,
    satellite_image_region_index: torch.Tensor | None = None,
    historical_plan_targets: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    components: dict[str, dict[str, torch.Tensor | float]] = {}
    _add_profile_components(
        components,
        prefix_names=STREET_COMPONENT_NAMES,
        group_ratios=street_group_ratios,
        target_profile=street_target_profile,
        weights=street_weights,
    )
    if satellite_group_ratios is not None:
        _add_profile_components(
            components,
            prefix_names=SATELLITE_COMPONENT_NAMES,
            group_ratios=satellite_group_ratios,
            target_profile=satellite_target_profile or {},
            weights=satellite_weights or {},
        )

    resolved_cross_view_weights = cross_view_weights or {}
    street_variability = None
    if per_image_street_group_ratios is not None and street_image_region_index is not None:
        street_variability = _region_variability(
            per_image_street_group_ratios,
            image_region_index=street_image_region_index,
            num_regions=next(iter(street_group_ratios.values())).shape[0],
            groups=("road_surface", "enclosure", "vegetation", "mobility"),
        )
        continuity_weight = float(
            resolved_cross_view_weights.get(
                "street_boundary_continuity",
                _component_weight(
                    street_weights,
                    "road_surface",
                    "enclosure",
                    "vegetation",
                    "mobility",
                    default=1.1,
                ),
            )
        )
        components["proxy_street_boundary_continuity"] = {
            "actual": street_variability,
            "target": torch.zeros_like(street_variability),
            "abs_delta": street_variability,
            "weighted_delta": street_variability * continuity_weight,
            "weight": continuity_weight,
        }

    satellite_variability = None
    if (
        satellite_group_ratios is not None
        and per_image_satellite_group_ratios is not None
        and satellite_image_region_index is not None
    ):
        satellite_variability = _region_variability(
            per_image_satellite_group_ratios,
            image_region_index=satellite_image_region_index,
            num_regions=next(iter(satellite_group_ratios.values())).shape[0],
            groups=("road_surface", "enclosure", "vegetation"),
        )
        continuity_weight = float(
            resolved_cross_view_weights.get(
                "satellite_boundary_continuity",
                _component_weight(
                    satellite_weights or {},
                    "road_surface",
                    "enclosure",
                    "vegetation",
                    default=0.8,
                ),
            )
        )
        components["proxy_satellite_boundary_continuity"] = {
            "actual": satellite_variability,
            "target": torch.zeros_like(satellite_variability),
            "abs_delta": satellite_variability,
            "weighted_delta": satellite_variability * continuity_weight,
            "weight": continuity_weight,
        }

    if satellite_group_ratios is not None:
        consistency = _cross_view_gap(
            street_group_ratios,
            satellite_group_ratios,
            groups=("road_surface", "enclosure", "vegetation", "mobility"),
        )
        if consistency is not None:
            consistency_weight = float(
                resolved_cross_view_weights.get("street_satellite_consistency", 1.0)
            )
            components["proxy_street_satellite_consistency"] = {
                "actual": consistency,
                "target": torch.zeros_like(consistency),
                "abs_delta": consistency,
                "weighted_delta": consistency * consistency_weight,
                "weight": consistency_weight,
            }

    if historical_plan_targets is not None:
        reference = street_group_ratios["road_surface"]
        continuity_actual = (
            1.0 - street_variability
            if street_variability is not None
            else (
                street_group_ratios["road_surface"] * 0.45
                + (1.0 - street_group_ratios["mobility"]).clamp(min=0.0, max=1.0) * 0.55
            ).clamp(min=0.0, max=1.0)
        )
        walkability_actual = (
            street_group_ratios.get("pedestrian", torch.zeros_like(reference)) * 0.25
            + street_group_ratios["road_surface"] * 0.35
            + street_group_ratios["enclosure"] * 0.15
            - street_group_ratios["mobility"] * 0.50
        ).clamp(min=0.0, max=1.0)
        heritage_actual = (
            street_group_ratios["enclosure"] * 0.45
            + street_group_ratios["vegetation"] * 0.25
            + street_group_ratios.get("pedestrian", torch.zeros_like(reference)) * 0.10
            - street_group_ratios["mobility"] * 0.20
        ).clamp(min=0.0, max=1.0)
        service_access_actual = (
            street_group_ratios["road_surface"] * 0.30
            + street_group_ratios.get("pedestrian", torch.zeros_like(reference)) * 0.20
            + (1.0 - street_group_ratios["mobility"]).clamp(min=0.0, max=1.0) * 0.25
            + continuity_actual * 0.15
            + street_group_ratios["enclosure"] * 0.10
        ).clamp(min=0.0, max=1.0)
        showcase_actual = (
            heritage_actual * 0.50
            + walkability_actual * 0.30
            + street_group_ratios["enclosure"] * 0.10
            + street_group_ratios["vegetation"] * 0.10
        ).clamp(min=0.0, max=1.0)
        streetscape_style_actual = (
            heritage_actual * 0.45
            + street_group_ratios["enclosure"] * 0.20
            + street_group_ratios["vegetation"] * 0.20
            + (1.0 - street_group_ratios["mobility"]).clamp(min=0.0, max=1.0) * 0.15
        ).clamp(min=0.0, max=1.0)
        plan_pairs = [
            ("proxy_plan_width", street_group_ratios["road_surface"], "width_road_surface_target", 1.15),
            ("proxy_plan_walkability", walkability_actual, "walkability_target", 1.10),
            ("proxy_plan_connectivity", continuity_actual, "connectivity_target", 1.05),
            ("proxy_plan_heritage", heritage_actual, "heritage_target", 1.00),
            ("proxy_plan_service_access", service_access_actual, "service_access_target", 0.95),
            ("proxy_plan_emergency_access", continuity_actual, "emergency_access_target", 0.95),
            ("proxy_plan_showcase", showcase_actual, "showcase_target", 0.90),
            ("proxy_plan_streetscape_style", streetscape_style_actual, "streetscape_style_target", 0.90),
        ]
        for component_name, actual, target_name, weight in plan_pairs:
            target = _optional_target_tensor(
                historical_plan_targets,
                target_name,
                reference=actual,
            )
            valid = ~torch.isnan(target)
            target_resolved = torch.where(valid, target, actual)
            delta = torch.where(
                valid,
                torch.abs(actual - target_resolved),
                torch.zeros_like(actual),
            )
            components[component_name] = {
                "actual": actual,
                "target": target_resolved,
                "abs_delta": delta,
                "weighted_delta": delta * weight,
                "weight": weight,
            }

        if satellite_group_ratios is not None:
            satellite_continuity = (
                1.0 - satellite_variability
                if satellite_variability is not None
                else (
                    satellite_group_ratios["road_surface"] * 0.45
                    + (1.0 - satellite_group_ratios["mobility"]).clamp(min=0.0, max=1.0) * 0.55
                ).clamp(min=0.0, max=1.0)
            )
            satellite_connectivity_actual = (
                satellite_group_ratios["road_surface"] * 0.45
                + satellite_continuity * 0.45
                + (1.0 - satellite_group_ratios["mobility"]).clamp(min=0.0, max=1.0) * 0.10
            ).clamp(min=0.0, max=1.0)
            satellite_heritage_actual = (
                satellite_group_ratios["enclosure"] * 0.50
                + satellite_group_ratios["vegetation"] * 0.30
                + (1.0 - satellite_group_ratios["mobility"]).clamp(min=0.0, max=1.0) * 0.20
            ).clamp(min=0.0, max=1.0)
            satellite_service_access_actual = (
                satellite_group_ratios["road_surface"] * 0.35
                + satellite_connectivity_actual * 0.45
                + (1.0 - satellite_group_ratios["mobility"]).clamp(min=0.0, max=1.0) * 0.20
            ).clamp(min=0.0, max=1.0)
            satellite_showcase_actual = (
                satellite_heritage_actual * 0.70
                + satellite_group_ratios["vegetation"] * 0.20
                + (1.0 - satellite_group_ratios["mobility"]).clamp(min=0.0, max=1.0) * 0.10
            ).clamp(min=0.0, max=1.0)
            satellite_pairs = [
                ("proxy_plan_satellite_width", satellite_group_ratios["road_surface"], "satellite_width_target", 1.05),
                ("proxy_plan_satellite_connectivity", satellite_connectivity_actual, "satellite_connectivity_target", 1.00),
                ("proxy_plan_satellite_heritage", satellite_heritage_actual, "satellite_heritage_target", 0.95),
                ("proxy_plan_satellite_greening", satellite_group_ratios["vegetation"], "satellite_greening_target", 0.90),
                ("proxy_plan_satellite_service_access", satellite_service_access_actual, "satellite_service_access_target", 0.90),
                ("proxy_plan_satellite_showcase", satellite_showcase_actual, "satellite_showcase_target", 0.85),
            ]
            for component_name, actual, target_name, weight in satellite_pairs:
                target = _optional_target_tensor(
                    historical_plan_targets,
                    target_name,
                    reference=actual,
                )
                valid = ~torch.isnan(target)
                target_resolved = torch.where(valid, target, actual)
                delta = torch.where(
                    valid,
                    torch.abs(actual - target_resolved),
                    torch.zeros_like(actual),
                )
                components[component_name] = {
                    "actual": actual,
                    "target": target_resolved,
                    "abs_delta": delta,
                    "weighted_delta": delta * weight,
                    "weight": weight,
                }

    numerator = None
    denominator = 0.0
    for component in components.values():
        weighted_delta = component["weighted_delta"]
        numerator = weighted_delta if numerator is None else numerator + weighted_delta
        denominator += float(component["weight"])
    if numerator is None:
        return torch.zeros_like(next(iter(street_group_ratios.values())))
    return numerator / max(denominator, 1e-6)
