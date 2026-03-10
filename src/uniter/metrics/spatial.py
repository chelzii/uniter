from __future__ import annotations

import torch


def compute_class_ratios(
    segmentation_logits: torch.Tensor,
    image_region_index: torch.Tensor,
    num_regions: int,
) -> torch.Tensor:
    predicted = segmentation_logits.argmax(dim=1)
    num_labels = segmentation_logits.shape[1]
    per_image_counts = []
    for label_map in predicted:
        counts = torch.bincount(label_map.reshape(-1), minlength=num_labels).float()
        per_image_counts.append(counts)
    image_counts = torch.stack(per_image_counts, dim=0)

    region_counts = torch.zeros(
        num_regions,
        num_labels,
        device=segmentation_logits.device,
        dtype=segmentation_logits.dtype,
    )
    region_counts.index_add_(0, image_region_index, image_counts.to(segmentation_logits.device))
    return region_counts / region_counts.sum(dim=1, keepdim=True).clamp_min(1.0)


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


def compute_ifi(
    group_ratios: dict[str, torch.Tensor],
    *,
    target_profile: dict[str, float],
    weights: dict[str, float],
) -> torch.Tensor:
    components = compute_ifi_components(
        group_ratios,
        target_profile=target_profile,
        weights=weights,
    )
    numerator = None
    denominator = 0.0
    for component in components.values():
        weighted_delta = component["weighted_delta"]
        numerator = weighted_delta if numerator is None else numerator + weighted_delta
        denominator += float(component["weight"])
    if numerator is None:
        raise ValueError("No group ratios were provided for IFI computation.")
    return numerator / max(denominator, 1e-6)


def compute_ifi_components(
    group_ratios: dict[str, torch.Tensor],
    *,
    target_profile: dict[str, float],
    weights: dict[str, float],
) -> dict[str, dict[str, torch.Tensor | float]]:
    components: dict[str, dict[str, torch.Tensor | float]] = {}
    for group_name, actual in group_ratios.items():
        target = float(target_profile.get(group_name, 0.0))
        weight = float(weights.get(group_name, 1.0))
        abs_delta = torch.abs(actual - target)
        components[group_name] = {
            "actual": actual,
            "target": target,
            "abs_delta": abs_delta,
            "weighted_delta": abs_delta * weight,
            "weight": weight,
        }
    return components
