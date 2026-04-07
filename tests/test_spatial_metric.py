from __future__ import annotations

import torch

from uniter.metrics.spatial import compute_ifi_components, compute_spatial_proxy_score


def test_compute_ifi_components_adds_structural_discontinuity() -> None:
    street_group_ratios = {
        "road_surface": torch.tensor([0.3]),
        "enclosure": torch.tensor([0.3]),
        "vegetation": torch.tensor([0.2]),
        "sky": torch.tensor([0.1]),
        "mobility": torch.tensor([0.1]),
    }
    per_image_street_group_ratios = {
        "road_surface": torch.tensor([0.5, 0.1]),
        "enclosure": torch.tensor([0.2, 0.4]),
        "vegetation": torch.tensor([0.1, 0.3]),
        "mobility": torch.tensor([0.15, 0.05]),
    }
    satellite_group_ratios = {
        "road_surface": torch.tensor([0.2]),
        "enclosure": torch.tensor([0.4]),
        "vegetation": torch.tensor([0.2]),
        "sky": torch.tensor([0.0]),
        "mobility": torch.tensor([0.05]),
    }

    components = compute_ifi_components(
        street_group_ratios=street_group_ratios,
        street_target_profile={
            "road_surface": torch.tensor([0.22]),
            "enclosure": torch.tensor([0.28]),
            "vegetation": torch.tensor([0.18]),
            "sky": torch.tensor([0.20]),
            "mobility": torch.tensor([0.08]),
        },
        street_weights={
            "road_surface": 1.0,
            "enclosure": 1.2,
            "vegetation": 1.1,
            "sky": 0.8,
            "mobility": 1.1,
        },
        satellite_group_ratios=satellite_group_ratios,
        satellite_target_profile={
            "road_surface": torch.tensor([0.18]),
            "enclosure": torch.tensor([0.34]),
            "vegetation": torch.tensor([0.20]),
            "mobility": torch.tensor([0.06]),
        },
        satellite_weights={
            "road_surface": 1.2,
            "enclosure": 1.2,
            "vegetation": 1.1,
            "mobility": 0.9,
        },
        cross_view_weights={
            "street_boundary_continuity": 1.1,
            "street_satellite_consistency": 1.0,
            "street_view_geometry": 1.0,
        },
        per_image_street_group_ratios=per_image_street_group_ratios,
        street_image_region_index=torch.tensor([0, 0]),
        street_image_point_ids=["p001", "p001"],
        street_image_view_directions=["north", "south"],
        street_image_longitudes=[108.95, 108.95],
        street_image_latitudes=[34.25, 34.25],
        historical_plan_targets={
            "width_road_surface_target": torch.tensor([0.18]),
            "walkability_target": torch.tensor([0.62]),
            "connectivity_target": torch.tensor([0.75]),
            "heritage_target": torch.tensor([0.60]),
            "service_access_target": torch.tensor([0.68]),
            "emergency_access_target": torch.tensor([0.74]),
            "showcase_target": torch.tensor([0.70]),
            "streetscape_style_target": torch.tensor([0.66]),
            "satellite_service_access_target": torch.tensor([0.70]),
            "satellite_showcase_target": torch.tensor([0.66]),
        },
    )

    assert "street_boundary_continuity" in components
    assert "street_satellite_consistency" in components
    assert "street_view_geometry" in components
    assert "historical_plan_width" in components
    assert "historical_plan_walkability" in components
    assert "historical_plan_service_access" in components
    assert "historical_plan_emergency_access" in components
    assert "historical_plan_showcase" in components
    assert "historical_plan_streetscape_style" in components
    assert "historical_plan_satellite_width" in components
    assert "historical_plan_satellite_connectivity" in components
    assert "historical_plan_satellite_heritage" in components
    assert "historical_plan_satellite_service_access" in components
    assert "historical_plan_satellite_showcase" in components
    assert float(components["street_boundary_continuity"]["actual"][0].item()) > 0.0  # type: ignore[index]
    assert float(components["street_satellite_consistency"]["actual"][0].item()) > 0.0  # type: ignore[index]
    assert float(components["street_view_geometry"]["actual"][0].item()) >= 0.0  # type: ignore[index]


def test_compute_spatial_proxy_score_uses_plan_targets_without_masks() -> None:
    proxy_score = compute_spatial_proxy_score(
        street_group_ratios={
            "road_surface": torch.tensor([0.30], requires_grad=True),
            "enclosure": torch.tensor([0.28], requires_grad=True),
            "vegetation": torch.tensor([0.18], requires_grad=True),
            "sky": torch.tensor([0.16], requires_grad=True),
            "mobility": torch.tensor([0.08], requires_grad=True),
            "pedestrian": torch.tensor([0.05], requires_grad=True),
        },
        street_target_profile={
            "road_surface": torch.tensor([0.22]),
            "enclosure": torch.tensor([0.28]),
            "vegetation": torch.tensor([0.18]),
            "sky": torch.tensor([0.20]),
            "mobility": torch.tensor([0.08]),
            "pedestrian": torch.tensor([0.04]),
        },
        street_weights={
            "road_surface": 1.0,
            "enclosure": 1.2,
            "vegetation": 1.1,
            "sky": 0.8,
            "mobility": 1.1,
            "pedestrian": 0.8,
        },
        historical_plan_targets={
            "width_road_surface_target": torch.tensor([0.18]),
            "walkability_target": torch.tensor([0.62]),
            "connectivity_target": torch.tensor([0.75]),
            "heritage_target": torch.tensor([0.60]),
            "service_access_target": torch.tensor([0.68]),
            "showcase_target": torch.tensor([0.70]),
            "streetscape_style_target": torch.tensor([0.66]),
        },
    )

    assert proxy_score.shape == torch.Size([1])
    assert float(proxy_score[0].item()) >= 0.0
