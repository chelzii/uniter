from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
import math
from typing import Any

import torch
import torch.nn.functional as F

FUNCTION_DOMAIN_KEYWORDS: dict[str, tuple[str, ...]] = {
    "religious_heritage": (
        "寺",
        "禅",
        "宗教",
        "遗产",
        "历史",
        "文化",
        "文保",
        "碑",
    ),
    "tourism_leisure": ("游客", "旅游", "景点", "祈福", "驻留", "游览", "打卡"),
    "mobility_access": (
        "道路",
        "交通",
        "停车",
        "慢行",
        "步行",
        "通达",
        "通行",
        "衔接",
        "连通",
        "街巷",
        "地铁",
        "公交",
        "集散",
    ),
    "residential_administrative": ("居住", "居民", "社区", "行政", "办公", "服务"),
    "commercial_service": ("商业", "消费", "沿街", "店", "餐饮", "零售", "复合型服务"),
}

INTERFACE_ROLE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "buffer_transition": ("缓冲", "过渡", "衔接", "连接"),
    "boundary_edge": ("边界", "界面", "侧翼", "沿街"),
    "mixed_use": ("复合", "叠合", "混行", "交织"),
    "node_flow": ("节点", "驻留", "集散", "通达"),
}

IDENTITY_STRUCTURE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "linear_edge": ("线性", "界面", "沿街", "侧翼"),
    "corridor_network": ("街巷", "廊道", "轴线", "连通"),
    "node_residence": ("节点", "驻留", "停留", "集散"),
    "buffer_zone": ("缓冲", "过渡", "衔接"),
}

CONFLICT_PAIRS: tuple[tuple[str, str], ...] = (
    ("religious_heritage", "commercial_service"),
    ("religious_heritage", "mobility_access"),
    ("religious_heritage", "residential_administrative"),
    ("tourism_leisure", "residential_administrative"),
    ("tourism_leisure", "mobility_access"),
)


def _safe_normalizer(value: float) -> float:
    return max(float(value), 1e-6)


def _embedding_drift(
    identity_embeddings: torch.Tensor | None,
    *,
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    normalizer: float,
    identity_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    drift = image_embeddings.new_full((image_embeddings.shape[0],), float("nan"))
    if identity_embeddings is None:
        return drift

    mask = (
        identity_mask.to(device=image_embeddings.device, dtype=torch.bool)
        if identity_mask is not None
        else torch.ones(image_embeddings.shape[0], device=image_embeddings.device, dtype=torch.bool)
    )
    if not torch.any(mask):
        return drift

    multimodal_center = F.normalize(image_embeddings[mask] + text_embeddings[mask], dim=-1)
    identity_center = F.normalize(identity_embeddings[mask], dim=-1)
    similarity = F.cosine_similarity(multimodal_center, identity_center, dim=-1)
    normalized = (1.0 - similarity) / _safe_normalizer(normalizer)
    drift[mask] = normalized.clamp(min=0.0, max=1.0)
    return drift


def _normalize_text(value: object) -> str:
    return " ".join(str(value).strip().lower().split())


def _iter_identity_texts(metadata: dict[str, Any]) -> Iterable[str]:
    for key in ("_selected_identity_texts", "identity_texts"):
        raw_value = metadata.get(key)
        if not isinstance(raw_value, list):
            continue
        for item in raw_value:
            normalized = _normalize_text(item)
            if normalized:
                yield normalized


def _extract_matches(text: str, keyword_map: dict[str, tuple[str, ...]]) -> set[str]:
    return {
        category
        for category, keywords in keyword_map.items()
        if any(keyword in text for keyword in keywords)
    }


def _normalized_entropy(counter: Counter[str]) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    probabilities = [count / total for count in counter.values() if count > 0]
    if len(probabilities) <= 1:
        return 0.0
    entropy = -sum(probability * math.log(probability) for probability in probabilities)
    return entropy / math.log(len(probabilities))


def _normalized_distribution(
    counter: Counter[str],
    categories: Iterable[str],
) -> dict[str, float]:
    total = sum(counter.values())
    if total <= 0:
        return {category: 0.0 for category in categories}
    return {
        category: counter.get(category, 0) / total
        for category in categories
    }


def build_identity_attribute_vector(metadata: dict[str, Any]) -> dict[str, float] | None:
    texts = list(_iter_identity_texts(metadata))
    if not texts:
        return None

    function_counts: Counter[str] = Counter()
    interface_hits: Counter[str] = Counter()
    structure_hits: Counter[str] = Counter()
    mixed_function_texts = 0
    conflict_texts = 0

    for text in texts:
        function_domains = _extract_matches(text, FUNCTION_DOMAIN_KEYWORDS)
        interface_roles = _extract_matches(text, INTERFACE_ROLE_KEYWORDS)
        structure_roles = _extract_matches(text, IDENTITY_STRUCTURE_KEYWORDS)
        if not function_domains and not interface_roles and not structure_roles:
            continue
        if len(function_domains) > 1:
            mixed_function_texts += 1
        if any(
            left in function_domains and right in function_domains
            for left, right in CONFLICT_PAIRS
        ):
            conflict_texts += 1
        for category in function_domains:
            function_counts[category] += 1
        for role in interface_roles:
            interface_hits[role] += 1
        for role in structure_roles:
            structure_hits[role] += 1

    if not function_counts:
        return None

    total_assignments = sum(function_counts.values())
    function_distribution = _normalized_distribution(
        function_counts,
        FUNCTION_DOMAIN_KEYWORDS.keys(),
    )
    interface_distribution = _normalized_distribution(
        interface_hits,
        INTERFACE_ROLE_KEYWORDS.keys(),
    )
    structure_distribution = _normalized_distribution(
        structure_hits,
        IDENTITY_STRUCTURE_KEYWORDS.keys(),
    )
    dominant_share = max(function_distribution.values())
    dominant_interface_share = max(interface_distribution.values()) if interface_distribution else 0.0
    dominant_structure_share = max(structure_distribution.values()) if structure_distribution else 0.0
    mixture_entropy = _normalized_entropy(function_counts)
    interface_entropy = _normalized_entropy(interface_hits)
    structure_entropy = _normalized_entropy(structure_hits)
    mixture_ratio = mixed_function_texts / max(len(texts), 1)
    transition_ratio = (
        interface_distribution.get("buffer_transition", 0.0)
        + interface_distribution.get("mixed_use", 0.0)
        + structure_distribution.get("buffer_zone", 0.0)
    ) / 3.0

    present_domains = {name for name, count in function_counts.items() if count > 0}
    conflict_pairs_hit = sum(
        1 for left, right in CONFLICT_PAIRS if left in present_domains and right in present_domains
    )
    pair_conflict_ratio = conflict_pairs_hit / max(len(CONFLICT_PAIRS), 1)
    within_text_conflict_ratio = conflict_texts / max(len(texts), 1)
    distributional_conflict = sum(
        function_distribution.get(left, 0.0) * function_distribution.get(right, 0.0)
        for left, right in CONFLICT_PAIRS
    )
    conflict_score = pair_conflict_ratio * 0.30 + within_text_conflict_ratio * 0.30 + min(
        distributional_conflict * 2.0,
        1.0,
    ) * 0.40

    heritage_pressure = function_distribution.get("religious_heritage", 0.0)
    access_pressure = (
        function_distribution.get("commercial_service", 0.0)
        + function_distribution.get("mobility_access", 0.0)
        + function_distribution.get("tourism_leisure", 0.0)
    ) / 3.0
    functional_tension = min(heritage_pressure, access_pressure) * 2.0

    attribute_vector: dict[str, float] = {
        **{f"function_{name}": value for name, value in function_distribution.items()},
        **{f"interface_{name}": value for name, value in interface_distribution.items()},
        **{f"structure_{name}": value for name, value in structure_distribution.items()},
        "function_entropy": mixture_entropy,
        "interface_entropy": interface_entropy,
        "structure_entropy": structure_entropy,
        "function_dominance_gap": 1.0 - dominant_share,
        "interface_dominance_gap": 1.0 - dominant_interface_share,
        "structure_dominance_gap": 1.0 - dominant_structure_share,
        "mixed_function_ratio": mixture_ratio,
        "identity_transition_ratio": transition_ratio,
        "identity_conflict_score": conflict_score,
        "functional_tension": functional_tension,
    }
    return attribute_vector


def _identity_confusion_score(metadata: dict[str, Any]) -> float | None:
    attribute_vector = build_identity_attribute_vector(metadata)
    if attribute_vector is None:
        return None

    score = (
        attribute_vector["function_entropy"] * 0.24
        + attribute_vector["interface_entropy"] * 0.10
        + attribute_vector["structure_entropy"] * 0.08
        + attribute_vector["function_dominance_gap"] * 0.12
        + attribute_vector["interface_dominance_gap"] * 0.06
        + attribute_vector["mixed_function_ratio"] * 0.12
        + attribute_vector["identity_transition_ratio"] * 0.10
        + attribute_vector["identity_conflict_score"] * 0.12
        + attribute_vector["functional_tension"] * 0.06
    )
    return min(max(score, 0.0), 1.0)


def compute_iai(
    identity_embeddings: torch.Tensor | None,
    *,
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    normalizer: float,
    identity_mask: torch.Tensor | None = None,
    metadata: list[dict[str, Any]] | None = None,
) -> torch.Tensor:
    embedding_drift = _embedding_drift(
        identity_embeddings,
        image_embeddings=image_embeddings,
        text_embeddings=text_embeddings,
        normalizer=normalizer,
        identity_mask=identity_mask,
    )
    drift = embedding_drift.clone()
    if metadata is None:
        return drift

    mask = (
        identity_mask.to(device=image_embeddings.device, dtype=torch.bool)
        if identity_mask is not None
        else torch.ones(image_embeddings.shape[0], device=image_embeddings.device, dtype=torch.bool)
    )
    for index, region_metadata in enumerate(metadata):
        if index >= drift.shape[0] or not bool(mask[index].item()):
            continue
        metadata_score = _identity_confusion_score(region_metadata)
        if metadata_score is None:
            continue
        embedding_score = drift[index]
        if torch.isnan(embedding_score):
            drift[index] = metadata_score
        else:
            drift[index] = float(metadata_score) * 0.90 + embedding_score * 0.10
    return drift.clamp(min=0.0, max=1.0)
