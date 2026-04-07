from __future__ import annotations

import csv
import json
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

from uniter.data.manifest import load_manifest


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _normalized_manifest_path(manifest_path: str | Path | None) -> str | None:
    if manifest_path in (None, ""):
        return None
    return str(Path(manifest_path).resolve())


@lru_cache(maxsize=8)
def _manifest_records(manifest_path: str | None) -> tuple[Any, ...]:
    if manifest_path is None:
        return ()
    path = Path(manifest_path)
    if not path.exists():
        return ()
    return tuple(load_manifest(path, check_files=False))


def _workspace_root_from_image_path(image_path: Path) -> Path | None:
    current = image_path.resolve()
    for parent in (current.parent, *current.parents):
        if parent.name == "images":
            return parent.parent
    return None


@lru_cache(maxsize=8)
def _candidate_data_roots(manifest_path: str | None) -> tuple[Path, ...]:
    candidates: list[Path] = []
    if manifest_path is not None:
        manifest = Path(manifest_path).resolve()
        candidates.append(manifest.parent)
        for record in _manifest_records(manifest_path):
            for image_path in getattr(record, "image_paths", []):
                root = _workspace_root_from_image_path(Path(image_path))
                if root is not None:
                    candidates.append(root)
                    break
            if len(candidates) >= 4:
                break
    candidates.append(_project_root() / "datasets" / "data_workspace_cleaned")

    unique_candidates: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_candidates.append(resolved)
    return tuple(unique_candidates)


def _resolve_existing_data_root(manifest_path: str | None) -> Path | None:
    for candidate in _candidate_data_roots(manifest_path):
        if (
            (candidate / "current_texts" / "current_text_index_cleaned.csv").exists()
            or (candidate / "images" / "image_index.csv").exists()
            or (candidate / "metadata").exists()
        ):
            return candidate
    return None


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _group_rows_by_region(path: Path) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in _read_csv_rows(path):
        region_id = row.get("region_id", "").strip()
        if not region_id:
            continue
        grouped.setdefault(region_id, []).append(row)
    return grouped


def _bool_from_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return {
        key: counter[key]
        for key in sorted(counter, key=lambda item: (-counter[item], item))
    }


def _non_empty_string(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_text(value: object) -> str | None:
    text = _non_empty_string(value)
    if text is None:
        return None
    return " ".join(text.split())


@lru_cache(maxsize=8)
def _current_text_rows_by_region(
    manifest_path: str | None = None,
) -> dict[str, list[dict[str, str]]]:
    data_root = _resolve_existing_data_root(manifest_path)
    if data_root is None:
        return {}
    return _group_rows_by_region(data_root / "current_texts" / "current_text_index_cleaned.csv")


@lru_cache(maxsize=8)
def _image_rows_by_region(
    manifest_path: str | None = None,
) -> dict[str, list[dict[str, str]]]:
    data_root = _resolve_existing_data_root(manifest_path)
    if data_root is None:
        return {}
    return _group_rows_by_region(data_root / "images" / "image_index.csv")


@lru_cache(maxsize=8)
def _metadata_by_region(
    manifest_path: str | None = None,
) -> dict[str, dict[str, Any]]:
    data_root = _resolve_existing_data_root(manifest_path)
    if data_root is None:
        return {}
    metadata_root = data_root / "metadata"
    if not metadata_root.exists():
        return {}
    payloads: dict[str, dict[str, Any]] = {}
    for path in sorted(metadata_root.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        region_id = _non_empty_string(payload.get("region_id")) or path.stem
        payloads[region_id] = payload
    return payloads


def resolve_lookup_region_id(region_id: str, metadata: dict[str, Any]) -> str:
    parent_region_id = _non_empty_string(metadata.get("parent_region_id"))
    return parent_region_id or region_id


def build_region_context(
    *,
    region_id: str,
    metadata: dict[str, Any],
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    manifest_key = _normalized_manifest_path(manifest_path)
    lookup_region_id = resolve_lookup_region_id(region_id, metadata)
    cleaned_metadata = (
        _metadata_by_region(manifest_key)
        if manifest_key is not None
        else _metadata_by_region()
    ).get(lookup_region_id, {})
    current_rows = (
        _current_text_rows_by_region(manifest_key)
        if manifest_key is not None
        else _current_text_rows_by_region()
    ).get(lookup_region_id, [])
    kept_current_rows = [
        row
        for row in current_rows
        if _bool_from_value(row.get("keep_for_training"))
    ]
    image_rows = (
        _image_rows_by_region(manifest_key)
        if manifest_key is not None
        else _image_rows_by_region()
    ).get(lookup_region_id, [])
    bootstrap_point_ids = {
        str(item).strip()
        for item in metadata.get("bootstrap_point_ids", [])
        if str(item).strip()
    }
    selected_image_filenames_full = {
        str(item).strip()
        for item in metadata.get("_selected_image_filenames_full", [])
        if str(item).strip()
    }
    if bootstrap_point_ids:
        scoped_image_rows = [
            row
            for row in image_rows
            if row.get("point_id", "").strip() in bootstrap_point_ids
        ]
    elif selected_image_filenames_full:
        scoped_image_rows = [
            row
            for row in image_rows
            if row.get("file_name", "").strip() in selected_image_filenames_full
        ]
    else:
        scoped_image_rows = image_rows

    used_image_filenames = {
        str(item).strip()
        for item in metadata.get("_used_image_filenames", [])
        if str(item).strip()
    }
    if used_image_filenames:
        used_image_rows = [
            row
            for row in scoped_image_rows
            if row.get("file_name", "").strip() in used_image_filenames
        ]
    else:
        used_image_rows = scoped_image_rows

    selected_current_texts = [
        normalized
        for text in metadata.get("_selected_current_texts", [])
        if (normalized := _normalize_text(text)) is not None
    ]
    selected_current_text_counter = Counter(selected_current_texts)

    if selected_current_text_counter:
        matched_current_rows: list[dict[str, str]] = []
        remaining = Counter(selected_current_text_counter)
        for row in kept_current_rows:
            candidates = [
                _normalize_text(row.get("cleaned_text")),
                _normalize_text(row.get("text")),
            ]
            for candidate in candidates:
                if candidate is None or remaining[candidate] <= 0:
                    continue
                matched_current_rows.append(row)
                remaining[candidate] -= 1
                break
    else:
        matched_current_rows = kept_current_rows

    current_platforms = Counter(
        row["source_platform"].strip()
        for row in matched_current_rows
        if row.get("source_platform", "").strip()
    )
    parent_current_platforms = Counter(
        row["source_platform"].strip()
        for row in kept_current_rows
        if row.get("source_platform", "").strip()
    )
    image_platforms = Counter(
        row["source_platform"].strip()
        for row in scoped_image_rows
        if row.get("source_platform", "").strip()
    )
    street_image_count = sum(
        row.get("image_type", "").strip().lower() == "street"
        for row in scoped_image_rows
    )
    satellite_image_count = sum(
        row.get("image_type", "").strip().lower() == "satellite"
        for row in scoped_image_rows
    )
    has_view_direction = any(
        _non_empty_string(row.get("view_direction")) is not None
        for row in scoped_image_rows
    )
    has_spatial_points = any(
        _non_empty_string(row.get("lon")) is not None
        and _non_empty_string(row.get("lat")) is not None
        for row in scoped_image_rows
    )
    point_ids = {
        row["point_id"].strip()
        for row in scoped_image_rows
        if row.get("point_id", "").strip()
    }
    time_range = cleaned_metadata.get("time_range")
    if not isinstance(time_range, dict):
        time_range = metadata.get("time_range", {})
    if not isinstance(time_range, dict):
        time_range = {}
    data_sources = cleaned_metadata.get("data_sources")
    if not isinstance(data_sources, list):
        data_sources = metadata.get("data_sources", [])
    if not isinstance(data_sources, list):
        data_sources = []

    current_text_count = metadata.get("current_text_count")
    if not isinstance(current_text_count, int):
        current_text_count = cleaned_metadata.get("current_text_count")
    if not isinstance(current_text_count, int):
        current_text_count = len(matched_current_rows)

    historical_text_count = metadata.get("historical_text_count")
    if not isinstance(historical_text_count, int):
        historical_text_count = cleaned_metadata.get("historical_text_count")

    identity_text_count = metadata.get("identity_text_count")
    if not isinstance(identity_text_count, int):
        identity_text_count = cleaned_metadata.get("identity_text_count")

    image_count = metadata.get("image_count")
    if not isinstance(image_count, int):
        image_count = cleaned_metadata.get("image_count")
    if not isinstance(image_count, int):
        image_count = len(scoped_image_rows)

    used_image_count = metadata.get("selected_image_count")
    if not isinstance(used_image_count, int):
        used_image_count = len(used_image_rows)

    parent_image_count = cleaned_metadata.get("image_count")
    if not isinstance(parent_image_count, int):
        parent_image_count = len(image_rows)

    parent_cleaned_current_text_count = cleaned_metadata.get("current_text_count")
    if not isinstance(parent_cleaned_current_text_count, int):
        parent_cleaned_current_text_count = len(kept_current_rows)

    return {
        "lookup_region_id": lookup_region_id,
        "region_name": _non_empty_string(metadata.get("region_name"))
        or _non_empty_string(cleaned_metadata.get("region_name")),
        "site_name": _non_empty_string(metadata.get("site_name"))
        or _non_empty_string(cleaned_metadata.get("site_name")),
        "city": _non_empty_string(metadata.get("city"))
        or _non_empty_string(cleaned_metadata.get("city")),
        "district": _non_empty_string(metadata.get("district"))
        or _non_empty_string(cleaned_metadata.get("district")),
        "region_type": _non_empty_string(metadata.get("region_type"))
        or _non_empty_string(cleaned_metadata.get("region_type")),
        "image_count": image_count,
        "current_text_count": current_text_count,
        "historical_text_count": historical_text_count,
        "identity_text_count": identity_text_count,
        "used_image_count": used_image_count,
        "parent_image_count": parent_image_count,
        "raw_current_text_count": cleaned_metadata.get("raw_current_text_count"),
        "cleaned_current_text_count": len(matched_current_rows),
        "parent_cleaned_current_text_count": parent_cleaned_current_text_count,
        "current_text_source_platforms": _sorted_counter(current_platforms),
        "parent_current_text_source_platforms": _sorted_counter(parent_current_platforms),
        "image_source_platforms": _sorted_counter(image_platforms),
        "street_image_count": street_image_count,
        "satellite_image_count": satellite_image_count,
        "has_view_direction": has_view_direction,
        "has_spatial_points": has_spatial_points,
        "point_count": len(point_ids),
        "data_sources": [str(item) for item in data_sources if str(item).strip()],
        "time_range": time_range,
        "bad_image_skip_count": int(metadata.get("bad_image_skip_count", 0) or 0),
        "had_bad_image_skip": bool(metadata.get("bad_image_skip_count", 0)),
    }
