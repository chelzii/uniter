from __future__ import annotations

import csv
import difflib
import json
import math
from pathlib import Path
from typing import Any
import unicodedata

TARGET_FIELDS = {
    "lost_space_label",
    "sentiment_label",
    "historical_sentiment_label",
    "ifi",
    "mdi",
    "iai",
}
TEXT_LABEL_FIELDS = {
    "current_sentiment_labels": "sentiment_label",
    "historical_sentiment_labels": "historical_sentiment_label",
}
LOST_SPACE_LABELS = {"none": 0, "light": 1, "moderate": 2, "severe": 3}


def _non_empty_string(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _read_json_payload(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON annotation file: {path}") from exc
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    raise ValueError(f"Unsupported JSON annotation payload: {path}")


def _read_jsonl_payload(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                records.append(payload)
    return records


def _read_csv_payload(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _iter_annotation_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _read_csv_payload(path)
    if suffix == ".json":
        return _read_json_payload(path)
    if suffix == ".jsonl":
        return _read_jsonl_payload(path)
    return []


def _resolve_numeric_target(
    field_name: str,
    value: object,
    *,
    path: Path,
) -> float | int | None:
    if value in (None, ""):
        return None
    if field_name == "lost_space_label":
        if isinstance(value, int):
            return value
        normalized = str(value).strip().lower()
        if normalized in LOST_SPACE_LABELS:
            return LOST_SPACE_LABELS[normalized]
        try:
            return int(normalized)
        except ValueError as exc:
            raise ValueError(f"Unsupported lost space label in {path}: {value}") from exc
    if field_name in {"sentiment_label", "historical_sentiment_label"}:
        return int(str(value).strip())
    return float(str(value).strip())


def _round_half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def _aggregate_target_values(field_name: str, values: list[float | int]) -> float | int | None:
    if not values:
        return None
    if field_name in {"sentiment_label", "historical_sentiment_label", "lost_space_label"}:
        return _round_half_up(sum(float(value) for value in values) / len(values))
    return sum(float(value) for value in values) / len(values)


def _normalize_text_key(value: object) -> str | None:
    text = _non_empty_string(value)
    if text is None:
        return None
    normalized = unicodedata.normalize("NFKC", text)
    collapsed = "".join(normalized.split())
    stripped = "".join(
        character
        for character in collapsed
        if not unicodedata.category(character).startswith(("P", "S", "M", "C"))
    )
    return stripped or None


def _select_best_containment_candidate(
    text_key: str,
    candidates: list[dict[str, Any]],
) -> dict[str, Any] | None:
    matches: list[tuple[int, float, dict[str, Any]]] = []
    for candidate in candidates:
        candidate_key = candidate["text_key"]
        if text_key in candidate_key or candidate_key in text_key:
            overlap_length = min(len(text_key), len(candidate_key))
            coverage = overlap_length / max(len(text_key), len(candidate_key))
            matches.append((overlap_length, coverage, candidate))
    if not matches:
        return None
    matches.sort(key=lambda item: (item[0], item[1]), reverse=True)
    best_overlap, best_coverage, best_candidate = matches[0]
    second_overlap = matches[1][0] if len(matches) > 1 else 0
    second_coverage = matches[1][1] if len(matches) > 1 else 0.0
    if best_overlap < 12:
        return None
    if best_overlap < 24 and best_coverage < 0.35:
        return None
    if second_overlap == best_overlap and second_coverage >= best_coverage - 0.05:
        return None
    return best_candidate


def _select_best_fuzzy_candidate(
    text_key: str,
    candidates: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if len(text_key) < 12:
        return None
    scored: list[tuple[float, dict[str, Any]]] = []
    for candidate in candidates:
        ratio = difflib.SequenceMatcher(a=text_key, b=candidate["text_key"]).ratio()
        scored.append((ratio, candidate))
    if not scored:
        return None
    scored.sort(key=lambda item: item[0], reverse=True)
    best_ratio, best_candidate = scored[0]
    second_ratio = scored[1][0] if len(scored) > 1 else 0.0
    best_length = min(len(text_key), len(best_candidate["text_key"]))
    if best_ratio >= 0.9:
        return best_candidate
    if best_ratio >= 0.75 and best_ratio - second_ratio >= 0.1:
        return best_candidate
    if best_ratio >= 0.7 and best_length >= 40 and best_ratio - second_ratio >= 0.2:
        return best_candidate
    if best_ratio >= 0.6 and best_length >= 80 and best_ratio - second_ratio >= 0.25:
        return best_candidate
    return None


def _resolve_lookup_region_id(record: dict[str, Any]) -> str:
    metadata = record.get("metadata", {})
    if isinstance(metadata, dict):
        parent_region_id = _non_empty_string(metadata.get("parent_region_id"))
        if parent_region_id is not None:
            return parent_region_id
    region_id = _non_empty_string(record.get("region_id"))
    if region_id is None:
        raise ValueError("Manifest record is missing region_id.")
    return region_id


def _discover_annotation_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".csv", ".json", ".jsonl"}
    )


def _collect_updates(
    annotation_root: Path,
) -> tuple[
    dict[str, dict[str, float | int]],
    dict[str, dict[str, list[dict[str, Any]]]],
    list[str],
]:
    values_by_region: dict[str, dict[str, list[float | int]]] = {}
    text_labels_by_region: dict[str, dict[str, list[dict[str, Any]]]] = {}
    used_files: list[str] = []
    for path in _discover_annotation_files(annotation_root):
        records = _iter_annotation_records(path)
        if not records:
            continue
        file_used = False
        for record in records:
            region_id = _non_empty_string(record.get("region_id"))
            if region_id is None:
                continue
            for field_name in TARGET_FIELDS:
                raw_value = record.get(field_name)
                if field_name == "lost_space_label" and raw_value in (None, ""):
                    raw_value = record.get("lost_space_label_id")
                resolved = _resolve_numeric_target(field_name, raw_value, path=path)
                if resolved is None:
                    continue
                values_by_region.setdefault(region_id, {}).setdefault(field_name, []).append(
                    resolved
                )
                file_used = True
                for text_label_field, target_field in TEXT_LABEL_FIELDS.items():
                    if field_name != target_field:
                        continue
                    text_key = _normalize_text_key(record.get("text"))
                    if text_key is None:
                        continue
                    text_labels_by_region.setdefault(region_id, {}).setdefault(
                        text_label_field,
                        [],
                    ).append(
                        {
                            "text_key": text_key,
                            "label": int(resolved),
                        }
                    )
        if file_used:
            used_files.append(str(path.resolve()))

    aggregated: dict[str, dict[str, float | int]] = {}
    for region_id, target_lists in values_by_region.items():
        region_updates: dict[str, float | int] = {}
        for field_name, values in target_lists.items():
            resolved = _aggregate_target_values(field_name, values)
            if resolved is not None:
                region_updates[field_name] = resolved
        if region_updates:
            aggregated[region_id] = region_updates
    return aggregated, text_labels_by_region, used_files


def _match_text_labels(
    texts: list[object],
    label_entries: list[dict[str, Any]],
) -> tuple[list[int | None], int]:
    candidates: list[dict[str, Any]] = []
    labels_by_text: dict[str, list[dict[str, Any]]] = {}
    for entry in label_entries:
        text_key = _normalize_text_key(entry.get("text_key"))
        label = entry.get("label")
        if text_key is None or label is None:
            continue
        candidate = {
            "text_key": text_key,
            "label": int(label),
            "used": False,
        }
        labels_by_text.setdefault(text_key, []).append(candidate)
        candidates.append(candidate)

    resolved: list[int | None] = []
    matched = 0
    for text in texts:
        text_key = _normalize_text_key(text)
        if text_key is None:
            resolved.append(None)
            continue
        exact_candidates = [candidate for candidate in labels_by_text.get(text_key, []) if not candidate["used"]]
        if exact_candidates:
            exact_candidates[0]["used"] = True
            resolved.append(exact_candidates[0]["label"])
            matched += 1
            continue

        fallback_candidates = [candidate for candidate in candidates if not candidate["used"]]
        containment_candidate = _select_best_containment_candidate(text_key, fallback_candidates)
        if containment_candidate is not None:
            containment_candidate["used"] = True
            resolved.append(containment_candidate["label"])
            matched += 1
            continue

        fuzzy_candidate = _select_best_fuzzy_candidate(text_key, fallback_candidates)
        if fuzzy_candidate is not None:
            fuzzy_candidate["used"] = True
            resolved.append(fuzzy_candidate["label"])
            matched += 1
            continue

        resolved.append(None)
    return resolved, matched


def import_annotations_into_manifest(
    *,
    manifest_path: str | Path,
    annotation_root: str | Path,
    output_path: str | Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    manifest = Path(manifest_path).resolve()
    destination = Path(output_path).resolve() if output_path is not None else manifest
    annotation_root_path = Path(annotation_root).resolve()
    records = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    updates_by_region, text_labels_by_region, used_files = _collect_updates(annotation_root_path)

    updated_record_count = 0
    updated_field_count = 0
    updated_text_label_count = 0
    matched_text_label_count = 0
    for record in records:
        targets = record.setdefault("targets", {})
        if not isinstance(targets, dict):
            targets = {}
            record["targets"] = targets
        lookup_region_id = _resolve_lookup_region_id(record)
        region_updates = updates_by_region.get(lookup_region_id, {})
        touched = False
        for field_name, value in region_updates.items():
            if targets.get(field_name) != value:
                targets[field_name] = value
                updated_field_count += 1
                touched = True
        text_label_updates = text_labels_by_region.get(lookup_region_id, {})
        for text_field, label_field in (
            ("current_texts", "current_sentiment_labels"),
            ("historical_texts", "historical_sentiment_labels"),
        ):
            texts = record.get(text_field, [])
            if not isinstance(texts, list) or not texts:
                continue
            matched_labels, matched_count = _match_text_labels(
                texts,
                text_label_updates.get(label_field, []),
            )
            if matched_count == 0:
                continue
            matched_text_label_count += matched_count
            if record.get(label_field) != matched_labels:
                record[label_field] = matched_labels
                updated_text_label_count += matched_count
                touched = True
        if touched:
            updated_record_count += 1

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    summary = {
        "manifest_path": str(destination),
        "annotation_root": str(annotation_root_path),
        "used_files": used_files,
        "used_file_count": len(used_files),
        "region_count_with_updates": len(updates_by_region),
        "updated_record_count": updated_record_count,
        "updated_field_count": updated_field_count,
        "updated_text_label_count": updated_text_label_count,
        "matched_text_label_count": matched_text_label_count,
        "targets_by_region": updates_by_region,
    }
    return destination, summary
