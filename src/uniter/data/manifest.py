from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

VALID_SPLITS = {"train", "val", "test"}


def _is_integer_target(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_numeric_target(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


@dataclass(slots=True)
class RegionTargets:
    lost_space_label: int | None = None
    sentiment_label: int | None = None
    historical_sentiment_label: int | None = None
    ifi: float | None = None
    mdi: float | None = None
    iai: float | None = None


@dataclass(slots=True)
class RegionRecord:
    region_id: str
    split: str
    image_paths: list[Path]
    current_texts: list[str]
    segmentation_mask_paths: list[Path | None] = field(default_factory=list)
    current_sentiment_labels: list[int | None] = field(default_factory=list)
    historical_texts: list[str] = field(default_factory=list)
    historical_sentiment_labels: list[int | None] = field(default_factory=list)
    identity_texts: list[str] = field(default_factory=list)
    metadata: dict[str, str | int | float | bool | None] = field(default_factory=dict)
    targets: RegionTargets = field(default_factory=RegionTargets)


def _resolve_path(
    raw_path: str,
    *,
    manifest_dir: Path,
    image_root: Path | None,
) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    if image_root is not None:
        return image_root / candidate
    return manifest_dir / candidate


def _parse_targets(payload: dict[str, object] | None) -> RegionTargets:
    payload = payload or {}
    return RegionTargets(
        lost_space_label=payload.get("lost_space_label"),  # type: ignore[arg-type]
        sentiment_label=payload.get("sentiment_label"),  # type: ignore[arg-type]
        historical_sentiment_label=payload.get("historical_sentiment_label"),  # type: ignore[arg-type]
        ifi=payload.get("ifi"),  # type: ignore[arg-type]
        mdi=payload.get("mdi"),  # type: ignore[arg-type]
        iai=payload.get("iai"),  # type: ignore[arg-type]
    )


def _validate_record_payload(payload: dict[str, object], *, line_number: int) -> list[str]:
    errors: list[str] = []
    required = ["region_id", "split", "image_paths", "current_texts"]
    for key in required:
        if key not in payload:
            errors.append(f"line {line_number}: missing required field '{key}'")

    region_id = payload.get("region_id")
    if region_id is not None and (not isinstance(region_id, str) or not region_id.strip()):
        errors.append(f"line {line_number}: 'region_id' must be a non-empty string")

    split = payload.get("split")
    if split is not None:
        if not isinstance(split, str) or not split.strip():
            errors.append(f"line {line_number}: 'split' must be a non-empty string")
        elif split not in VALID_SPLITS:
            errors.append(
                f"line {line_number}: 'split' must be one of {sorted(VALID_SPLITS)}"
            )

    if "image_paths" in payload and not isinstance(payload["image_paths"], list):
        errors.append(f"line {line_number}: 'image_paths' must be a list")
    elif "image_paths" in payload:
        image_paths = payload["image_paths"]
        if not image_paths:
            errors.append(f"line {line_number}: 'image_paths' must not be empty")
        else:
            for image_path in image_paths:
                if not isinstance(image_path, str) or not image_path.strip():
                    errors.append(
                        f"line {line_number}: each 'image_paths' entry must be a non-empty string"
                    )

    if "segmentation_mask_paths" in payload:
        mask_paths = payload["segmentation_mask_paths"]
        if not isinstance(mask_paths, list):
            errors.append(f"line {line_number}: 'segmentation_mask_paths' must be a list")
        elif "image_paths" in payload and isinstance(payload["image_paths"], list):
            if len(mask_paths) != len(payload["image_paths"]):
                errors.append(
                    "line "
                    f"{line_number}: 'segmentation_mask_paths' must match the length of "
                    "'image_paths'"
                )
        if isinstance(mask_paths, list):
            for mask_path in mask_paths:
                if mask_path is not None and (not isinstance(mask_path, str) or not mask_path.strip()):
                    errors.append(
                        "line "
                        f"{line_number}: each 'segmentation_mask_paths' entry must be a "
                        "non-empty string or null"
                    )

    if "current_texts" in payload and not isinstance(payload["current_texts"], list):
        errors.append(f"line {line_number}: 'current_texts' must be a list")
    elif "current_texts" in payload:
        current_texts = payload["current_texts"]
        if not current_texts:
            errors.append(f"line {line_number}: 'current_texts' must not be empty")
        else:
            for text in current_texts:
                if not isinstance(text, str) or not text.strip():
                    errors.append(
                        f"line {line_number}: each 'current_texts' entry must be a non-empty string"
                    )

    if "historical_texts" in payload and not isinstance(payload["historical_texts"], list):
        errors.append(f"line {line_number}: 'historical_texts' must be a list")
    elif "historical_texts" in payload:
        for text in payload["historical_texts"]:
            if not isinstance(text, str) or not text.strip():
                errors.append(
                    f"line {line_number}: each 'historical_texts' entry must be a non-empty string"
                )

    if "identity_texts" in payload and not isinstance(payload["identity_texts"], list):
        errors.append(f"line {line_number}: 'identity_texts' must be a list")
    elif "identity_texts" in payload:
        for text in payload["identity_texts"]:
            if not isinstance(text, str) or not text.strip():
                errors.append(
                    f"line {line_number}: each 'identity_texts' entry must be a non-empty string"
                )

    metadata = payload.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        errors.append(f"line {line_number}: 'metadata' must be a JSON object")

    for text_field, label_field in (
        ("current_texts", "current_sentiment_labels"),
        ("historical_texts", "historical_sentiment_labels"),
    ):
        if label_field not in payload:
            continue
        labels = payload[label_field]
        if not isinstance(labels, list):
            errors.append(f"line {line_number}: '{label_field}' must be a list")
            continue
        texts = payload.get(text_field, [])
        if isinstance(texts, list) and len(labels) != len(texts):
            errors.append(
                f"line {line_number}: '{label_field}' must match the length of '{text_field}'"
            )
        for label in labels:
            if label is not None and not _is_integer_target(label):
                errors.append(
                    f"line {line_number}: each '{label_field}' entry must be an integer or null"
                )

    targets = payload.get("targets")
    if targets is not None:
        if not isinstance(targets, dict):
            errors.append(f"line {line_number}: 'targets' must be a JSON object")
        else:
            for key in (
                "lost_space_label",
                "sentiment_label",
                "historical_sentiment_label",
            ):
                value = targets.get(key)
                if value is not None and not _is_integer_target(value):
                    errors.append(
                        f"line {line_number}: target '{key}' must be an integer or null"
                    )
            historical_sentiment_label = targets.get("historical_sentiment_label")
            historical_texts = payload.get("historical_texts", [])
            if historical_sentiment_label is not None and not historical_texts:
                errors.append(
                    "line "
                    f"{line_number}: target 'historical_sentiment_label' requires non-empty "
                    "'historical_texts'"
                )
            for key in ("ifi", "mdi"):
                value = targets.get(key)
                if value is not None and not _is_numeric_target(value):
                    errors.append(
                        f"line {line_number}: target '{key}' must be a number or null"
                    )
            iai = targets.get("iai")
            if iai is not None and not _is_numeric_target(iai):
                errors.append(f"line {line_number}: target 'iai' must be a number or null")
    return errors


def load_manifest(
    manifest_path: str | Path,
    *,
    image_root: str | Path | None = None,
    check_files: bool = False,
) -> list[RegionRecord]:
    path = Path(manifest_path)
    root = Path(image_root) if image_root else None
    errors = validate_manifest(
        manifest_path=path,
        image_root=root,
        check_files=check_files,
    )
    if errors:
        raise ValueError("Manifest validation failed:\n" + "\n".join(errors))

    records: list[RegionRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            image_paths = [
                _resolve_path(image_path, manifest_dir=path.parent, image_root=root)
                for image_path in payload["image_paths"]
            ]
            raw_mask_paths = payload.get("segmentation_mask_paths", [])
            segmentation_mask_paths = [
                (
                    _resolve_path(mask_path, manifest_dir=path.parent, image_root=root)
                    if mask_path is not None
                    else None
                )
                for mask_path in raw_mask_paths
            ]
            records.append(
                RegionRecord(
                    region_id=payload["region_id"],
                    split=payload["split"],
                    image_paths=image_paths,
                    segmentation_mask_paths=segmentation_mask_paths,
                    current_texts=[str(text) for text in payload["current_texts"]],
                    current_sentiment_labels=[
                        int(label) if label is not None else None
                        for label in payload.get("current_sentiment_labels", [])
                    ],
                    historical_texts=[
                        str(text) for text in payload.get("historical_texts", [])
                    ],
                    historical_sentiment_labels=[
                        int(label) if label is not None else None
                        for label in payload.get("historical_sentiment_labels", [])
                    ],
                    identity_texts=[str(text) for text in payload.get("identity_texts", [])],
                    metadata=payload.get("metadata", {}),
                    targets=_parse_targets(payload.get("targets")),
                )
            )
    return records


def validate_manifest(
    *,
    manifest_path: Path,
    image_root: Path | None = None,
    check_files: bool = False,
) -> list[str]:
    errors: list[str] = []
    if not manifest_path.exists():
        return [f"Manifest file does not exist: {manifest_path}"]

    seen_region_ids: dict[str, int] = {}
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_number}: invalid JSON - {exc.msg}")
                continue

            if not isinstance(payload, dict):
                errors.append(f"line {line_number}: record must be a JSON object")
                continue

            errors.extend(_validate_record_payload(payload, line_number=line_number))
            region_id = payload.get("region_id")
            if isinstance(region_id, str) and region_id.strip():
                previous_line = seen_region_ids.get(region_id)
                if previous_line is not None:
                    errors.append(
                        f"line {line_number}: duplicate 'region_id' '{region_id}' "
                        f"(already defined at line {previous_line})"
                    )
                else:
                    seen_region_ids[region_id] = line_number
            if check_files and isinstance(payload.get("image_paths"), list):
                for raw_path in payload["image_paths"]:
                    resolved = _resolve_path(
                        raw_path,
                        manifest_dir=manifest_path.parent,
                        image_root=image_root,
                    )
                    if not resolved.exists():
                        errors.append(
                            f"line {line_number}: image file does not exist - {resolved}"
                        )
            if check_files and isinstance(payload.get("segmentation_mask_paths"), list):
                for raw_path in payload["segmentation_mask_paths"]:
                    if raw_path is None:
                        continue
                    resolved = _resolve_path(
                        raw_path,
                        manifest_dir=manifest_path.parent,
                        image_root=image_root,
                    )
                    if not resolved.exists():
                        errors.append(
                            f"line {line_number}: segmentation mask does not exist - {resolved}"
                        )

    return errors
