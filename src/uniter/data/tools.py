from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from uniter.data.manifest import VALID_SPLITS, validate_manifest

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TEXT_SUFFIXES = {".txt", ".json", ".jsonl"}


def _sorted_files(directory: Path, *, suffixes: set[str]) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in suffixes
    )


def _load_split_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Split map must be a JSON object of region_id -> split.")
    result: dict[str, str] = {}
    for region_id, split in payload.items():
        if not isinstance(region_id, str) or not isinstance(split, str):
            raise ValueError("Split map entries must be string -> string.")
        if split not in VALID_SPLITS:
            raise ValueError(f"Unsupported split '{split}' in split map.")
        result[region_id] = split
    return result


def _discover_image_regions(root: Path | None) -> dict[str, list[Path]]:
    if root is None or not root.exists():
        return {}
    regions: dict[str, list[Path]] = {}
    for child in sorted(path for path in root.iterdir() if path.is_dir()):
        files = _sorted_files(child, suffixes=IMAGE_SUFFIXES)
        if files:
            regions[child.name] = files
    return regions


def _discover_region_files(root: Path | None, *, suffixes: set[str]) -> dict[str, Path]:
    if root is None or not root.exists():
        return {}
    mapping: dict[str, Path] = {}
    for file_path in _sorted_files(root, suffixes=suffixes):
        mapping[file_path.stem] = file_path
    return mapping


def _relative_path(path: Path, *, base_dir: Path) -> str:
    return os.path.relpath(path.resolve(), start=base_dir.resolve())


def _load_text_entries(path: Path | None) -> list[str]:
    if path is None:
        return []
    if path.suffix.lower() == ".txt":
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if path.suffix.lower() == ".jsonl":
        texts: list[str] = []
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, str) and payload.strip():
                texts.append(payload.strip())
            elif isinstance(payload, dict):
                text = payload.get("text")
                if isinstance(text, str) and text.strip():
                    texts.append(text.strip())
        return texts
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [str(item).strip() for item in payload if str(item).strip()]
    if isinstance(payload, dict):
        texts = payload.get("texts", [])
        if isinstance(texts, list):
            return [str(item).strip() for item in texts if str(item).strip()]
    raise ValueError(f"Unsupported text payload format: {path}")


def _load_metadata(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Metadata file must contain a JSON object: {path}")
    return payload


def _match_mask_paths(image_paths: list[Path], mask_paths: list[Path]) -> list[Path | None]:
    if not mask_paths:
        return [None] * len(image_paths)
    by_stem = {path.stem: path for path in mask_paths}
    matched: list[Path | None] = []
    for image_path in image_paths:
        matched.append(by_stem.get(image_path.stem))
    return matched


def build_manifest_from_directories(
    *,
    output_path: str | Path,
    image_dir: str | Path | None,
    current_text_dir: str | Path | None,
    historical_text_dir: str | Path | None = None,
    identity_text_dir: str | Path | None = None,
    segmentation_mask_dir: str | Path | None = None,
    metadata_dir: str | Path | None = None,
    split_map_path: str | Path | None = None,
    default_split: str = "train",
    strict: bool = False,
) -> tuple[Path, dict[str, Any]]:
    if default_split not in VALID_SPLITS:
        raise ValueError(f"default_split must be one of {sorted(VALID_SPLITS)}.")

    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    base_dir = destination.parent

    image_regions = _discover_image_regions(Path(image_dir).resolve()) if image_dir else {}
    mask_regions = (
        _discover_image_regions(Path(segmentation_mask_dir).resolve())
        if segmentation_mask_dir
        else {}
    )
    current_files = (
        _discover_region_files(Path(current_text_dir).resolve(), suffixes=TEXT_SUFFIXES)
        if current_text_dir
        else {}
    )
    historical_files = (
        _discover_region_files(Path(historical_text_dir).resolve(), suffixes=TEXT_SUFFIXES)
        if historical_text_dir
        else {}
    )
    identity_files = (
        _discover_region_files(Path(identity_text_dir).resolve(), suffixes=TEXT_SUFFIXES)
        if identity_text_dir
        else {}
    )
    metadata_files = (
        _discover_region_files(Path(metadata_dir).resolve(), suffixes={".json"})
        if metadata_dir
        else {}
    )
    split_map = _load_split_map(Path(split_map_path).resolve()) if split_map_path else {}

    region_ids = sorted(
        set(image_regions)
        | set(current_files)
        | set(historical_files)
        | set(identity_files)
        | set(metadata_files)
        | set(mask_regions)
    )
    if not region_ids:
        raise RuntimeError("No region data was discovered from the provided directories.")

    rows: list[str] = []
    skipped: list[dict[str, str]] = []
    for region_id in region_ids:
        image_paths = image_regions.get(region_id, [])
        current_texts = _load_text_entries(current_files.get(region_id))
        if not image_paths or not current_texts:
            reason = []
            if not image_paths:
                reason.append("missing_images")
            if not current_texts:
                reason.append("missing_current_texts")
            skipped.append({"region_id": region_id, "reason": "+".join(reason)})
            if strict:
                raise RuntimeError(
                    f"Region '{region_id}' is incomplete: {'+'.join(reason)}."
                )
            continue

        mask_paths = _match_mask_paths(image_paths, mask_regions.get(region_id, []))
        payload = {
            "region_id": region_id,
            "split": split_map.get(region_id, default_split),
            "image_paths": [_relative_path(path, base_dir=base_dir) for path in image_paths],
            "segmentation_mask_paths": [
                _relative_path(path, base_dir=base_dir) if path is not None else None
                for path in mask_paths
            ],
            "current_texts": current_texts,
            "historical_texts": _load_text_entries(historical_files.get(region_id)),
            "identity_texts": _load_text_entries(identity_files.get(region_id)),
            "metadata": _load_metadata(metadata_files.get(region_id)),
            "targets": {
                "lost_space_label": None,
                "sentiment_label": None,
                "historical_sentiment_label": None,
                "ifi": None,
                "mdi": None,
                "iai": None,
            },
        }
        rows.append(json.dumps(payload, ensure_ascii=False))

    destination.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    summary = summarize_manifest(destination)
    summary["skipped_regions"] = skipped
    summary["skipped_region_count"] = len(skipped)
    return destination, summary


def summarize_manifest(manifest_path: str | Path) -> dict[str, Any]:
    path = Path(manifest_path).resolve()
    errors = validate_manifest(manifest_path=path, check_files=False)
    if errors:
        raise ValueError("Manifest validation failed:\n" + "\n".join(errors))

    split_counts = {split: 0 for split in sorted(VALID_SPLITS)}
    image_counts: list[int] = []
    current_text_counts: list[int] = []
    historical_text_counts: list[int] = []
    identity_text_counts: list[int] = []
    segmentation_coverage = 0
    target_counts = {
        "sentiment_label": 0,
        "historical_sentiment_label": 0,
        "lost_space_label": 0,
        "ifi": 0,
        "mdi": 0,
        "iai": 0,
    }
    region_count = 0
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            region_count += 1
            split_counts[payload["split"]] += 1
            image_counts.append(len(payload["image_paths"]))
            current_text_counts.append(len(payload["current_texts"]))
            historical_text_counts.append(len(payload.get("historical_texts", [])))
            identity_text_counts.append(len(payload.get("identity_texts", [])))
            segmentation_coverage += sum(
                int(mask_path is not None)
                for mask_path in payload.get("segmentation_mask_paths", [])
            )
            targets = payload.get("targets", {})
            if isinstance(targets, dict):
                for key in target_counts:
                    target_counts[key] += int(targets.get(key) is not None)

    def _stat(values: list[int]) -> dict[str, float | int]:
        if not values:
            return {"count": 0, "mean": 0.0, "min": 0, "max": 0}
        return {
            "count": len(values),
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }

    return {
        "manifest_path": str(path),
        "region_count": region_count,
        "split_counts": split_counts,
        "images_per_region": _stat(image_counts),
        "current_texts_per_region": _stat(current_text_counts),
        "historical_texts_per_region": _stat(historical_text_counts),
        "identity_texts_per_region": _stat(identity_text_counts),
        "segmentation_mask_count": segmentation_coverage,
        "target_counts": target_counts,
    }
