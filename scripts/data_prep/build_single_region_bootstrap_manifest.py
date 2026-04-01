from __future__ import annotations

import csv
import json
import os
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_MANIFEST = REPO_ROOT / "datasets" / "data_workspace_cleaned" / "regions.jsonl"
DEFAULT_CURRENT_TEXT_INDEX = (
    REPO_ROOT / "datasets" / "data_workspace_cleaned" / "current_texts" / "current_text_index_cleaned.csv"
)
DEFAULT_CURRENT_SENTIMENT_ANNOTATIONS = (
    REPO_ROOT
    / "datasets"
    / "annotations"
    / "kaitong_west_lane"
    / "current_sentiment"
    / "current_sentiment_labels.csv"
)
DEFAULT_HISTORICAL_SENTIMENT_ANNOTATIONS = (
    REPO_ROOT
    / "datasets"
    / "annotations"
    / "kaitong_west_lane"
    / "historical_sentiment"
    / "historical_sentiment_labels.csv"
)
DEFAULT_LOST_SPACE_ANNOTATIONS = (
    REPO_ROOT
    / "datasets"
    / "annotations"
    / "kaitong_west_lane"
    / "lost_space"
    / "region_lost_space_label.json"
)
DEFAULT_OUTPUT_MANIFEST = REPO_ROOT / "data" / "regions.jsonl"
DEFAULT_FULL_MANIFEST_COPY = REPO_ROOT / "data" / "regions.single_region.full.jsonl"
DEFAULT_SUMMARY_PATH = REPO_ROOT / "data" / "regions.bootstrap_summary.json"
SEED = 42


@dataclass(frozen=True)
class BootstrapView:
    suffix: str
    split: str


@dataclass(frozen=True)
class CurrentTextRecord:
    record_id: str
    text: str


BOOTSTRAP_VIEWS = (
    BootstrapView("train_a", "train"),
    BootstrapView("train_b", "train"),
    BootstrapView("val", "val"),
    BootstrapView("test", "test"),
)


def _load_single_region_manifest(path: Path) -> dict[str, object]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError(
            f"Expected exactly 1 region record in {path}, found {len(lines)}."
        )
    payload = json.loads(lines[0])
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def _group_point_ids(image_paths: list[str]) -> list[str]:
    point_ids = sorted({Path(image_path).stem.split("_", maxsplit=1)[0] for image_path in image_paths})
    if not point_ids:
        raise ValueError("Source manifest does not contain any image paths.")
    return point_ids


def _chunk_sizes(total: int, parts: int) -> list[int]:
    base = total // parts
    remainder = total % parts
    return [base + (1 if index < remainder else 0) for index in range(parts)]


def _split_ordered(items: list[str], sizes: list[int]) -> list[list[str]]:
    chunks: list[list[str]] = []
    cursor = 0
    for size in sizes:
        chunks.append(items[cursor : cursor + size])
        cursor += size
    return chunks


def _split_shuffled(items: list[str], sizes: list[int], *, seed: int) -> list[list[str]]:
    shuffled = items[:]
    random.Random(seed).shuffle(shuffled)
    return _split_ordered(shuffled, sizes)


def _load_cleaned_current_texts(
    path: Path,
    *,
    region_id: str,
) -> list[CurrentTextRecord]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        records = [
            CurrentTextRecord(record_id=row["record_id"], text=row["text"])
            for row in reader
            if row["region_id"] == region_id and row["keep_for_training"].lower() == "true"
        ]
    if not records:
        raise ValueError(f"No cleaned current texts found for region '{region_id}' in {path}.")
    record_ids = [record.record_id for record in records]
    texts = [record.text for record in records]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError(f"Duplicate current_text record_ids found for region '{region_id}' in {path}.")
    if len(texts) != len(set(texts)):
        raise ValueError(f"Duplicate cleaned current texts found for region '{region_id}' in {path}.")
    return records


def _load_current_sentiment_labels(path: Path) -> dict[str, int]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        labels = {row["record_id"]: int(row["sentiment_label"]) for row in reader}
    if not labels:
        raise ValueError(f"No current sentiment labels found in {path}.")
    return labels


def _load_historical_sentiment_labels(path: Path) -> dict[str, int]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        labels = {row["text"]: int(row["historical_sentiment_label"]) for row in reader}
    if not labels:
        raise ValueError(f"No historical sentiment labels found in {path}.")
    return labels


def _load_lost_space_label(path: Path, *, region_id: str) -> int:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("region_id") != region_id:
        raise ValueError(
            f"Lost-space annotation region_id mismatch: expected '{region_id}', "
            f"found '{payload.get('region_id')}'."
        )
    label = payload.get("lost_space_label_id")
    if not isinstance(label, int):
        raise ValueError(f"Lost-space annotation must contain integer 'lost_space_label_id': {path}")
    return label


def _aggregate_integer_label(labels: list[int]) -> int:
    counts = Counter(labels)
    max_count = max(counts.values())
    candidates = [label for label, count in counts.items() if count == max_count]
    if len(candidates) == 1:
        return candidates[0]
    rounded_mean = round(sum(labels) / len(labels))
    if rounded_mean in candidates:
        return rounded_mean
    return sorted(candidates)[0]


def build_bootstrap_manifest(
    *,
    source_manifest: Path = DEFAULT_SOURCE_MANIFEST,
    current_text_index: Path = DEFAULT_CURRENT_TEXT_INDEX,
    current_sentiment_annotations: Path = DEFAULT_CURRENT_SENTIMENT_ANNOTATIONS,
    historical_sentiment_annotations: Path = DEFAULT_HISTORICAL_SENTIMENT_ANNOTATIONS,
    lost_space_annotations: Path = DEFAULT_LOST_SPACE_ANNOTATIONS,
    output_manifest: Path = DEFAULT_OUTPUT_MANIFEST,
    full_manifest_copy: Path = DEFAULT_FULL_MANIFEST_COPY,
    summary_path: Path = DEFAULT_SUMMARY_PATH,
) -> None:
    payload = _load_single_region_manifest(source_manifest)
    source_root = source_manifest.parent
    output_root = output_manifest.parent

    base_region_id = str(payload["region_id"])
    raw_image_paths = [str(path) for path in payload["image_paths"]]
    image_paths = [
        Path(os.path.relpath((source_root / raw_path).resolve(), output_root.resolve())).as_posix()
        for raw_path in raw_image_paths
    ]
    raw_mask_paths = payload.get("segmentation_mask_paths", [None] * len(raw_image_paths))
    segmentation_mask_paths = [
        (
            Path(
                os.path.relpath((source_root / raw_path).resolve(), output_root.resolve())
            ).as_posix()
            if raw_path is not None
            else None
        )
        for raw_path in raw_mask_paths
    ]
    current_text_records = _load_cleaned_current_texts(current_text_index, region_id=base_region_id)
    current_texts = [record.text for record in current_text_records]
    historical_texts = [str(text) for text in payload.get("historical_texts", [])]
    identity_texts = [str(text) for text in payload.get("identity_texts", [])]
    metadata = dict(payload.get("metadata", {}))
    current_sentiment_by_text = _load_current_sentiment_labels(current_sentiment_annotations)
    current_record_ids = {record.record_id for record in current_text_records}
    missing_current_annotations = sorted(current_record_ids - set(current_sentiment_by_text))
    if missing_current_annotations:
        raise ValueError(
            "Missing current sentiment annotations for cleaned texts:\n"
            + "\n".join(missing_current_annotations[:10])
        )

    historical_sentiment_by_text = _load_historical_sentiment_labels(historical_sentiment_annotations)
    missing_historical_annotations = sorted(set(historical_texts) - set(historical_sentiment_by_text))
    if missing_historical_annotations:
        raise ValueError(
            "Missing historical sentiment annotations for historical texts:\n"
            + "\n".join(missing_historical_annotations[:10])
        )
    historical_sentiment_label = _aggregate_integer_label(
        [historical_sentiment_by_text[text] for text in historical_texts]
    )
    lost_space_label = _load_lost_space_label(lost_space_annotations, region_id=base_region_id)

    base_targets = dict(payload.get("targets", {}))
    base_targets["historical_sentiment_label"] = historical_sentiment_label
    base_targets["lost_space_label"] = lost_space_label

    point_ids = _group_point_ids(image_paths)
    point_chunks = _split_ordered(point_ids, _chunk_sizes(len(point_ids), len(BOOTSTRAP_VIEWS)))
    text_chunks = _split_shuffled(
        current_text_records,
        _chunk_sizes(len(current_texts), len(BOOTSTRAP_VIEWS)),
        seed=SEED,
    )

    derived_records: list[dict[str, object]] = []
    summary_records: list[dict[str, object]] = []
    for view, point_chunk, text_chunk_records in zip(
        BOOTSTRAP_VIEWS, point_chunks, text_chunks, strict=True
    ):
        point_set = set(point_chunk)
        view_image_paths: list[str] = []
        view_mask_paths: list[str | None] = []
        for image_path, mask_path in zip(image_paths, segmentation_mask_paths, strict=True):
            point_id = Path(image_path).stem.split("_", maxsplit=1)[0]
            if point_id in point_set:
                view_image_paths.append(image_path)
                view_mask_paths.append(mask_path)

        text_chunk = [record.text for record in text_chunk_records]
        if not view_image_paths:
            raise ValueError(f"Bootstrap view '{view.suffix}' ended up with no images.")
        if not text_chunk:
            raise ValueError(f"Bootstrap view '{view.suffix}' ended up with no current texts.")

        derived_region_id = f"{base_region_id}_{view.suffix}"
        current_sentiment_counts = Counter(
            current_sentiment_by_text[record.record_id] for record in text_chunk_records
        )
        current_sentiment_label = _aggregate_integer_label(
            [current_sentiment_by_text[record.record_id] for record in text_chunk_records]
        )
        view_metadata = {
            **metadata,
            "parent_region_id": base_region_id,
            "bootstrap_view": view.suffix,
            "bootstrap_seed": SEED,
            "bootstrap_point_ids": point_chunk,
            "bootstrap_image_count": len(view_image_paths),
            "bootstrap_current_text_count": len(text_chunk),
            "bootstrap_sentiment_label_counts": {
                str(label): count for label, count in sorted(current_sentiment_counts.items())
            },
        }
        view_targets = {
            **base_targets,
            "sentiment_label": current_sentiment_label,
        }
        derived_records.append(
            {
                "region_id": derived_region_id,
                "split": view.split,
                "image_paths": view_image_paths,
                "segmentation_mask_paths": view_mask_paths,
                "current_texts": text_chunk,
                "historical_texts": historical_texts,
                "identity_texts": identity_texts,
                "metadata": view_metadata,
                "targets": view_targets,
            }
        )
        summary_records.append(
            {
                "region_id": derived_region_id,
                "split": view.split,
                "point_ids": point_chunk,
                "image_count": len(view_image_paths),
                "current_text_count": len(text_chunk),
                "sentiment_label": current_sentiment_label,
                "sentiment_label_counts": {
                    str(label): count for label, count in sorted(current_sentiment_counts.items())
                },
                "historical_sentiment_label": historical_sentiment_label,
                "lost_space_label": lost_space_label,
            }
        )

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in derived_records) + "\n",
        encoding="utf-8",
    )
    full_manifest_copy.write_text(
        json.dumps(
            {
                **payload,
                "image_paths": image_paths,
                "segmentation_mask_paths": segmentation_mask_paths,
                "current_texts": current_texts,
                "targets": {
                    **base_targets,
                    "sentiment_label": _aggregate_integer_label(
                        [current_sentiment_by_text[record.record_id] for record in current_text_records]
                    ),
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(
            {
                "source_manifest": str(source_manifest.relative_to(REPO_ROOT)),
                "derived_record_count": len(derived_records),
                "views": summary_records,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    build_bootstrap_manifest()
