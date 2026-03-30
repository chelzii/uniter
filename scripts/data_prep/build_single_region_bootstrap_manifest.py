from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_MANIFEST = REPO_ROOT / "datasets" / "data_workspace_cleaned" / "regions.jsonl"
DEFAULT_OUTPUT_MANIFEST = REPO_ROOT / "data" / "regions.jsonl"
DEFAULT_FULL_MANIFEST_COPY = REPO_ROOT / "data" / "regions.single_region.full.jsonl"
DEFAULT_SUMMARY_PATH = REPO_ROOT / "data" / "regions.bootstrap_summary.json"
SEED = 42


@dataclass(frozen=True)
class BootstrapView:
    suffix: str
    split: str


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


def build_bootstrap_manifest(
    *,
    source_manifest: Path = DEFAULT_SOURCE_MANIFEST,
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
    current_texts = [str(text) for text in payload["current_texts"]]
    historical_texts = [str(text) for text in payload.get("historical_texts", [])]
    identity_texts = [str(text) for text in payload.get("identity_texts", [])]
    metadata = dict(payload.get("metadata", {}))
    targets = dict(payload.get("targets", {}))

    point_ids = _group_point_ids(image_paths)
    point_chunks = _split_ordered(point_ids, _chunk_sizes(len(point_ids), len(BOOTSTRAP_VIEWS)))
    text_chunks = _split_shuffled(
        current_texts,
        _chunk_sizes(len(current_texts), len(BOOTSTRAP_VIEWS)),
        seed=SEED,
    )

    derived_records: list[dict[str, object]] = []
    summary_records: list[dict[str, object]] = []
    for view, point_chunk, text_chunk in zip(BOOTSTRAP_VIEWS, point_chunks, text_chunks, strict=True):
        point_set = set(point_chunk)
        view_image_paths: list[str] = []
        view_mask_paths: list[str | None] = []
        for image_path, mask_path in zip(image_paths, segmentation_mask_paths, strict=True):
            point_id = Path(image_path).stem.split("_", maxsplit=1)[0]
            if point_id in point_set:
                view_image_paths.append(image_path)
                view_mask_paths.append(mask_path)

        if not view_image_paths:
            raise ValueError(f"Bootstrap view '{view.suffix}' ended up with no images.")
        if not text_chunk:
            raise ValueError(f"Bootstrap view '{view.suffix}' ended up with no current texts.")

        derived_region_id = f"{base_region_id}_{view.suffix}"
        view_metadata = {
            **metadata,
            "parent_region_id": base_region_id,
            "bootstrap_view": view.suffix,
            "bootstrap_seed": SEED,
            "bootstrap_point_ids": point_chunk,
            "bootstrap_image_count": len(view_image_paths),
            "bootstrap_current_text_count": len(text_chunk),
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
                "targets": targets,
            }
        )
        summary_records.append(
            {
                "region_id": derived_region_id,
                "split": view.split,
                "point_ids": point_chunk,
                "image_count": len(view_image_paths),
                "current_text_count": len(text_chunk),
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
