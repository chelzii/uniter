from __future__ import annotations

import json
import os
from pathlib import Path

import uniter.reporting.context as reporting_context
from uniter.reporting.experiments import summarize_experiments_root, write_artifact_index
from uniter.reporting.context import build_region_context
from uniter.reporting.region_results import export_region_results_table


def test_summarize_experiments_root_collects_training_and_evaluation_files(
    tmp_path: Path,
) -> None:
    experiment_dir = tmp_path / "run_a"
    summaries_dir = experiment_dir / "summaries"
    evaluations_dir = experiment_dir / "evaluations"
    summaries_dir.mkdir(parents=True)
    evaluations_dir.mkdir(parents=True)
    (experiment_dir / "training_state.json").write_text(
        json.dumps({"status": "completed", "current_epoch": 2}),
        encoding="utf-8",
    )
    (experiment_dir / "best_checkpoint.json").write_text(
        json.dumps({"best_epoch": 2, "best_metric": 0.8}),
        encoding="utf-8",
    )
    (summaries_dir / "epoch_002.json").write_text(
        json.dumps(
            {
                "epoch": 2,
                "train": {"loss": 0.9, "ifi": 0.3, "mdi": 0.2},
                "val": {"loss": 0.8, "ifi": 0.25, "mdi": 0.18},
            }
        ),
        encoding="utf-8",
    )
    (evaluations_dir / "evaluation_val.json").write_text(
        json.dumps({"summary": {"region_count": 4}}),
        encoding="utf-8",
    )

    payload = summarize_experiments_root(tmp_path)

    assert payload["experiment_count"] == 1
    summary = payload["experiments"][0]
    assert summary["experiment_name"] == "run_a"
    assert summary["latest_metrics"]["val_loss"] == 0.8
    assert summary["evaluations"]["val"]["region_count"] == 4


def test_write_artifact_index_and_region_results_export(tmp_path: Path) -> None:
    experiment_dir = tmp_path / "run_a"
    (experiment_dir / "exports").mkdir(parents=True)
    (experiment_dir / "calibration").mkdir(parents=True)
    (experiment_dir / "exports" / "region_metrics_all.csv").write_text(
        "region_id\nregion_a\n",
        encoding="utf-8",
    )
    (experiment_dir / "calibration" / "thresholds_train.json").write_text(
        "{}",
        encoding="utf-8",
    )
    artifact_index_path = write_artifact_index(experiment_dir)
    assert artifact_index_path.exists()
    artifact_index = json.loads(artifact_index_path.read_text(encoding="utf-8"))
    assert artifact_index["groups"]["calibration"] == ["calibration/thresholds_train.json"]

    results_csv, results_md = export_region_results_table(
        rows=[
            {
                "region_id": "region_a",
                "split": "test",
                "final_level": "moderate",
                "risk_score": 0.75,
                "mdi_mode": "embedding_drift",
                "data_sources": json.dumps(["街景图", "微博"], ensure_ascii=False),
            }
        ],
        output_path=experiment_dir / "exports" / "region_results_all.csv",
        markdown_path=experiment_dir / "exports" / "region_results_all.md",
    )
    assert results_csv.exists()
    assert results_md is not None and results_md.exists()
    assert "embedding_drift" in results_md.read_text(encoding="utf-8")


def test_build_region_context_uses_split_scoped_text_and_image_metadata(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        reporting_context,
        "_metadata_by_region",
        lambda: {
            "region_parent": {
                "region_name": "Parent Region",
                "image_count": 90,
                "current_text_count": 161,
                "raw_current_text_count": 1254,
            }
        },
    )
    monkeypatch.setattr(
        reporting_context,
        "_current_text_rows_by_region",
        lambda: {
            "region_parent": [
                {
                    "cleaned_text": "text a",
                    "text": "text a",
                    "source_platform": "weibo",
                    "keep_for_training": "true",
                },
                {
                    "cleaned_text": "text b",
                    "text": "text b",
                    "source_platform": "xiaohongshu",
                    "keep_for_training": "true",
                },
                {
                    "cleaned_text": "text c",
                    "text": "text c",
                    "source_platform": "xiaohongshu",
                    "keep_for_training": "true",
                },
            ]
        },
    )
    monkeypatch.setattr(
        reporting_context,
        "_image_rows_by_region",
        lambda: {
            "region_parent": [
                {
                    "point_id": "p001",
                    "file_name": "p001_north.png",
                    "image_type": "street",
                    "view_direction": "north",
                    "lon": "1",
                    "lat": "2",
                    "source_platform": "street_view",
                },
                {
                    "point_id": "p001",
                    "file_name": "p001_satellite.tif",
                    "image_type": "satellite",
                    "view_direction": "",
                    "lon": "1",
                    "lat": "2",
                    "source_platform": "satellite",
                },
                {
                    "point_id": "p002",
                    "file_name": "p002_north.png",
                    "image_type": "street",
                    "view_direction": "north",
                    "lon": "3",
                    "lat": "4",
                    "source_platform": "street_view",
                },
            ]
        },
    )

    context = build_region_context(
        region_id="region_child",
        metadata={
            "parent_region_id": "region_parent",
            "region_name": "Child Region",
            "bootstrap_point_ids": ["p001"],
            "image_count": 2,
            "selected_image_count": 1,
            "current_text_count": 2,
            "historical_text_count": 1,
            "identity_text_count": 1,
            "_selected_current_texts": ["text a", "text b"],
            "bad_image_skip_count": 1,
        },
    )

    assert context["image_count"] == 2
    assert context["used_image_count"] == 1
    assert context["parent_image_count"] == 90
    assert context["street_image_count"] == 1
    assert context["satellite_image_count"] == 1
    assert context["point_count"] == 1
    assert context["current_text_count"] == 2
    assert context["cleaned_current_text_count"] == 2
    assert context["parent_cleaned_current_text_count"] == 161
    assert context["current_text_source_platforms"] == {"weibo": 1, "xiaohongshu": 1}
    assert context["parent_current_text_source_platforms"] == {
        "xiaohongshu": 2,
        "weibo": 1,
    }


def test_build_region_context_resolves_workspace_from_manifest(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "images" / "region_parent").mkdir(parents=True)
    (workspace / "current_texts").mkdir(parents=True)
    (workspace / "metadata").mkdir(parents=True)
    (workspace / "images" / "image_index.csv").write_text(
        "\n".join(
            [
                "region_id,point_id,file_name,image_type,view_direction,lon,lat,source_platform",
                "region_parent,p001,p001_north.png,street,north,1,2,street_view",
            ]
        ),
        encoding="utf-8",
    )
    (workspace / "current_texts" / "current_text_index_cleaned.csv").write_text(
        "\n".join(
            [
                "region_id,source_platform,text,cleaned_text,keep_for_training",
                "region_parent,weibo,text a,text a,true",
            ]
        ),
        encoding="utf-8",
    )
    (workspace / "metadata" / "region_parent.json").write_text(
        json.dumps(
            {
                "region_id": "region_parent",
                "region_name": "Parent Region",
                "image_count": 10,
                "current_text_count": 8,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    image_path = workspace / "images" / "region_parent" / "p001_north.png"
    image_path.write_bytes(b"placeholder")
    manifest_path = tmp_path / "data" / "regions.jsonl"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            {
                "region_id": "region_child",
                "split": "train",
                "image_paths": [os.path.relpath(image_path, start=manifest_path.parent)],
                "current_texts": ["text a"],
                "historical_texts": [],
                "identity_texts": [],
                "metadata": {"parent_region_id": "region_parent"},
                "targets": {
                    "lost_space_label": None,
                    "sentiment_label": None,
                    "historical_sentiment_label": None,
                    "ifi": None,
                    "mdi": None,
                    "iai": None,
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    context = build_region_context(
        region_id="region_child",
        metadata={"parent_region_id": "region_parent", "_selected_current_texts": ["text a"]},
        manifest_path=manifest_path,
    )

    assert context["region_name"] == "Parent Region"
    assert context["image_count"] == 10
    assert context["current_text_source_platforms"] == {"weibo": 1}
