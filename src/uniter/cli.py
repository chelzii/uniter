from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from uniter.config import DEFAULT_CONFIG, load_config
from uniter.data.manifest import validate_manifest
from uniter.data.tools import build_manifest_from_directories, summarize_manifest
from uniter.utils.logging import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="uniter-base",
        description="UNITER-inspired base code for lost-space multimodal analysis.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    describe = subparsers.add_parser("describe-config", help="Print a resolved config file.")
    describe.add_argument("config", nargs="?", default=None, help="Path to a TOML config file.")

    validate = subparsers.add_parser("validate-manifest", help="Validate a region manifest.")
    validate.add_argument("manifest", help="Path to the JSONL manifest.")
    validate.add_argument("--image-root", default=None, help="Optional shared image root.")
    validate.add_argument(
        "--check-files",
        action="store_true",
        help="Also verify that referenced image files exist on disk.",
    )

    build_manifest = subparsers.add_parser(
        "build-manifest",
        help="Build a manifest from region-organized image/text directories.",
    )
    build_manifest.add_argument("--output", required=True, help="Output JSONL manifest path.")
    build_manifest.add_argument(
        "--image-dir",
        default=None,
        help="Directory containing one image subdirectory per region_id.",
    )
    build_manifest.add_argument(
        "--current-text-dir",
        default=None,
        help="Directory containing one current text file per region_id.",
    )
    build_manifest.add_argument(
        "--historical-text-dir",
        default=None,
        help="Optional directory containing one historical text file per region_id.",
    )
    build_manifest.add_argument(
        "--identity-text-dir",
        default=None,
        help="Optional directory containing one identity text file per region_id.",
    )
    build_manifest.add_argument(
        "--segmentation-mask-dir",
        default=None,
        help="Optional directory containing one segmentation-mask subdirectory per region_id.",
    )
    build_manifest.add_argument(
        "--metadata-dir",
        default=None,
        help="Optional directory containing one metadata JSON file per region_id.",
    )
    build_manifest.add_argument(
        "--split-map",
        default=None,
        help="Optional JSON file mapping region_id to split.",
    )
    build_manifest.add_argument(
        "--default-split",
        default="train",
        choices=["train", "val", "test"],
        help="Fallback split when a region is absent from --split-map.",
    )
    build_manifest.add_argument(
        "--strict",
        action="store_true",
        help="Fail instead of skipping incomplete regions.",
    )

    summarize = subparsers.add_parser(
        "summarize-manifest",
        help="Export a manifest summary as JSON.",
    )
    summarize.add_argument("manifest", help="Path to the JSONL manifest.")
    summarize.add_argument(
        "--output",
        default=None,
        help="Optional summary JSON path. Defaults next to the manifest.",
    )

    train = subparsers.add_parser("train", help="Train the multimodal base model.")
    train.add_argument("--config", required=True, help="Path to a TOML config file.")
    train.add_argument(
        "--resume-from",
        default=None,
        help="Optional checkpoint path to resume training from.",
    )

    evaluate = subparsers.add_parser(
        "evaluate",
        help="Evaluate a checkpoint and export split-level summary metrics as JSON.",
    )
    evaluate.add_argument("--config", required=True, help="Path to a TOML config file.")
    evaluate.add_argument(
        "--output",
        default=None,
        help="Optional evaluation JSON path. Defaults to <output_dir>/evaluations/...",
    )
    evaluate.add_argument(
        "--checkpoint",
        default=None,
        help="Optional checkpoint path created by the training command.",
    )
    evaluate.add_argument(
        "--split",
        default="val",
        choices=["train", "val", "test", "all"],
        help="Which split to evaluate.",
    )
    evaluate.add_argument(
        "--allow-random-init",
        action="store_true",
        help="Allow inference without a checkpoint. Intended only for debugging.",
    )

    calibrate = subparsers.add_parser(
        "calibrate-thresholds",
        help="Estimate rule thresholds from a split and export them as JSON.",
    )
    calibrate.add_argument("--config", required=True, help="Path to a TOML config file.")
    calibrate.add_argument(
        "--output",
        default=None,
        help="Optional calibration JSON path. Defaults to <output_dir>/calibration/...",
    )
    calibrate.add_argument(
        "--checkpoint",
        default=None,
        help="Optional checkpoint path created by the training command.",
    )
    calibrate.add_argument(
        "--split",
        default="train",
        choices=["train", "val", "test", "all"],
        help="Which split to use for threshold calibration.",
    )
    calibrate.add_argument(
        "--allow-random-init",
        action="store_true",
        help="Allow calibration without a checkpoint. Intended only for debugging.",
    )

    export = subparsers.add_parser(
        "export-region-metrics",
        help="Run region-level inference and export IFI/MDI/sentiment metrics as CSV.",
    )
    export.add_argument("--config", required=True, help="Path to a TOML config file.")
    export.add_argument(
        "--output",
        default=None,
        help=(
            "Optional CSV output path. Defaults to "
            "<output_dir>/exports/region_metrics_<split>.csv."
        ),
    )
    export.add_argument(
        "--checkpoint",
        default=None,
        help="Optional checkpoint path created by the training command.",
    )
    export.add_argument(
        "--split",
        default="all",
        choices=["train", "val", "test", "all"],
        help="Which split to export.",
    )
    export.add_argument(
        "--allow-random-init",
        action="store_true",
        help="Allow export without a checkpoint. Intended only for debugging.",
    )

    visualize = subparsers.add_parser(
        "export-visualizations",
        help="Export region reports, segmentation overlays, and training curves.",
    )
    visualize.add_argument("--config", required=True, help="Path to a TOML config file.")
    visualize.add_argument(
        "--output-dir",
        default=None,
        help="Optional visualization directory. Defaults to <output_dir>/visualizations/...",
    )
    visualize.add_argument(
        "--checkpoint",
        default=None,
        help="Optional checkpoint path created by the training command.",
    )
    visualize.add_argument(
        "--split",
        default="all",
        choices=["train", "val", "test", "all"],
        help="Which split to visualize.",
    )
    visualize.add_argument(
        "--allow-random-init",
        action="store_true",
        help="Allow visualization without a checkpoint. Intended only for debugging.",
    )

    summarize_experiments = subparsers.add_parser(
        "summarize-experiments",
        help="Aggregate training state and evaluation summaries under an outputs root.",
    )
    summarize_experiments.add_argument(
        "--root",
        required=True,
        help="Root directory containing experiment output folders.",
    )
    summarize_experiments.add_argument(
        "--output",
        required=True,
        help="JSON output path for the aggregated summary.",
    )

    return parser


def describe_config(config_path: str | None) -> int:
    payload = asdict(DEFAULT_CONFIG if config_path is None else load_config(config_path))
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def command_validate_manifest(manifest: str, image_root: str | None, check_files: bool) -> int:
    errors = validate_manifest(
        manifest_path=Path(manifest),
        image_root=Path(image_root) if image_root else None,
        check_files=check_files,
    )
    if errors:
        for error in errors:
            print(error)
        return 1

    print("Manifest schema is valid.")
    return 0


def command_build_manifest(
    *,
    output_path: str,
    image_dir: str | None,
    current_text_dir: str | None,
    historical_text_dir: str | None,
    identity_text_dir: str | None,
    segmentation_mask_dir: str | None,
    metadata_dir: str | None,
    split_map_path: str | None,
    default_split: str,
    strict: bool,
) -> int:
    manifest_path, summary = build_manifest_from_directories(
        output_path=output_path,
        image_dir=image_dir,
        current_text_dir=current_text_dir,
        historical_text_dir=historical_text_dir,
        identity_text_dir=identity_text_dir,
        segmentation_mask_dir=segmentation_mask_dir,
        metadata_dir=metadata_dir,
        split_map_path=split_map_path,
        default_split=default_split,
        strict=strict,
    )
    summary_path = manifest_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Built manifest at {manifest_path}")
    print(f"Exported manifest summary to {summary_path}")
    return 0


def command_summarize_manifest(manifest: str, output_path: str | None) -> int:
    summary = summarize_manifest(manifest)
    final_output_path = (
        Path(output_path).resolve()
        if output_path is not None
        else Path(manifest).resolve().with_suffix(".summary.json")
    )
    final_output_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Exported manifest summary to {final_output_path}")
    return 0


def command_train(config_path: str, *, resume_from: str | None) -> int:
    config = load_config(config_path)
    if resume_from is not None:
        config.training.resume_from = str(Path(resume_from).resolve())
    configure_logging(config.output_dir)

    from uniter.training.trainer import Trainer

    trainer = Trainer(config)
    trainer.train()
    return 0


def command_evaluate(
    config_path: str,
    *,
    output_path: str | None,
    checkpoint_path: str | None,
    split: str,
    allow_random_init: bool,
) -> int:
    config = load_config(config_path)
    configure_logging(config.output_dir)

    from uniter.inference.evaluator import RegionEvaluator

    evaluator = RegionEvaluator(config)
    final_output_path = evaluator.evaluate(
        output_path=output_path,
        checkpoint_path=checkpoint_path,
        split=split,
        allow_random_init=allow_random_init,
    )
    print(f"Exported evaluation summary to {final_output_path}")
    return 0


def command_calibrate_thresholds(
    config_path: str,
    *,
    output_path: str | None,
    checkpoint_path: str | None,
    split: str,
    allow_random_init: bool,
) -> int:
    config = load_config(config_path)
    configure_logging(config.output_dir)

    from uniter.inference.calibration import ThresholdCalibrator

    calibrator = ThresholdCalibrator(config)
    final_output_path = calibrator.calibrate(
        output_path=output_path,
        checkpoint_path=checkpoint_path,
        split=split,
        allow_random_init=allow_random_init,
    )
    print(f"Exported calibrated thresholds to {final_output_path}")
    return 0


def command_export_region_metrics(
    config_path: str,
    *,
    output_path: str | None,
    checkpoint_path: str | None,
    split: str,
    allow_random_init: bool,
) -> int:
    config = load_config(config_path)
    configure_logging(config.output_dir)

    from uniter.inference.exporter import RegionMetricExporter

    exporter = RegionMetricExporter(config)
    final_output_path = exporter.export(
        output_path=output_path,
        checkpoint_path=checkpoint_path,
        split=split,
        allow_random_init=allow_random_init,
    )
    print(f"Exported region metrics to {final_output_path}")
    return 0


def command_export_visualizations(
    config_path: str,
    *,
    output_dir: str | None,
    checkpoint_path: str | None,
    split: str,
    allow_random_init: bool,
) -> int:
    config = load_config(config_path)
    configure_logging(config.output_dir)

    from uniter.inference.visualization import VisualizationExporter

    exporter = VisualizationExporter(config)
    final_output_path = exporter.export(
        output_dir=output_dir,
        checkpoint_path=checkpoint_path,
        split=split,
        allow_random_init=allow_random_init,
    )
    print(f"Exported visualizations to {final_output_path}")
    return 0


def command_summarize_experiments(root_dir: str, output_path: str) -> int:
    from uniter.reporting.experiments import export_experiment_summary

    final_output_path = export_experiment_summary(
        root_dir=root_dir,
        output_path=output_path,
    )
    print(f"Exported experiment summary to {final_output_path}")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.command == "describe-config":
            return describe_config(args.config)
        if args.command == "validate-manifest":
            return command_validate_manifest(args.manifest, args.image_root, args.check_files)
        if args.command == "build-manifest":
            return command_build_manifest(
                output_path=args.output,
                image_dir=args.image_dir,
                current_text_dir=args.current_text_dir,
                historical_text_dir=args.historical_text_dir,
                identity_text_dir=args.identity_text_dir,
                segmentation_mask_dir=args.segmentation_mask_dir,
                metadata_dir=args.metadata_dir,
                split_map_path=args.split_map,
                default_split=args.default_split,
                strict=args.strict,
            )
        if args.command == "summarize-manifest":
            return command_summarize_manifest(args.manifest, args.output)
        if args.command == "train":
            return command_train(args.config, resume_from=args.resume_from)
        if args.command == "evaluate":
            return command_evaluate(
                args.config,
                output_path=args.output,
                checkpoint_path=args.checkpoint,
                split=args.split,
                allow_random_init=args.allow_random_init,
            )
        if args.command == "calibrate-thresholds":
            return command_calibrate_thresholds(
                args.config,
                output_path=args.output,
                checkpoint_path=args.checkpoint,
                split=args.split,
                allow_random_init=args.allow_random_init,
            )
        if args.command == "export-region-metrics":
            return command_export_region_metrics(
                args.config,
                output_path=args.output,
                checkpoint_path=args.checkpoint,
                split=args.split,
                allow_random_init=args.allow_random_init,
            )
        if args.command == "export-visualizations":
            return command_export_visualizations(
                args.config,
                output_dir=args.output_dir,
                checkpoint_path=args.checkpoint,
                split=args.split,
                allow_random_init=args.allow_random_init,
            )
        if args.command == "summarize-experiments":
            return command_summarize_experiments(args.root, args.output)
    except ModuleNotFoundError as exc:
        if exc.name in {"torch", "transformers"}:
            print(
                f"Missing required dependency '{exc.name}'. "
                "Install project dependencies first, for example with `uv sync`.",
                file=sys.stderr,
            )
            return 1
        raise
    except RuntimeError as exc:
        if "Missing required dependency" in str(exc):
            print(str(exc), file=sys.stderr)
            return 1
        raise

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
