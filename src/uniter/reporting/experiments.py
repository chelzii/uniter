from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any] | list[Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _latest_epoch_summary(summary_dir: Path) -> dict[str, Any] | None:
    candidates = sorted(summary_dir.glob("epoch_*.json"))
    if not candidates:
        return None
    payload = _read_json(candidates[-1])
    return payload if isinstance(payload, dict) else None


def _evaluation_summaries(evaluation_dir: Path) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    if not evaluation_dir.exists():
        return summaries
    for path in sorted(evaluation_dir.glob("evaluation_*.json")):
        payload = _read_json(path)
        if not isinstance(payload, dict):
            continue
        split = path.stem.replace("evaluation_", "", 1)
        summary = payload.get("summary")
        if isinstance(summary, dict):
            summaries[split] = summary
    return summaries


def summarize_experiment_directory(experiment_dir: str | Path) -> dict[str, Any]:
    root = Path(experiment_dir).resolve()
    training_state = _read_json(root / "training_state.json")
    best_checkpoint = _read_json(root / "best_checkpoint.json")
    latest_summary = _latest_epoch_summary(root / "summaries")
    evaluations = _evaluation_summaries(root / "evaluations")

    summary: dict[str, Any] = {
        "experiment_dir": str(root),
        "experiment_name": root.name,
        "training_state": training_state if isinstance(training_state, dict) else None,
        "best_checkpoint": best_checkpoint if isinstance(best_checkpoint, dict) else None,
        "latest_summary": latest_summary,
        "evaluations": evaluations,
    }

    if isinstance(latest_summary, dict):
        train = latest_summary.get("train")
        val = latest_summary.get("val")
        summary["latest_metrics"] = {
            "train_loss": train.get("loss") if isinstance(train, dict) else None,
            "val_loss": val.get("loss") if isinstance(val, dict) else None,
            "train_ifi": train.get("ifi") if isinstance(train, dict) else None,
            "val_ifi": val.get("ifi") if isinstance(val, dict) else None,
            "train_mdi": train.get("mdi") if isinstance(train, dict) else None,
            "val_mdi": val.get("mdi") if isinstance(val, dict) else None,
            "train_lost_space_accuracy": (
                train.get("lost_space_accuracy") if isinstance(train, dict) else None
            ),
            "val_lost_space_accuracy": (
                val.get("lost_space_accuracy") if isinstance(val, dict) else None
            ),
        }
    else:
        summary["latest_metrics"] = {}

    return summary


def build_artifact_index(experiment_dir: str | Path) -> dict[str, Any]:
    root = Path(experiment_dir).resolve()
    groups = {
        "checkpoints": sorted(
            str(path.relative_to(root))
            for path in (root / "checkpoints").glob("*.pt")
        )
        if (root / "checkpoints").exists()
        else [],
        "summaries": sorted(
            str(path.relative_to(root))
            for path in (root / "summaries").glob("*.json")
        )
        if (root / "summaries").exists()
        else [],
        "evaluations": sorted(
            str(path.relative_to(root))
            for path in (root / "evaluations").glob("*.json")
        )
        if (root / "evaluations").exists()
        else [],
        "calibration": sorted(
            str(path.relative_to(root))
            for path in (root / "calibration").glob("*.json")
        )
        if (root / "calibration").exists()
        else [],
        "exports": sorted(
            str(path.relative_to(root))
            for path in (root / "exports").glob("*")
            if path.is_file()
        )
        if (root / "exports").exists()
        else [],
        "visualizations": sorted(
            str(path.relative_to(root))
            for path in (root / "visualizations").rglob("*")
            if path.is_file()
        )
        if (root / "visualizations").exists()
        else [],
    }
    return {
        "experiment_dir": str(root),
        "resolved_config": (
            "resolved_config.yaml" if (root / "resolved_config.yaml").exists() else None
        ),
        "groups": groups,
    }


def write_artifact_index(experiment_dir: str | Path) -> Path:
    root = Path(experiment_dir).resolve()
    payload = build_artifact_index(root)
    destination = root / "artifacts_index.json"
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return destination


def summarize_experiments_root(root_dir: str | Path) -> dict[str, Any]:
    root = Path(root_dir).resolve()
    if not root.exists():
        raise RuntimeError(f"Experiment root does not exist: {root}")
    experiments = [
        summarize_experiment_directory(path)
        for path in sorted(root.iterdir())
        if path.is_dir()
    ]
    return {
        "root_dir": str(root),
        "experiment_count": len(experiments),
        "experiments": experiments,
    }


def export_experiment_summary(
    *,
    root_dir: str | Path,
    output_path: str | Path,
) -> Path:
    payload = summarize_experiments_root(root_dir)
    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return destination
