from __future__ import annotations

import json
from pathlib import Path

from uniter.reporting.experiments import summarize_experiments_root


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
