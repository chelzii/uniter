from __future__ import annotations

import sys

import uniter.cli as cli


def test_main_dispatches_calibrate_thresholds_without_threshold_argument(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["uniter-base", "calibrate-thresholds", "--config", "configs/base.yaml"],
    )

    captured: dict[str, object] = {}

    def _fake_command(
        config_path: str,
        *,
        output_path: str | None,
        checkpoint_path: str | None,
        split: str,
        allow_random_init: bool,
    ) -> int:
        captured.update(
            {
                "config_path": config_path,
                "output_path": output_path,
                "checkpoint_path": checkpoint_path,
                "split": split,
                "allow_random_init": allow_random_init,
            }
        )
        return 0

    monkeypatch.setattr(cli, "command_calibrate_thresholds", _fake_command)

    assert cli.main() == 0
    assert captured == {
        "config_path": "configs/base.yaml",
        "output_path": None,
        "checkpoint_path": None,
        "split": "train",
        "allow_random_init": False,
    }


def test_main_dispatches_export_region_metrics_thresholds_argument(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "uniter-base",
            "export-region-metrics",
            "--config",
            "configs/base.yaml",
            "--thresholds",
            "runs/train/example/calibration/thresholds_train.json",
        ],
    )

    captured: dict[str, object] = {}

    def _fake_command(
        config_path: str,
        *,
        output_path: str | None,
        checkpoint_path: str | None,
        split: str,
        allow_random_init: bool,
        thresholds_path: str | None,
    ) -> int:
        captured.update(
            {
                "config_path": config_path,
                "output_path": output_path,
                "checkpoint_path": checkpoint_path,
                "split": split,
                "allow_random_init": allow_random_init,
                "thresholds_path": thresholds_path,
            }
        )
        return 0

    monkeypatch.setattr(cli, "command_export_region_metrics", _fake_command)

    assert cli.main() == 0
    assert captured == {
        "config_path": "configs/base.yaml",
        "output_path": None,
        "checkpoint_path": None,
        "split": "all",
        "allow_random_init": False,
        "thresholds_path": "runs/train/example/calibration/thresholds_train.json",
    }
