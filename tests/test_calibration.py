from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from uniter.config import AppConfig, CalibrationConfig
from uniter.inference.calibration import (
    ThresholdCalibrator,
    apply_calibrated_thresholds,
    fit_thresholds,
    fit_thresholds_or_default,
)


def test_fit_thresholds_uses_requested_quantiles() -> None:
    thresholds = fit_thresholds(
        [0.1, 0.2, 0.3, 0.4, 0.5],
        calibration=CalibrationConfig(
            light_quantile=0.25,
            moderate_quantile=0.50,
            severe_quantile=0.75,
            min_samples=3,
        ),
    )

    assert thresholds.light == pytest.approx(0.2)
    assert thresholds.moderate == pytest.approx(0.3)
    assert thresholds.severe == pytest.approx(0.4)


def test_fit_thresholds_rejects_small_sample_sets() -> None:
    with pytest.raises(ValueError, match="Not enough samples"):
        fit_thresholds(
            [0.1, 0.2],
            calibration=CalibrationConfig(min_samples=3),
        )


def test_fit_thresholds_or_default_falls_back_when_samples_are_insufficient() -> None:
    fallback = AppConfig().judgement.iai
    thresholds, sample_count, was_calibrated, mode, effective_weight, uncertainty = fit_thresholds_or_default(
        [0.1, None],
        calibration=CalibrationConfig(min_samples=3),
        fallback=fallback,
    )

    assert sample_count == 1
    assert was_calibrated is True
    assert mode == "bootstrap_shrinkage"
    assert effective_weight >= 0.35
    assert thresholds != fallback
    assert uncertainty["light"] == pytest.approx(0.0)


def test_threshold_calibrator_exports_iai_thresholds(monkeypatch, tmp_path: Path) -> None:
    class _FakeRuntime:
        def __init__(self, config: AppConfig) -> None:
            self.config = config

        def iterate_region_batches(self, *, split: str, checkpoint_path=None, allow_random_init: bool = False):
            del checkpoint_path, allow_random_init
            assert split == "train"
            yield None, SimpleNamespace(
                ifi=torch.tensor([0.10, 0.20, 0.30]),
                mdi=torch.tensor([0.20, 0.30, 0.40]),
                iai=torch.tensor([0.30, 0.40, 0.50]),
                alignment_gap=torch.tensor([0.15, 0.25, 0.35]),
            )

        def resolve_records(self, split: str):
            assert split == "train"
            return ["a", "b", "c"]

    import uniter.inference.calibration as calibration_module

    monkeypatch.setattr(calibration_module, "InferenceRuntime", _FakeRuntime)

    config = AppConfig()
    config.experiment.output_dir = str(tmp_path / "run")
    config.calibration.min_samples = 3
    calibrator = ThresholdCalibrator(config)
    output_path = calibrator.calibrate(
        checkpoint_path=None,
        split="train",
        output_path=None,
        allow_random_init=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["thresholds"]["iai"]["light"] == pytest.approx(0.4)
    assert payload["calibrated_metrics"]["iai"] is True
    assert payload["metric_sample_counts"]["iai"] == 3
    assert payload["calibration_modes"]["iai"] == "quantile"
    assert payload["effective_weights"]["iai"] == pytest.approx(1.0)


def test_threshold_calibrator_applies_label_guided_adjustment_for_single_region_bootstrap(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class _Record:
        def __init__(self, region_id: str, bootstrap_view: str) -> None:
            self.region_id = region_id
            self.metadata = {
                "parent_region_id": "parent_a",
                "bootstrap_view": bootstrap_view,
            }

    class _FakeBatch:
        def __init__(self, labels: list[int]) -> None:
            self.targets = {"lost_space_label": torch.tensor(labels, dtype=torch.long)}

    class _FakeRuntime:
        def __init__(self, config: AppConfig) -> None:
            self.config = config
            self._records = [_Record("region_a", "train_a"), _Record("region_b", "train_b")]

        def iterate_region_batches(self, *, split: str, checkpoint_path=None, allow_random_init: bool = False):
            del checkpoint_path, allow_random_init
            assert split == "train"
            yield _FakeBatch([2, 2]), SimpleNamespace(
                ifi=torch.tensor([0.18, 0.21]),
                mdi=torch.tensor([0.08, 0.09]),
                iai=torch.tensor([0.73, 0.73]),
                alignment_gap=torch.tensor([0.91, 0.92]),
            )

        def resolve_records(self, split: str):
            assert split == "train"
            return list(self._records)

    import uniter.inference.calibration as calibration_module

    monkeypatch.setattr(calibration_module, "InferenceRuntime", _FakeRuntime)

    config = AppConfig()
    config.experiment.output_dir = str(tmp_path / "run")
    config.calibration.min_samples = 10
    config.judgement.rule_mode = "thesis"
    config.judgement.use_alignment_gap = False
    calibrator = ThresholdCalibrator(config)
    output_path = calibrator.calibrate(
        checkpoint_path=None,
        split="train",
        output_path=None,
        allow_random_init=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["bootstrap_context"]["single_region_bootstrap"] is True
    assert payload["label_guided_adjustment"]["applied"] is True
    assert payload["judgement_preview"]["moderate"] == 2
    assert payload["judgement_preview"]["severe"] == 0


def test_apply_calibrated_thresholds_updates_runtime_config(tmp_path: Path) -> None:
    threshold_path = tmp_path / "thresholds_train.json"
    threshold_path.write_text(
        json.dumps(
            {
                "thresholds": {
                    "ifi": {"light": 0.11, "moderate": 0.22, "severe": 0.33},
                    "mdi": {"light": 0.12, "moderate": 0.23, "severe": 0.34},
                    "iai": {"light": 0.13, "moderate": 0.24, "severe": 0.35},
                    "alignment_gap": {
                        "light": 0.14,
                        "moderate": 0.25,
                        "severe": 0.36,
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    config = AppConfig()
    config.inference.thresholds_path = str(threshold_path)

    resolved = apply_calibrated_thresholds(config)

    assert resolved == threshold_path.resolve()
    assert config.judgement.ifi.light == pytest.approx(0.11)
    assert config.judgement.mdi.moderate == pytest.approx(0.23)
    assert config.judgement.iai.severe == pytest.approx(0.35)
    assert config.judgement.alignment_gap.light == pytest.approx(0.14)
