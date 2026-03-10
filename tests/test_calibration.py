from __future__ import annotations

import pytest

from uniter.config import CalibrationConfig
from uniter.inference.calibration import fit_thresholds


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
