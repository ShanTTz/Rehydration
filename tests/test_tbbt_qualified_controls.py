from __future__ import annotations

import numpy as np
import pandas as pd

from bdmtf.revision.tbbt_qualified_controls import (
    _hac_mean_interval,
    _simplex_weights,
    _source_overlap_audit,
)


def test_simplex_weights_are_nonnegative_and_normalized() -> None:
    donors = np.column_stack(
        [
            np.linspace(0.0, 1.0, 40),
            np.linspace(1.0, 0.0, 40),
            np.sin(np.linspace(0.0, 2.0, 40)),
        ]
    )
    target = 0.7 * donors[:, 0] + 0.3 * donors[:, 1]
    weights = _simplex_weights(target, donors, 0.0)
    assert np.all(weights >= 0)
    assert np.isclose(weights.sum(), 1.0)
    assert np.mean((target - donors @ weights) ** 2) < 1e-8


def test_source_overlap_audit_accepts_identical_series() -> None:
    index = pd.date_range("2020-01-01", periods=75, freq="D", tz="UTC")
    values = pd.Series(np.linspace(10, 100, len(index)), index=index)
    relative = pd.Series(np.arange(-60, 15), index=index)
    audit = _source_overlap_audit(
        values,
        values,
        relative,
        pre_days=60,
        validation_days=14,
    )
    assert audit["source_train_correlation"] > 0.999
    assert audit["source_validation_correlation"] > 0.999
    assert audit["source_log_ratio_mad"] < 1e-10


def test_hac_interval_centers_on_sample_mean() -> None:
    estimate, low, high, standard_error = _hac_mean_interval(
        np.array([-0.2, -0.1, -0.3, -0.2, -0.1]),
        2,
    )
    assert np.isclose(estimate, -0.18)
    assert low <= estimate <= high
    assert standard_error >= 0
