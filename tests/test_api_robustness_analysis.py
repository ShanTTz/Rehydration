from __future__ import annotations

import numpy as np

from bdmtf.revision.api_robustness_analysis import (
    _fleiss_kappa,
    _share_interval,
)


def test_fleiss_kappa_is_one_for_unanimous_raters() -> None:
    values = np.asarray(
        [
            ["reply", "reply", "reply"],
            ["abstain", "abstain", "abstain"],
        ]
    )
    assert _fleiss_kappa(values) == 1.0


def test_share_interval_contains_observed_share() -> None:
    values = np.asarray([True, True, False, True, False])
    result = _share_interval(
        values,
        bootstrap_samples=500,
        rng=np.random.default_rng(30371),
    )
    assert result["mean"] == 0.6
    assert result["ci_low"] <= result["mean"] <= result["ci_high"]
