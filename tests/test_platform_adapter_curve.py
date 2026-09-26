from __future__ import annotations

import pandas as pd

from bdmtf.platform_adapter_curve import (
    frame_with_validation_budget,
    nested_validation_subset,
    summarize_budget_curve,
)


def _platform_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "community": ["hn"] * 8,
            "post_id": [str(value) for value in range(8)],
            "created_at": pd.date_range(
                "2025-01-01", periods=8, freq="h", tz="UTC"
            ),
            "split": [
                "train",
                "train",
                "validation",
                "validation",
                "validation",
                "test",
                "test",
                "test",
            ],
        }
    )


def test_validation_budgets_are_exact_and_nested() -> None:
    frame = _platform_frame()
    small = nested_validation_subset(frame, 1)
    large = nested_validation_subset(frame, 3)
    assert len(small) == 1
    assert len(large) == 3
    assert set(small["post_id"]).issubset(set(large["post_id"]))


def test_budget_frame_never_removes_train_or_test() -> None:
    frame = _platform_frame()
    budget = frame_with_validation_budget(frame, 1)
    assert (budget["split"] == "train").sum() == 2
    assert (budget["split"] == "validation").sum() == 1
    assert (budget["split"] == "test").sum() == 3


def test_curve_summary_uses_prespecified_metrics() -> None:
    rows = []
    for model, values in {
        "platform_adapted_n0": (0.5, 0.9),
        "platform_adapted_n10": (0.3, 0.7),
        "target_default_learned_bdmtf": (0.6, 0.8),
        "reddit_zero_shot_learned_bdmtf": (1.0, 1.2),
    }.items():
        rows.extend(
            [
                {
                    "model": model,
                    "metric": "max_depth",
                    "normalized_wasserstein": values[0],
                },
                {
                    "model": model,
                    "metric": "size",
                    "normalized_wasserstein": values[1],
                },
            ]
        )
    curve = summarize_budget_curve(
        pd.DataFrame(rows), [0, 10], ["max_depth"]
    )
    assert list(curve["validation_budget"]) == [0, 10]
    assert curve.loc[1, "primary_metric_median_distance"] == 0.3
    assert curve.loc[1, "gain_over_reddit_zero_shot"] > 0
