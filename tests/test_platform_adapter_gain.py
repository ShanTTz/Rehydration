from __future__ import annotations

import pandas as pd

from bdmtf.platform_adapter_gain import evaluate_adapter_gain


def test_adapter_gain_gate_supports_bounded_adaptation() -> None:
    rows = []
    for index in range(10):
        metric = f"m{index}"
        for model, distance in (
            ("platform_adapted_learned_bdmtf", 0.4),
            ("target_default_learned_bdmtf", 0.6 if index < 7 else 0.3),
            ("reddit_zero_shot_learned_bdmtf", 1.0 if index < 9 else 0.2),
            ("target_hawkes", 0.35),
        ):
            rows.append(
                {
                    "community": "topstories",
                    "metric": metric,
                    "model": model,
                    "normalized_wasserstein": distance,
                }
            )
    units, manifest = evaluate_adapter_gain(
        pd.DataFrame(rows),
        {
            "adapted_beats_target_default_metric_share_min": 0.6,
            "adapted_beats_reddit_zero_shot_metric_share_min": 0.8,
            "require_positive_median_gain_over_target_default": True,
            "require_positive_median_gain_over_reddit_zero_shot": True,
            "require_beating_every_classical_baseline": False,
        },
    )
    assert len(units) == 10
    assert manifest["supports_platform_adaptation"]
    assert manifest["adapted_beats_target_default_metric_share"] == 0.7
    assert manifest["adapted_beats_reddit_zero_shot_metric_share"] == 0.9
    assert manifest["adapted_beats_best_classical_metric_share"] == 0.0
