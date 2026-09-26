from __future__ import annotations

import numpy as np
import pandas as pd

from bdmtf.reviewer_response_benchmark import benchmark_factorial_response


def test_interaction_model_beats_additive_on_heldout_blocks() -> None:
    rows = []
    splits = []
    for post_index in range(20):
        split = "train" if post_index < 12 else "test"
        post_id = f"p{post_index}"
        splits.append(
            {"community": "science", "post_id": post_id, "split": split}
        )
        for seed in range(2):
            baseline_volume = 100.0 + post_index + seed
            baseline_leaf = 10.0 + post_index * 0.01
            core_volume = baseline_volume * np.exp(0.5)
            ranking_volume = baseline_volume * np.exp(0.1)
            joint_volume = baseline_volume * np.exp(0.5 + 0.1 + 0.3)
            rows.append(
                {
                    "community": "science",
                    "post_id": post_id,
                    "seed": seed,
                    "comment_volume__baseline_best": baseline_volume,
                    "comment_volume__baseline_controversial": ranking_volume,
                    "comment_volume__toxic_best": core_volume,
                    "comment_volume__toxic_controversial": joint_volume,
                    "mean_leaf_depth__baseline_best": baseline_leaf,
                    "mean_leaf_depth__baseline_controversial": baseline_leaf - 0.2,
                    "mean_leaf_depth__toxic_best": baseline_leaf - 1.0,
                    "mean_leaf_depth__toxic_controversial": baseline_leaf - 2.0,
                }
            )
    summary, improvements, predictions, manifest = benchmark_factorial_response(
        pd.DataFrame(rows),
        pd.DataFrame(splits),
        bootstrap_samples=100,
        seed=11,
    )
    assert manifest["evaluation_split"] == "test"
    for metric in ("comment_volume", "mean_leaf_depth"):
        metric_summary = summary[summary["metric"] == metric].set_index("model")
        assert (
            metric_summary.loc["interaction_global", "mae"]
            < metric_summary.loc["additive_global", "mae"]
        )
    assert (improvements["mean_absolute_error_improvement"] > 0).all()
    assert len(predictions) == 8 * 2 * 2 * 7
