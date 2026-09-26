from __future__ import annotations

from pathlib import Path

import pandas as pd

from bdmtf.revision.live_generation_identification import (
    PROTOCOL_FROZEN,
    PROTOCOL_LIVE,
)
from bdmtf.revision.multithread_identification import (
    select_stratified_test_posts,
    summarize_multithread_dispersion,
)


def test_quantile_selection_is_unique_and_deterministic(tmp_path: Path) -> None:
    social_root = tmp_path / "social"
    community_dir = social_root / "science_data"
    community_dir.mkdir(parents=True)
    posts = pd.DataFrame(
        [
            {
                "post_id": f"p{index:02d}",
                "title": f"Thread {index}",
                "full_text": "Held-out content",
                "is_viral": int(index >= 5),
            }
            for index in range(10)
        ]
    )
    posts.to_csv(community_dir / "posts_features_science.csv", index=False)
    splits = pd.DataFrame(
        [
            {"community": "science", "post_id": f"p{index:02d}", "split": "test"}
            for index in range(10)
        ]
    )
    metrics = pd.DataFrame(
        [
            {
                "community": "science",
                "post_id": f"p{index:02d}",
                "split": "test",
                "size": 10 * (index + 1),
                "is_viral": int(index >= 5),
            }
            for index in range(10)
        ]
    )
    splits_path = tmp_path / "splits.csv"
    metrics_path = tmp_path / "metrics.csv"
    splits.to_csv(splits_path, index=False)
    metrics.to_csv(metrics_path, index=False)

    kwargs = {
        "social_root": social_root,
        "splits_path": splits_path,
        "metrics_path": metrics_path,
        "communities": ["science"],
        "quantiles": [0.1, 0.3, 0.5, 0.7, 0.9],
    }
    first = select_stratified_test_posts(**kwargs)
    second = select_stratified_test_posts(**kwargs)
    first_ids = [item["post"]["post_id"] for item in first]
    second_ids = [item["post"]["post_id"] for item in second]

    assert first_ids == second_ids
    assert len(first_ids) == 5
    assert len(set(first_ids)) == 5


def test_multithread_summary_counts_reduced_dispersion() -> None:
    rows = []
    for thread in range(5):
        live = [-2.0, -1.0, 0.0, 1.0, 2.0]
        frozen = [-0.2, -0.1, 0.0, 0.1, 0.2]
        if thread == 4:
            live, frozen = frozen, live
        for repeat, (live_value, frozen_value) in enumerate(
            zip(live, frozen, strict=True)
        ):
            rows.extend(
                [
                    {
                        "community": "science",
                        "post_id": f"p{thread}",
                        "protocol": PROTOCOL_LIVE,
                        "repeat": repeat,
                        "log_volume_effect": live_value,
                    },
                    {
                        "community": "science",
                        "post_id": f"p{thread}",
                        "protocol": PROTOCOL_FROZEN,
                        "repeat": repeat,
                        "log_volume_effect": frozen_value,
                    },
                ]
            )

    thread_summary, aggregate = summarize_multithread_dispersion(
        pd.DataFrame(rows), bootstrap_samples=300, seed=7
    )

    assert len(thread_summary) == 5
    assert aggregate["threads_with_reduced_dispersion"] == 4
    assert aggregate["median_variance_ratio_live_over_frozen"] > 10
