from __future__ import annotations

import numpy as np
import pandas as pd

from bdmtf.revision.lemmy_path_fidelity import (
    PathView,
    _nearest_training_ids,
    _path_errors,
)


def _path(times: list[float], depths: list[float]) -> PathView:
    daily = np.zeros(2)
    for time in times:
        daily[min(int(time // 1440), 1)] += 1
    return PathView(
        times=np.asarray(times, dtype=float),
        depths=np.asarray(depths, dtype=float),
        daily_counts=daily,
        root_share=0.5 if times else 0.0,
        parent_hhi=0.5 if times else 0.0,
    )


def test_identical_paths_have_zero_error() -> None:
    value = _path([10.0, 1500.0], [1.0, 2.0])
    errors = _path_errors(
        value, value, horizon_days=2, count_scale=4.0, depth_scale=3.0
    )
    assert all(error == 0.0 for error in errors.values())


def test_empty_mismatch_is_penalized() -> None:
    observed = _path([10.0], [1.0])
    predicted = _path([], [])
    errors = _path_errors(
        observed, predicted, horizon_days=2, count_scale=4.0, depth_scale=3.0
    )
    assert errors["arrival_time_nwd"] == 1.0
    assert errors["depth_nwd"] == 1.0


def test_nearest_path_candidates_are_training_only() -> None:
    frame = pd.DataFrame(
        [
            {"intervention_id": "train", "intervention_type": "lock_post", "split": "train", "community_id": "c", "distance": 0.0},
            {"intervention_id": "validation", "intervention_type": "lock_post", "split": "validation", "community_id": "c", "distance": 0.0},
            {"intervention_id": "test", "intervention_type": "lock_post", "split": "test", "community_id": "c", "distance": 0.0},
        ]
    )
    ids = _nearest_training_ids(
        frame,
        frame.iloc[2],
        feature_columns=["distance"],
        k=2,
    )
    assert ids == ["train"]
