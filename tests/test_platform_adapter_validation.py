from __future__ import annotations

import pandas as pd

from bdmtf.platform_adapter_validation import chronological_platform_split


def test_platform_split_is_chronological_and_leakage_free() -> None:
    metrics = pd.DataFrame(
        {
            "platform": ["HackerNews"] * 10,
            "community": ["topstories"] * 10,
            "post_id": [str(index) for index in range(10)],
            "size": range(1, 11),
        }
    )
    events = pd.DataFrame(
        {
            "platform": ["HackerNews"] * 10,
            "community": ["topstories"] * 10,
            "content_id": [str(index) for index in range(10)],
            "created_at": pd.date_range(
                "2026-01-01", periods=10, freq="D", tz="UTC"
            ),
        }
    )
    split, manifest = chronological_platform_split(
        metrics,
        events,
        "HackerNews",
        {"train": 0.6, "validation": 0.2, "test": 0.2},
    )
    assert split["split"].value_counts().to_dict() == {
        "train": 6,
        "validation": 2,
        "test": 2,
    }
    assert not manifest["temporal_leakage_detected"]
    assert (
        split.loc[split["split"] == "train", "created_at"].max()
        < split.loc[split["split"] == "validation", "created_at"].min()
    )
    assert (
        split.loc[split["split"] == "validation", "created_at"].max()
        < split.loc[split["split"] == "test", "created_at"].min()
    )
