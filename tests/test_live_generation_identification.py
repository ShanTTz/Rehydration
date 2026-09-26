from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from bdmtf.revision.live_generation_identification import (
    PROTOCOL_FROZEN,
    PROTOCOL_LIVE,
    build_generation_prompt,
    dispersion_summary,
    select_real_test_post,
    validate_generation_response,
)


def test_generation_prompt_hides_state_only_for_frozen_protocol() -> None:
    post = {"title": "A result", "full_text": "Study body", "community": "science"}
    personas = [{"agent_id": 0, "persona": "Careful reader", "bio": "", "interests": []}]
    frozen = build_generation_prompt(post, personas, None)
    live = build_generation_prompt(
        post,
        personas,
        {"ranking": "controversial", "comments_so_far": 12},
    )
    assert "No engagement counts" in frozen
    assert "comments_so_far" not in frozen
    assert "comments_so_far" in live
    assert "untrusted_platform_state" in live


def test_response_validation_requires_each_agent_once() -> None:
    content = json.dumps(
        {
            "items": [
                {"agent_id": 0, "action": "reply", "polarity": "supportive", "content": "Useful context."},
                {"agent_id": 1, "action": "abstain", "content": "ignored"},
            ]
        }
    )
    items = validate_generation_response(content, {0, 1})
    assert items[1]["content"] == ""


def test_select_real_test_post_uses_nearest_median(tmp_path: Path) -> None:
    social_root = tmp_path / "social"
    community_dir = social_root / "science_data"
    community_dir.mkdir(parents=True)
    posts = pd.DataFrame(
        [
            {"post_id": "a", "title": "A", "full_text": "", "is_viral": 0},
            {"post_id": "b", "title": "B", "full_text": "", "is_viral": 0},
            {"post_id": "c", "title": "C", "full_text": "", "is_viral": 1},
        ]
    )
    posts.to_csv(community_dir / "posts_features_science.csv", index=False)
    splits = pd.DataFrame(
        [
            {"community": "science", "post_id": post_id, "split": "test"}
            for post_id in ["a", "b", "c"]
        ]
    )
    metrics = pd.DataFrame(
        [
            {"community": "science", "post_id": "a", "split": "test", "size": 10},
            {"community": "science", "post_id": "b", "split": "test", "size": 20},
            {"community": "science", "post_id": "c", "split": "test", "size": 100},
        ]
    )
    splits_path = tmp_path / "splits.csv"
    metrics_path = tmp_path / "metrics.csv"
    splits.to_csv(splits_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    selected, empirical = select_real_test_post(
        social_root, splits_path, metrics_path, "science"
    )
    assert selected["post_id"] == "b"
    assert empirical["community_test_median_size"] == 20.0


def test_dispersion_summary_detects_larger_live_variance() -> None:
    rows = []
    live = [-2.0, -1.0, 0.0, 1.0, 2.0, -1.5, 1.5, -0.5]
    frozen = [-0.2, -0.1, 0.0, 0.1, 0.2, -0.15, 0.15, -0.05]
    for repeat, (live_value, frozen_value) in enumerate(zip(live, frozen)):
        rows.extend(
            [
                {"repeat": repeat, "protocol": PROTOCOL_LIVE, "effect": live_value},
                {"repeat": repeat, "protocol": PROTOCOL_FROZEN, "effect": frozen_value},
            ]
        )
    summary = dispersion_summary(
        pd.DataFrame(rows), "effect", bootstrap_samples=300, permutation_samples=1000, seed=7
    )
    assert summary["live_sd"] > summary["frozen_sd"]
    assert summary["variance_ratio_live_over_frozen"] > 10
