from __future__ import annotations

import pandas as pd

from bdmtf.revision.lemmy_content_matched_validation import (
    _greedy_one_to_one,
    build_exact_url_pairs,
)


def test_exact_url_matching_is_one_to_one_and_time_bounded() -> None:
    hn = pd.DataFrame(
        [
            {
                "content_id": "h1",
                "event_id": "h1",
                "parent_event_id": "",
                "created_at": "2026-01-01T00:00:00Z",
                "event_type": "post",
                "content_url": "https://example.test/story?utm_source=x",
                "text": "Story",
            }
        ]
    )
    lemmy = pd.DataFrame(
        [
            {
                "content_id": "l1",
                "published": "2026-01-01T01:00:00Z",
                "url": "https://example.test/story",
                "title": "Story",
                "reported_comments": 4,
            },
            {
                "content_id": "l2",
                "published": "2026-01-10T00:00:00Z",
                "url": "https://example.test/story",
                "title": "Story",
                "reported_comments": 5,
            },
        ]
    )
    pairs = build_exact_url_pairs(hn, lemmy, {}, 72.0)
    assert len(pairs) == 1
    assert pairs.iloc[0]["lemmy_content_id"] == "l1"
    assert pairs.iloc[0]["time_distance_hours"] == 1.0


def test_greedy_pairing_never_reuses_a_post() -> None:
    candidates = pd.DataFrame(
        [
            {
                "hn_content_id": "h1",
                "lemmy_content_id": "l1",
                "time_distance_hours": 1.0,
                "title_similarity": 1.0,
            },
            {
                "hn_content_id": "h1",
                "lemmy_content_id": "l2",
                "time_distance_hours": 2.0,
                "title_similarity": 1.0,
            },
            {
                "hn_content_id": "h2",
                "lemmy_content_id": "l1",
                "time_distance_hours": 3.0,
                "title_similarity": 1.0,
            },
        ]
    )
    selected = _greedy_one_to_one(candidates)
    assert not selected["hn_content_id"].duplicated().any()
    assert not selected["lemmy_content_id"].duplicated().any()
