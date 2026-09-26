from __future__ import annotations

import pandas as pd

from bdmtf.revision.lemmy_agent_replay import (
    ReplayParameters,
    _daily_metrics,
    _materialize_condition,
    _observed_effects,
)
from bdmtf.revision.policies import EventNode


def _parameters() -> ReplayParameters:
    return ReplayParameters(
        root_reply_share=0.5,
        repeat_author_probability=0.5,
        position_decay=0.82,
        depth_bias=0.0,
        viewport_k=10,
        ranking="new",
    )


def _plan() -> list[dict[str, float]]:
    return [
        {
            "minute": 10.0,
            "visibility_draw": 0.1,
            "author_repeat_draw": 0.9,
            "author_choice_draw": 0.1,
            "parent_root_draw": 0.1,
            "parent_choice_draw": 0.1,
        },
        {
            "minute": 20.0,
            "visibility_draw": 0.9,
            "author_repeat_draw": 0.9,
            "author_choice_draw": 0.9,
            "parent_root_draw": 0.9,
            "parent_choice_draw": 0.9,
        },
    ]


def test_lock_post_generates_no_future_intervention_events() -> None:
    initial = [
        EventNode(
            "post",
            None,
            0,
            -60.0,
            author_id="post_author",
            metadata={"root_post": True},
        )
    ]
    control = _materialize_condition(
        initial_nodes=initial,
        plan=_plan(),
        parameters=_parameters(),
        condition="no_intervention",
        intervention_type="lock_post",
        remove_visibility_remaining=0.2,
    )
    treated = _materialize_condition(
        initial_nodes=initial,
        plan=_plan(),
        parameters=_parameters(),
        condition="intervention",
        intervention_type="lock_post",
        remove_visibility_remaining=0.2,
    )
    assert _daily_metrics(control, 1)["reply_count"] == 2.0
    assert _daily_metrics(treated, 1)["reply_count"] == 0.0


def test_remove_post_thins_events_and_keeps_valid_parent_chain() -> None:
    initial = [
        EventNode(
            "post",
            None,
            0,
            -60.0,
            author_id="post_author",
            metadata={"root_post": True},
        )
    ]
    treated = _materialize_condition(
        initial_nodes=initial,
        plan=_plan(),
        parameters=_parameters(),
        condition="intervention",
        intervention_type="remove_post",
        remove_visibility_remaining=0.2,
    )
    generated = [
        node for node in treated if node.metadata.get("simulated_future")
    ]
    assert len(generated) == 1
    ids = {node.node_id for node in treated}
    assert generated[0].parent_id in ids


def test_observed_effect_is_matched_pair_difference_in_differences() -> None:
    rows = []
    for treated, pre, post in ((0, 2.0, 3.0), (1, 2.5, 1.5)):
        for relative in range(-7, 0):
            rows.append(
                {
                    "intervention_id": "i1",
                    "outcome": "reply_count",
                    "treated": treated,
                    "relative_period": relative,
                    "value": pre,
                }
            )
        for relative in range(0, 8):
            rows.append(
                {
                    "intervention_id": "i1",
                    "outcome": "reply_count",
                    "treated": treated,
                    "relative_period": relative,
                    "value": post,
                }
            )
    effect = _observed_effects(
        pd.DataFrame(rows),
        ["i1"],
        ["reply_count"],
    )
    assert effect.iloc[0]["observed_effect"] == -2.0
