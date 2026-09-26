from __future__ import annotations

import pandas as pd

from bdmtf.revision.tbbt_intervention_fidelity import (
    _bootstrap_improvement,
    mechanism_ratio,
)


def test_quarantine_reduces_volume_through_platform_and_agent_channels() -> None:
    ratio = mechanism_ratio(
        "quarantine",
        novel_share=0.1,
        switch_share=0.9,
        removed_rate=0.0,
        external_traffic_multiplier=1.1,
        conflict_response_coefficient=0.75,
        hostile_dropout_probability=0.05,
        toxic_activation_boost=0.25,
        visibility_remaining=0.4,
    )
    assert 0.0 < ratio < 1.0
    lower_novelty = mechanism_ratio(
        "quarantine",
        novel_share=0.05,
        switch_share=0.9,
        removed_rate=0.0,
        external_traffic_multiplier=1.1,
        conflict_response_coefficient=0.75,
        hostile_dropout_probability=0.05,
        toxic_activation_boost=0.25,
        visibility_remaining=0.4,
    )
    assert ratio < lower_novelty


def test_post_removal_rehydrates_switchable_intents() -> None:
    ratio = mechanism_ratio(
        "post_removal",
        novel_share=0.1,
        switch_share=0.9,
        removed_rate=0.03,
        external_traffic_multiplier=1.1,
        conflict_response_coefficient=0.75,
        hostile_dropout_probability=0.05,
        toxic_activation_boost=0.25,
        visibility_remaining=0.4,
    )
    assert ratio > 1.0
    no_switch = mechanism_ratio(
        "post_removal",
        novel_share=0.1,
        switch_share=0.0,
        removed_rate=0.03,
        external_traffic_multiplier=1.1,
        conflict_response_coefficient=0.75,
        hostile_dropout_probability=0.05,
        toxic_activation_boost=0.25,
        visibility_remaining=0.4,
    )
    assert no_switch < 1.0


def test_bootstrap_improvement_uses_event_level_pairing() -> None:
    frame = pd.DataFrame(
        {
            "zero_absolute_error_pp": [10.0, 20.0, 30.0],
            "absolute_error_pp": [2.0, 4.0, 6.0],
        }
    )
    result = _bootstrap_improvement(frame, draws=2000, seed=7)
    assert result["mean"] == 16.0
    assert result["ci_low"] > 0.0
