from __future__ import annotations

from dataclasses import replace

from bdmtf.reviewer_trait_coupling import (
    analyze_trait_coupling,
    make_post_core_trait_transform,
)
from bdmtf.schema import AgentProfile, Intervention


def _agents() -> list[AgentProfile]:
    return [
        AgentProfile(
            agent_id=index,
            antagonism=index / 9,
            attention=0.5,
            prosocial=max(0.0, 1.0 - 1.5 * index / 9),
            threshold=0.5,
        )
        for index in range(10)
    ]


def test_default_coupled_transform_preserves_shifted_agents() -> None:
    agents = _agents()
    shifted = [agent.with_core_shift("toxic") for agent in agents]
    transform = make_post_core_trait_transform(agents, "coupled_original", 7)
    assert transform(shifted, Intervention.from_dict({"core": "toxic"})) == shifted


def test_decoupled_modes_preserve_prosocial_across_core_shift() -> None:
    agents = _agents()
    intervention = Intervention.from_dict({"core": "toxic"})
    for mode in (
        "independent_marginal",
        "shuffled_marginal",
        "aggressive_constructive",
    ):
        transform = make_post_core_trait_transform(agents, mode, 17)
        baseline = transform(
            [agent.with_core_shift("baseline") for agent in agents],
            Intervention.from_dict({"core": "baseline"}),
        )
        toxic = transform(
            [agent.with_core_shift("toxic") for agent in agents],
            intervention,
        )
        assert [agent.prosocial for agent in baseline] == [
            agent.prosocial for agent in toxic
        ]
        assert any(
            base.antagonism != shifted.antagonism
            for base, shifted in zip(baseline, toxic)
        )
    aggressive = make_post_core_trait_transform(
        agents, "aggressive_constructive", 17
    )(
        [agent.with_core_shift("baseline") for agent in agents],
        Intervention.from_dict({"core": "baseline"}),
    )
    assert any(
        agent.antagonism >= 0.75 and agent.prosocial >= 0.75
        for agent in aggressive
    )


def test_analysis_excludes_incomplete_pair_blocks() -> None:
    records = [
        {
            "community": "science",
            "post_id": "complete",
            "seed": 0,
            "trait_mode": "coupled_original",
            "condition": "BASELINE",
            "comment_volume": 10.0,
            "mean_leaf_depth": 8.0,
        },
        {
            "community": "science",
            "post_id": "complete",
            "seed": 0,
            "trait_mode": "coupled_original",
            "condition": "CORE_TOXIC_CONTROVERSIAL",
            "comment_volume": 30.0,
            "mean_leaf_depth": 4.0,
        },
        {
            "community": "science",
            "post_id": "partial",
            "seed": 0,
            "trait_mode": "coupled_original",
            "condition": "BASELINE",
            "comment_volume": 12.0,
            "mean_leaf_depth": 7.0,
        },
    ]
    contrasts, summary, manifest = analyze_trait_coupling(
        records, bootstrap_samples=0
    )
    assert len(contrasts) == 1
    assert contrasts.iloc[0]["post_id"] == "complete"
    assert manifest["complete_blocks"] == 1
    assert not summary[["ci_low", "ci_high"]].isna().any().any()
