from __future__ import annotations

import pandas as pd

from bdmtf.mechanism_phase_map import (
    classify_effect,
    classify_stress_tier,
    parse_scenario,
    prepare_broad_phase_cells,
)


def test_effect_regions_cover_all_directional_quadrants() -> None:
    assert classify_effect(1.2, -0.5) == "shallow_swarm"
    assert classify_effect(1.2, 0.5) == "deep_swarm"
    assert classify_effect(0.8, -0.5) == "shallow_contraction"
    assert classify_effect(0.8, 0.5) == "deep_contraction"
    assert classify_effect(1.0, -0.5) == "boundary"


def test_scenario_parser_and_frozen_stress_tiers() -> None:
    parsed = parse_scenario(
        "factorial|trait=independent|ranking=hot|viewport=10"
    )
    assert parsed["design"] == "factorial"
    assert parsed["trait"] == "independent"
    assert parsed["ranking"] == "hot"
    assert parsed["viewport"] == "10"
    rules = {"viewport_k": ["-1"], "num_agents": ["500"]}
    assert classify_stress_tier("viewport_k=-1", rules) == "boundary_one_factor"
    assert classify_stress_tier("viewport_k=10", rules) == "bounded_one_factor"
    assert (
        classify_stress_tier(
            "factorial|trait=independent|ranking=hot|viewport=-1",
            rules,
        )
        == "multifactor_boundary"
    )


def test_broad_cells_recompute_shallow_swarm_from_effects() -> None:
    source = pd.DataFrame(
        {
            "scenario": ["reference", "num_agents=500"],
            "community": ["a", "a"],
            "volume_ratio": [1.4, 0.9],
            "leaf_depth_delta": [-1.0, 1.0],
        }
    )
    cells = prepare_broad_phase_cells(source, {"num_agents": ["500"]})
    assert cells["effect_region"].tolist() == [
        "shallow_swarm",
        "deep_contraction",
    ]
    assert cells["stress_tier"].tolist() == [
        "reference",
        "boundary_one_factor",
    ]
