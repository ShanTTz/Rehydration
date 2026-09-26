from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from bdmtf.data.social_loader import default_agents
from bdmtf.intent_pool import FrozenIntentPool
from bdmtf.metrics import compute_metrics
from bdmtf.schema import Intervention, RankingPolicy, SimulationConfig
from bdmtf.revision.reference_policy_strengthening import (
    ComparisonSimulator, activation_signal, event_records, factorial_contrasts, fit_activation_anchors, stable_seed,
)


def test_additive_signal_has_zero_cross_difference():
    anchors = {"drive_mean": [.3, .4, .2], "exposure_mean": [.2, .5, .3]}
    def f(d, e, mode):
        return activation_signal([d, .2, .4], [e, .5, .1], [1.8, .6, 1], mode, anchors)
    assert abs(f(.8, .9, "additive") - f(.8, .1, "additive") - f(.2, .9, "additive") + f(.2, .1, "additive")) < 1e-12
    assert abs(f(.8, .9, "original") - f(.8, .1, "original") - f(.2, .9, "original") + f(.2, .1, "original")) > .1


def test_training_gate_matching():
    rng = np.random.default_rng(42)
    d, e = rng.uniform(0, 1, (1000, 3)), rng.uniform(0, 1, (1000, 3))
    residual = rng.uniform(.1, .3, 1000)
    threshold = np.full(1000, .8)
    original = residual + activation_signal(d, e, [1.8, .6, 1], "original", {})
    anchors = fit_activation_anchors(np.column_stack([d, e, residual, threshold, original]), [1.8, .6, 1])
    for value in anchors["matched_gate_rates"].values():
        assert abs(value - anchors["target_gate_rate"]) <= .002


def _run(ranking, rank_null=False, form="original"):
    config = SimulationConfig(num_agents=6, steps=8, enable_external_traffic=False,
                              action_probability=.8, intent_pool_capacity_multiplier=100)
    anchors = {"drive_mean": [.3, .4, .2], "exposure_mean": [.2, .5, .3],
               "saturation_scale": .7, "offsets": {"additive": .1, "saturating": .2}}
    sim = ComparisonSimulator(default_agents(config), FrozenIntentPool([]), config=config, seed=17,
                              rank_null=rank_null, form=form, anchors=anchors)
    state, traces = sim.run(post_id="p", title="Example", intervention=Intervention("test", core="toxic", ranking=ranking))
    for node in state.comments.values():
        if node.parent_id:
            assert state.comments[node.parent_id].depth + 1 == node.depth
    return compute_metrics(state, traces), state


@pytest.mark.parametrize("form", ["original", "additive", "saturating"])
def test_replay_deterministic_and_valid_tree(form):
    first, _ = _run(RankingPolicy.BEST, form=form)
    second, _ = _run(RankingPolicy.BEST, form=form)
    assert first == second


def test_disconnected_rank_is_exact_null():
    first, _ = _run(RankingPolicy.BEST, rank_null=True)
    second, _ = _run(RankingPolicy.CONTROVERSIAL, rank_null=True)
    assert first == second


def test_event_records_round_trip_parquet(tmp_path):
    _, state = _run(RankingPolicy.BEST)
    nodes = [n for n in state.comments.values() if n.parent_id]
    records = event_records("example", nodes)
    assert records
    path = tmp_path / "events.parquet"
    pd.DataFrame(records).to_parquet(path, index=False)
    restored = pd.read_parquet(path)
    assert restored.step.tolist() == [n.created_step for n in nodes]
    assert restored.node_id.is_unique


def test_operation_stream_does_not_shift_after_unused_draws():
    cfg = SimulationConfig(num_agents=2)
    sim = ComparisonSimulator(default_agents(cfg), FrozenIntentPool([]), config=cfg, seed=77)
    sim._stream("target")
    expected = sim.rng.random()
    sim._stream("polarity")
    for _ in range(100):
        sim.rng.random()
    sim._stream("target")
    assert sim.rng.random() == expected
    assert stable_seed(1, "p", 2) == stable_seed(1, "p", 2)


def test_factorial_contrasts_known_values():
    cells = ["BASELINE", "BASELINE_CONTROVERSIAL", "CORE_TOXIC_BEST", "CORE_TOXIC_CONTROVERSIAL"]
    frame = pd.DataFrame([{"community": "c", "post_id": "p", "seed": 1, "panel": "qref", "family": "f", "form": "original",
                           "condition": c, "comment_volume": n, "mean_leaf_depth": d, "surface_toxicity_mean": 0.0}
                          for c, n, d in zip(cells, [9, 19, 29, 119], [5, 4, 4, 1])])
    result = factorial_contrasts(frame).iloc[0]
    assert np.exp(result.interaction_volume_log) == pytest.approx(2)
    assert result.interaction_depth == -2
    with pytest.raises((ValueError, KeyError)):
        factorial_contrasts(frame.iloc[:-1])
