from __future__ import annotations

import pandas as pd

from bdmtf.revision.topology_fidelity import (
    _sample_depth_counts,
    extended_topology_metrics,
    fit_frozen_topology_model,
    simulate_learned_topology,
    summarize_learned_interaction,
    summarize_topology_fidelity,
)
import numpy as np


def _config() -> dict:
    return {
        "seed": 7,
        "communities": ["science"],
        "seeds": [0, 1],
        "model": {"ridge_alpha": 1.0, "local_neighbors": 2, "max_comments": 200},
        "operators": {
            "activation_log_gain": 0.6,
            "depth_fatigue_gain": 0.4,
            "exposure_pool_size": 30,
            "exposure_viewport": 5,
        },
        "analysis": {"bootstrap_samples": 200},
    }


def _contexts() -> pd.DataFrame:
    rows = []
    for index, split in enumerate(["train", "train", "train", "test"]):
        rows.append(
            {
                "community": "science",
                "post_id": f"p{index}",
                "split": split,
                "size": 20 + index * 10,
                "early_num_comments": 2 + index,
                "early_score_sum": 5 + index,
                "early_depth_max": 1,
                "early_comment_length_avg": 30,
                "author_link_karma": 100,
                "author_comment_karma": 100,
                "subreddit_subscribers": 1000,
                "is_self": False,
                "is_video": False,
            }
        )
    return pd.DataFrame(rows)


def _shapes() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "community": "science",
                "post_id": f"p{index}",
                "split": split,
                "size": 20 + index * 10,
                "max_depth": 3,
                "mean_branching_factor": 1.8,
                "root_reply_share": 0.55,
                "depths": [1, 1, 1, 2, 2, 3],
                "depth_pmf": [0.5, 1 / 3, 1 / 6],
            }
            for index, split in enumerate(["train", "train", "train", "test"])
        ]
    )


def test_frozen_model_excludes_test_shapes() -> None:
    model = fit_frozen_topology_model(_contexts(), _shapes(), _config())
    ids = model["communities"]["science"]["training_post_ids"]
    assert ids == ["p0", "p1", "p2"]
    assert model["uses_test_outcomes"] is False


def test_learned_topology_is_deterministic_and_valid() -> None:
    config = _config()
    model = fit_frozen_topology_model(_contexts(), _shapes(), config)
    context = _contexts().iloc[-1].to_dict()
    first, first_metadata = simulate_learned_topology(
        model, context, 0, config, behavior=False, ranking_mode="best"
    )
    second, second_metadata = simulate_learned_topology(
        model, context, 0, config, behavior=False, ranking_mode="best"
    )
    assert extended_topology_metrics(first) == extended_topology_metrics(second)
    assert first_metadata == second_metadata
    lookup = {node.node_id: node for node in first}
    for node in first[1:]:
        assert node.parent_id in lookup
        assert lookup[node.parent_id].depth == node.depth - 1


def test_factorial_summary_uses_post_clusters() -> None:
    rows = []
    for post_id in ["a", "b", "c", "d"]:
        for seed in [0, 1]:
            for cell, size, depth, exposure in [
                ("B0_P0", 10, 2.0, 0.2),
                ("B0_P1", 10, 2.0, 0.4),
                ("B1_P0", 12, 1.8, 0.2),
                ("B1_P1", 18, 1.4, 0.4),
            ]:
                rows.append(
                    {
                        "community": "science",
                        "post_id": post_id,
                        "seed": seed,
                        "cell": cell,
                        "size": size,
                        "mean_leaf_depth": depth,
                        "exposure_signal": exposure,
                    }
                )
    effects, cells, summary = summarize_learned_interaction(pd.DataFrame(rows), _config())
    assert len(effects) == 8
    assert len(cells) == 4
    assert summary["posts"] == 4
    assert summary["volume_ratio_of_ratios"] > 1.0
    assert summary["depth_interaction"] < 0.0


def test_sampled_depth_support_has_no_gaps() -> None:
    counts = _sample_depth_counts(
        4,
        np.asarray([0.05, 0.0, 0.0, 0.95]),
        False,
        0.0,
        0.0,
        np.random.default_rng(3),
    )
    assert counts.sum() == 4
    assert np.all(counts > 0)


def test_topology_summary_keeps_real_size_for_strata() -> None:
    real_rows = []
    sim_rows = []
    real_profiles = {}
    sim_profiles = {}
    for index in range(10):
        post_id = f"p{index}"
        real_rows.append(
            {
                "community": "science",
                "post_id": post_id,
                "size": index + 1,
                "mean_leaf_depth": 1.5,
                "max_depth": 3.0,
                "root_reply_share": 0.5,
                "mean_branching_factor": 1.8,
                "width_gini": 0.2,
                "leaf_fraction": 0.6,
                "depth_variance": 0.4,
                "p90_depth": 2.0,
                "repeat_author_share": 0.2,
            }
        )
        for seed in [0, 1]:
            sim_rows.append({**real_rows[-1], "seed": seed})
        real_profiles[f"science|{post_id}"] = [0.6, 0.3, 0.1]
        sim_profiles[f"science|{post_id}"] = [0.6, 0.3, 0.1]
    config = _config()
    config["analysis"].update(
        {
            "scalar_nwd_support_threshold": 1.0,
            "depth_nwd_strong_threshold": 0.5,
            "depth_nwd_partial_threshold": 1.0,
        }
    )
    _, summary, strata, _ = summarize_topology_fidelity(
        pd.DataFrame(real_rows),
        pd.DataFrame(sim_rows),
        real_profiles,
        sim_profiles,
        config,
    )
    assert summary["posts"] == 10
    assert strata["size_stratum"].nunique() == 5
