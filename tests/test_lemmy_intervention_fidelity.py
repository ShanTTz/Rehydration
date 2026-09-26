from __future__ import annotations

import json

import pandas as pd

from bdmtf.revision.lemmy_intervention_fidelity import (
    prepare_intervention_dataset,
    run_lemmy_intervention_fidelity,
)


def _fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    panel_rows = []
    match_rows = []
    outcomes = ("reply_count", "active_authors", "max_depth", "removed_replies")
    for kind_index, intervention_type in enumerate(("lock_post", "remove_post")):
        for index in range(30):
            intervention_id = f"{intervention_type}:{index}"
            event_time = pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(
                days=kind_index * 100 + index
            )
            match_rows.append(
                {
                    "intervention_id": intervention_id,
                    "intervention_type": intervention_type,
                    "community_id": f"c{index % 3}",
                    "event_time": event_time,
                    "treated_age_hours": 10 + index,
                    "control_age_hours": 11 + index,
                    "distance": 0.1,
                    "pretrajectory_distance": 0.2,
                    "match_score": 0.3,
                }
            )
            activity = 1.0 + (index % 5)
            for outcome_index, outcome in enumerate(outcomes, start=1):
                for treated in (0, 1):
                    for period in range(-7, 8):
                        baseline = activity * outcome_index + 0.1 * period
                        value = max(0.0, baseline)
                        if treated and period >= 0:
                            interaction = (
                                -0.45 * activity * outcome_index
                                if intervention_type == "lock_post"
                                else -0.25 * activity * outcome_index
                            )
                            value = max(0.0, value + interaction)
                        panel_rows.append(
                            {
                                "unit_id": f"{intervention_id}|{treated}",
                                "period": event_time.floor("D")
                                + pd.Timedelta(days=period),
                                "treated": treated,
                                "relative_period": period,
                                "outcome": outcome,
                                "value": value,
                                "community_id": f"c{index % 3}",
                                "intervention_id": intervention_id,
                                "intervention_type": intervention_type,
                                "content_id": f"p{index}",
                            }
                        )
    return pd.DataFrame(panel_rows), pd.DataFrame(match_rows)


def _config() -> dict:
    return {
        "protocol": {
            "primary_outcomes": [
                "reply_count",
                "active_authors",
                "max_depth",
            ],
            "secondary_outcomes": ["removed_replies"],
        },
        "train_fraction": 0.6,
        "validation_fraction": 0.2,
        "pre_periods": list(range(-7, 0)),
        "post_periods": list(range(0, 8)),
        "ridge_grid": [0.1, 1.0],
        "prediction_interval": 0.95,
        "bootstrap_samples": 40,
        "seed": 11,
    }


def test_lemmy_fidelity_uses_chronological_pre_features(tmp_path) -> None:
    panel, matches = _fixture()
    features, effects, additive, interaction = prepare_intervention_dataset(
        panel,
        matches,
        _config(),
    )

    assert features["split"].value_counts().to_dict() == {
        "train": 36,
        "validation": 12,
        "test": 12,
    }
    assert not any("post_" in column for column in additive)
    assert len(interaction) > len(additive)
    assert len(effects) == 60 * 4
    for _, group in features.groupby("intervention_type"):
        assert (
            group[group["split"].eq("train")]["event_time"].max()
            <= group[group["split"].eq("validation")]["event_time"].min()
        )
        assert (
            group[group["split"].eq("validation")]["event_time"].max()
            <= group[group["split"].eq("test")]["event_time"].min()
        )

    panel_path = tmp_path / "panel.parquet"
    matches_path = tmp_path / "matches.csv"
    natural_path = tmp_path / "natural.json"
    panel.to_parquet(panel_path, index=False)
    matches.to_csv(matches_path, index=False)
    natural_path.write_text(
        json.dumps(
            {
                "primary_diagnostics": [
                    {
                        "outcome": outcome,
                        "effect": -1.0,
                        "ci_low": -2.0,
                        "ci_high": -0.1,
                    }
                    for outcome in (
                        "reply_count",
                        "active_authors",
                        "max_depth",
                    )
                ]
            }
        ),
        encoding="utf-8",
    )
    result = run_lemmy_intervention_fidelity(
        panel_path,
        matches_path,
        natural_path,
        tmp_path / "output",
        _config(),
    )

    assert result["status"] == "complete"
    assert result["chronological_no_test_leakage"] is True
    assert result["test_pairs"] == 12
    assert (tmp_path / "output" / "test_predictions.parquet").is_file()
