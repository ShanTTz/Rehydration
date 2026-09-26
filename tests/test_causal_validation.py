from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bdmtf.revision.causal_validation import (
    aggregate_calibrated_synthetic_control,
    analyze_human_rct,
    estimate_panel_did,
    evaluate_intervention_fidelity,
    fit_pre_intervention_forecast,
    rct_readiness,
    synthetic_control_estimate,
    uncontrolled_interrupted_time_series,
)
from bdmtf.revision.human_experiment import assignment_for_index, rank_thread, validate_protocol
from bdmtf.revision.workflows import run_natural_experiments_workflow


def _synthetic_panel(effect: float = 3.0) -> pd.DataFrame:
    records = []
    for treated in (0, 1):
        for unit in range(20):
            for relative in range(-4, 5):
                value = 5.0 + 0.2 * relative + treated * effect * (relative >= 0)
                records.append(
                    {
                        "unit_id": f"{treated}-{unit}",
                        "period": pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(days=relative),
                        "treated": treated,
                        "relative_period": relative,
                        "outcome": "messages",
                        "value": value,
                    }
                )
    return pd.DataFrame(records)


def test_panel_did_recovers_known_effect() -> None:
    result = estimate_panel_did(_synthetic_panel(), bootstrap_samples=100, seed=1)
    assert result["effect"] == pytest.approx(3.0)
    assert result["ci_low"] == pytest.approx(3.0)
    assert result["ci_high"] == pytest.approx(3.0)
    assert abs(result["pretrend_slope"]) < 1e-9


def test_synthetic_control_recovers_known_effect() -> None:
    result = synthetic_control_estimate(_synthetic_panel())
    assert len(result) == 20
    assert result["effect"].mean() == pytest.approx(3.0)
    assert result["pre_rmse"].max() < 1e-9
    assert (result["effective_donors"] >= 1).all()


def test_aggregate_calibrated_synthetic_control_recovers_effect() -> None:
    summary, event_table, weights = (
        aggregate_calibrated_synthetic_control(
            _synthetic_panel(),
            ["messages"],
            bootstrap_samples=100,
            seed=1,
        )
    )
    diagnostic = summary["diagnostics"][0]
    assert diagnostic["effect"] == pytest.approx(3.0)
    assert diagnostic["ci_low"] == pytest.approx(3.0)
    assert diagnostic["ci_high"] == pytest.approx(3.0)
    assert diagnostic["pretrend_flag"] is False
    assert summary["effective_controls"] >= 19.0
    assert weights["weight"].sum() == pytest.approx(1.0)
    assert len(event_table) == 9


def test_uncontrolled_its_is_labeled_noncausal() -> None:
    result = uncontrolled_interrupted_time_series(_synthetic_panel())
    treated = result[result["unit_id"].str.startswith("1-")]
    assert len(result) == 40
    assert treated["level_change"].mean() == pytest.approx(3.0)
    assert result["causal_claim_allowed"].eq(False).all()


def test_pre_intervention_forecast_does_not_require_controls() -> None:
    treated_only = _synthetic_panel().query("treated == 1")
    result = fit_pre_intervention_forecast(treated_only)
    assert len(result) == 20
    assert result["max_training_period"].eq(-1).all()


def test_rct_gate_requires_ethics_and_preregistration() -> None:
    frame = pd.DataFrame(
        [
            {
                "participant_id": "a",
                "arm": "control",
                "ranking": "new",
                "context": "neutral",
                "correction": "none",
                "consent": True,
                "eligible": True,
                "completed": True,
            },
            {
                "participant_id": "b",
                "arm": "treatment",
                "ranking": "top",
                "context": "conflict",
                "correction": "reply",
                "consent": True,
                "eligible": True,
                "completed": True,
            },
        ]
    )
    readiness = rct_readiness(frame, {"frozen_primary_outcomes": ["scroll_depth"]})
    assert readiness["claim_allowed"] is False
    assert any("ethics" in issue for issue in readiness["issues"])


def test_intervention_fidelity_metrics(tmp_path) -> None:
    predictions = pd.DataFrame(
        {
            "intervention_id": ["a", "b"],
            "outcome": ["size", "size"],
            "predicted_effect": [2.0, -1.0],
            "ci_low": [1.0, -2.0],
            "ci_high": [3.0, 0.0],
        }
    )
    observed = pd.DataFrame(
        {
            "intervention_id": ["a", "b"],
            "outcome": ["size", "size"],
            "observed_effect": [2.5, -0.5],
        }
    )
    result = evaluate_intervention_fidelity(predictions, observed, tmp_path)
    assert result["direction_accuracy"] == 1.0
    assert result["interval_coverage"] == 1.0
    assert result["mae"] == 0.5


def test_confirmatory_workflow_does_not_promote_underpowered_panel(
    tmp_path,
) -> None:
    panel = _synthetic_panel()
    panel["intervention_id"] = panel["unit_id"].str.split("-").str[-1]
    panel_path = tmp_path / "panel.parquet"
    panel.to_parquet(panel_path, index=False)
    result = run_natural_experiments_workflow(
        tmp_path,
        {
            "causal_analysis": {
                "panel_path": "panel.parquet",
                "output_name": "confirmatory",
                "bootstrap_samples": 20,
                "confirmatory": True,
                "required_outcomes": ["messages"],
                "claim_gates": {
                    "minimum_matched_pairs": 100,
                    "minimum_treated_overlap": 0.8,
                    "require_no_pretrend_flag": True,
                    "require_all_frozen_outcomes": True,
                },
            }
        },
    )
    assert result["status"] == "confirmatory_diagnostics_failed"
    assert result["claim_allowed"] is False
    assert any("matched pairs" in item for item in result["gate_failures"])


def test_human_experiment_block_contains_all_ten_cells() -> None:
    assignments = [assignment_for_index(index) for index in range(10)]
    cells = {
        (item["ranking"], item["context"])
        for item in assignments
    }
    assert len(cells) == 10
    assert {
        item["correction"] for item in assignments
    } == {"within_participant"}


def test_human_experiment_ranking_is_deterministic() -> None:
    comments = [
        {"comment_id": "a", "score": 1, "upvotes": 5, "downvotes": 4, "created_utc": 10},
        {"comment_id": "b", "score": 3, "upvotes": 3, "downvotes": 0, "created_utc": 20},
    ]
    first = rank_thread(comments, "controversial", now=30)
    second = rank_thread(comments, "controversial", now=30)
    assert first == second
    assert first[0]["comment_id"] == "a"


def test_human_server_refuses_unapproved_protocol() -> None:
    with pytest.raises(PermissionError):
        validate_protocol({"frozen_primary_outcomes": ["scroll_depth"]})
