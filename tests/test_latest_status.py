from __future__ import annotations

from pathlib import Path

from bdmtf.latest_status import build_latest_status


def test_latest_status_uses_current_frozen_results(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    result = build_latest_status(
        root,
        tmp_path / "latest_status.json",
        sync_documents=False,
    )

    exact = result["paper_exact_reproduction"]
    assert exact["completed_runs"] == 9000
    assert round(exact["baseline_mean_leaf_depth"], 3) == 18.498
    assert round(exact["toxic_mean_leaf_depth"], 3) == 6.061

    phase = result["mechanism_phase_map"]
    assert phase["local_support"] == 24
    assert phase["local_scenarios"] == 24

    hn = result["hacker_news_adapter"]
    assert hn["best_validation_budget"] == 50
    assert round(hn["best_median_distance"], 3) == 0.422

    semantic = result["semantic_validation"]
    assert semantic["completed_calls"] == 120
    assert semantic["comments"] == 800
    assert semantic["full_agent_intent_expected_calls"] == 1575
