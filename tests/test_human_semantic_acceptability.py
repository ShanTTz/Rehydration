from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_human_semantic_acceptability.py"
RATINGS = (
    ROOT
    / "artifacts"
    / "reviewer_validation"
    / "semantic_parent_audit"
    / "human_results_20260825"
    / "RATINGS_PAIRED_DECODED.csv"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("semantic_acceptability", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_human_semantic_acceptability_regression(tmp_path: Path) -> None:
    module = _load_module()
    summary = module.analyze(RATINGS, tmp_path)

    assert summary["ratings"] == 1800
    assert summary["items"] == 600
    assert summary["anonymous_assignments"] == 90
    assert round(summary["acceptable"]["frame"]["estimate"], 4) == 0.7767
    assert round(summary["acceptable"]["verbatim"]["estimate"], 4) == 0.1767
    assert round(summary["paired_differences"]["acceptable"]["estimate"], 4) == 0.6
    assert summary["acceptability_balance"]["conflict_minus_baseline"]["ci_low"] < 0
    assert summary["acceptability_balance"]["conflict_minus_baseline"]["ci_high"] > 0
    assert (tmp_path / "human_semantic_acceptability_macros.tex").exists()
    assert (tmp_path / "table_human_semantic_acceptability.tex").exists()

