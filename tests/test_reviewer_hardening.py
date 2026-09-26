from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_lemmy_path_metric_definition_and_weight_sensitivity(tmp_path: Path) -> None:
    module = _load_script("analyze_lemmy_path_metric")
    summary = module.analyze(
        module.DEFAULT_SCORES,
        module.DEFAULT_MANIFEST,
        tmp_path,
        random_draws=1_000,
        seed=30371,
    )

    assert summary["test_interventions"] == 103
    assert round(summary["equal_weight_mean_gain"], 6) == 0.026135
    assert summary["random_weight_positive_mean_gain_share"] == 1.0
    assert set(summary["components"]) == {
        "cumulative_count_nmae",
        "arrival_time_nwd",
        "depth_nwd",
        "root_share_ae",
        "parent_hhi_ae",
    }
    assert (tmp_path / "lemmy_path_metric_macros.tex").exists()


def test_qref_family_effect_sensitivity_uses_three_frozen_sources(tmp_path: Path) -> None:
    module = _load_script("analyze_qref_effect_sensitivity")
    manifest = module.analyze(
        module.DEFAULT_STRICT,
        module.DEFAULT_COUPLED,
        tmp_path,
        bootstrap_samples=500,
        seed=30371,
    )

    assert manifest["matched_posts"] == 50
    assert manifest["cached_intent_responses"] == 1_500
    assert manifest["families"] == ["deepseek", "openai", "qwen"]
    assert all(manifest["all_family_mean_directions"].values())
    assert (tmp_path / "table_qref_effect_sensitivity.tex").exists()


def test_live_semantic_ceiling_tasks_are_blinded_and_unique(tmp_path: Path, monkeypatch) -> None:
    module = _load_script("live_semantic_ceiling")
    monkeypatch.setattr(module, "OUTPUT", tmp_path)
    manifest = module.prepare(sample_size=200, seed=30371, model="gpt-4o-mini")

    tasks = [
        json.loads(line)
        for line in (tmp_path / "live_semantic_tasks.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert manifest["tasks"] == 200
    assert len({task["item_id"] for task in tasks}) == 200
    assert all("candidate_a" not in task for task in tasks)
    assert all("relevance_score" not in task for task in tasks)
    assert all("coherence_score" not in task for task in tasks)


def test_adaptive_fixed_effect_decomposition(tmp_path: Path) -> None:
    module = _load_script("analyze_adaptive_fixed_decomposition")
    summary = module.analyze(
        module.DEFAULT_INPUT,
        tmp_path,
        bootstrap_samples=1_000,
        seed=30371,
    )

    assert summary["threads"] == 25
    assert summary["paired_seeds_per_thread"] == 5
    assert round(summary["mean_gap"], 3) == -0.035
    assert round(summary["median_absolute_gap"], 3) == 0.072
    assert summary["direction_reversals"] == 0
    assert summary["mean_gap_ci"][0] < 0 < summary["mean_gap_ci"][1]


def test_three_arm_semantic_review_package_balances_items(tmp_path: Path) -> None:
    builder = _load_script("build_live_semantic_blind_review")
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    source_tasks = [
        json.loads(line)
        for line in (
            ROOT
            / "artifacts"
            / "reviewer_validation"
            / "live_semantic_ceiling"
            / "live_semantic_tasks.jsonl"
        ).read_text(encoding="utf-8").splitlines()[:20]
    ]
    with (input_dir / "live_semantic_tasks.jsonl").open("w", encoding="utf-8") as stream:
        for task in source_tasks:
            stream.write(json.dumps(task) + "\n")
    with (input_dir / "live_semantic_responses.jsonl").open("w", encoding="utf-8") as stream:
        for task in source_tasks:
            stream.write(
                json.dumps(
                    {
                        "task_id": task["task_id"],
                        "intent_preserving_reply": "intent-preserving comparator",
                        "live_reply": "live comparator",
                    }
                )
                + "\n"
            )

    manifest = builder.build(
        input_dir,
        output_dir,
        ratings_per_item=3,
        items_per_assignment=10,
        seed=30371,
    )
    master = pd.read_csv(output_dir / "researcher_only" / "RESEARCHER_MASTER_DO_NOT_SHARE.csv")
    assert manifest["assignments"] == 6
    assert master.groupby("task_id")["assignment_id"].nunique().eq(3).all()
    for _, row in master.iterrows():
        assert {
            row["candidate_A_source"],
            row["candidate_B_source"],
            row["candidate_C_source"],
        } == set(builder.SOURCES)


def test_full_pool_capacity_summary_accounts_for_catalog_and_cost(tmp_path: Path) -> None:
    module = _load_script("summarize_full_pool_capacity")
    manifest = module.summarize(
        ROOT / "artifacts" / "reviewer_validation" / "intent_pool_full_512",
        tmp_path,
        bootstrap_samples=100,
        seed=30371,
        capacity_multiplier=512,
    )

    assert manifest["posts"] == 500
    assert manifest["runs"] == 2_000
    assert manifest["unique_intents_total"] == 22_974
    assert manifest["opportunity_capacity_total"] == 11_762_688
    assert manifest["maximum_exhaustion_rate"] == 0
    assert manifest["additional_llm_calls_for_multiplier"] == 0
    assert manifest["additional_llm_tokens_for_multiplier"] == 0
