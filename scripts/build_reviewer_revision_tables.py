from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKAGE = (
    ROOT
    / "manuscript"
    / "iclr2026_overleaf_package_next_revision_20260819"
)


def _macro(name: str, value: object) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build reviewer-validation LaTeX tables")
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    args = parser.parse_args()
    package = args.package.resolve()
    generated = package / "generated"
    generated.mkdir(parents=True, exist_ok=True)

    scm = pd.read_csv(
        ROOT
        / "artifacts"
        / "reviewer_validation"
        / "ground_truth_scm"
        / "recovery_summary.csv"
    )
    pool = pd.read_csv(
        ROOT
        / "artifacts"
        / "reviewer_validation"
        / "intent_pool_convergence"
        / "effect_stability.csv"
    )
    exhaustion = pd.read_csv(
        ROOT
        / "artifacts"
        / "reviewer_validation"
        / "intent_pool_convergence"
        / "exhaustion_by_condition.csv"
    )
    lemmy = pd.read_csv(
        ROOT / "artifacts" / "interventions" / "agent_replay" / "model_ranking.csv"
    )
    cross = json.loads(
        (
            ROOT
            / "artifacts"
            / "reviewer_validation"
            / "cross_platform_budget_audit"
            / "manifest.json"
        ).read_text(encoding="utf-8")
    )
    equal_budget = pd.DataFrame(cross["equal_budget_models"])

    engagement = scm[
        scm["outcome"].eq("engagement") & scm["semantic_shift"].eq(0.25)
    ].set_index("method")
    zero_capacity = int(pool["intent_pool_capacity_multiplier"].max())
    zero_exhaustion = exhaustion[
        exhaustion["intent_pool_capacity_multiplier"].eq(zero_capacity)
    ]
    pool_reference = pool[
        pool["intent_pool_capacity_multiplier"].eq(zero_capacity)
    ].set_index("metric")
    macros = [
        _macro("SCMSemanticShift", "0.25"),
        _macro("SCMCommonSeedBias", f"{engagement.loc['common_seed_live', 'bias']:.3f}"),
        _macro("SCMCommonSeedCoverage", f"{100 * engagement.loc['common_seed_live', 'coverage_95']:.1f}"),
        _macro("SCMRehydrationBias", f"{engagement.loc['rehydration', 'bias']:.3f}"),
        _macro("SCMRehydrationCoverage", f"{100 * engagement.loc['rehydration', 'coverage_95']:.1f}"),
        _macro("PoolZeroExhaustionCapacity", str(zero_capacity)),
        _macro("PoolMaxExhaustionPct", f"{100 * zero_exhaustion['exhaustion_rate'].max():.1f}"),
        _macro("PoolConvergedJointVolume", f"{pool_reference.loc['joint_volume_ratio', 'estimate']:.3f}"),
        _macro("PoolConvergedJointDepth", f"{pool_reference.loc['joint_mean_leaf_depth_delta', 'estimate']:.3f}"),
        _macro("LemmyRidgeMAE", f"{lemmy.set_index('model').loc['additive_ridge', 'primary_mae']:.3f}"),
        _macro("LemmyMajorityMAE", f"{lemmy.set_index('model').loc['majority_sign', 'primary_mae']:.3f}"),
        _macro("LemmyNearestMAE", f"{lemmy.set_index('model').loc['historical_nearest_neighbor', 'primary_mae']:.3f}"),
        _macro("HNTargetDataGainPct", f"{100 * cross['target_data_gain']:.1f}"),
        _macro("HNIncrementalAdapterGainPct", f"{100 * cross['incremental_adapter_gain']:.1f}"),
        _macro("HNFairTargetDefaultDistance", f"{cross['fair_target_default']['median_nwd']:.3f}"),
        _macro("HNBestAdapterBudget", str(cross["best_adapter_validation_budget"])),
    ]
    (generated / "reviewer_validation_macros.tex").write_text(
        "\n".join(macros) + "\n", encoding="utf-8"
    )

    labels = {
        "independent_live": "Independent live",
        "common_seed_live": "Common-seed live",
        "rehydration": "Rehydration",
    }
    scm_lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule Method & Bias $\downarrow$ & MSE $\downarrow$ & 95\% coverage $\uparrow$ \\",
        r"\midrule",
    ]
    for method in ("independent_live", "common_seed_live", "rehydration"):
        row = engagement.loc[method]
        scm_lines.append(
            f"{labels[method]} & {row['bias']:.3f} & {row['mse']:.4f} & "
            f"{100 * row['coverage_95']:.1f}\\% \\\\"
        )
    scm_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (generated / "table_ground_truth_scm.tex").write_text(
        "\n".join(scm_lines) + "\n", encoding="utf-8"
    )

    pool_pivot = pool.pivot(
        index="intent_pool_capacity_multiplier",
        columns="metric",
        values="estimate",
    )
    exhaustion_max = exhaustion.groupby("intent_pool_capacity_multiplier")[
        "exhaustion_rate"
    ].max()
    pool_lines = [
        r"\begin{tabular}{rrrr}",
        r"\toprule Capacity & Max exhausted & Joint volume & $\Delta$ leaf depth \\",
        r"\midrule",
    ]
    for capacity, row in pool_pivot.sort_index().iterrows():
        pool_lines.append(
            f"{int(capacity)}$\\times$ & {100 * exhaustion_max.loc[capacity]:.1f}\\% & "
            f"{row['joint_volume_ratio']:.3f}$\\times$ & "
            f"{row['joint_mean_leaf_depth_delta']:.3f} \\\\"
        )
    pool_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (generated / "table_intent_pool_capacity.tex").write_text(
        "\n".join(pool_lines) + "\n", encoding="utf-8"
    )

    display = {
        "zero_effect": "Zero effect",
        "majority_sign": "Majority sign",
        "intervention_type_mean": "Type mean",
        "additive_ridge": "Additive Ridge",
        "historical_nearest_neighbor": "Historical nearest",
        "bdmtf_agent_replay": "BDMTF event replay",
    }
    lemmy_index = lemmy.set_index("model")
    lemmy_lines = [
        r"\begin{tabular}{lrr}",
        r"\toprule Model & MAE $\downarrow$ & Direction $\uparrow$ \\",
        r"\midrule",
    ]
    for model in display:
        row = lemmy_index.loc[model]
        lemmy_lines.append(
            f"{display[model]} & {row['primary_mae']:.3f} & "
            f"{100 * row['direction_accuracy']:.1f}\\% \\\\"
        )
    lemmy_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (generated / "table_lemmy_strong_baselines.tex").write_text(
        "\n".join(lemmy_lines) + "\n", encoding="utf-8"
    )

    cross_labels = {
        "target_empirical_bootstrap": "Empirical bootstrap",
        "target_hawkes": "Hawkes",
        "target_default_bdmtf_common_seed": "BDMTF target default",
        "target_branching_process": "Branching process",
    }
    cross_lines = [
        r"\begin{tabular}{lrr}",
        r"\toprule Model & HN train & Median NWD $\downarrow$ \\",
        r"\midrule",
    ]
    for row in equal_budget.itertuples(index=False):
        cross_lines.append(
            f"{cross_labels[row.model]} & {int(row.target_train_cascades):,} & "
            f"{row.median_nwd:.3f} \\\\"
        )
    cross_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (generated / "table_cross_platform_equal_budget.tex").write_text(
        "\n".join(cross_lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

