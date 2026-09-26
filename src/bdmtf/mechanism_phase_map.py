from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bdmtf.revision.provenance import sha256_file, write_json


QUADRANT_ORDER = (
    "shallow_swarm",
    "deep_swarm",
    "shallow_contraction",
    "deep_contraction",
    "boundary",
)


def classify_effect(volume_ratio: float, leaf_depth_delta: float) -> str:
    if not np.isfinite(volume_ratio) or not np.isfinite(leaf_depth_delta):
        return "boundary"
    if np.isclose(volume_ratio, 1.0) or np.isclose(leaf_depth_delta, 0.0):
        return "boundary"
    if volume_ratio > 1.0 and leaf_depth_delta < 0.0:
        return "shallow_swarm"
    if volume_ratio > 1.0 and leaf_depth_delta > 0.0:
        return "deep_swarm"
    if volume_ratio < 1.0 and leaf_depth_delta < 0.0:
        return "shallow_contraction"
    return "deep_contraction"


def parse_scenario(scenario: str) -> dict[str, str]:
    if scenario == "reference":
        return {"design": "reference", "factor": "reference", "value": "reference"}
    if scenario.startswith("factorial|"):
        values = {"design": "factorial", "factor": "trait_ranking_viewport"}
        for token in scenario.split("|")[1:]:
            key, value = token.split("=", 1)
            values[key] = value
        values["value"] = (
            f"trait={values['trait']}|ranking={values['ranking']}|"
            f"viewport={values['viewport']}"
        )
        return values
    factor, value = scenario.split("=", 1)
    return {
        "design": "one_factor",
        "factor": factor,
        "value": value,
    }


def classify_stress_tier(
    scenario: str,
    extreme_values: Mapping[str, list[str]],
) -> str:
    parsed = parse_scenario(scenario)
    if parsed["design"] == "reference":
        return "reference"
    if parsed["design"] == "factorial":
        if parsed.get("viewport") in {
            str(value) for value in extreme_values.get("viewport_k", [])
        }:
            return "multifactor_boundary"
        return "multifactor_stress"
    extreme = {
        str(value) for value in extreme_values.get(parsed["factor"], [])
    }
    return (
        "boundary_one_factor"
        if parsed["value"] in extreme
        else "bounded_one_factor"
    )


def prepare_broad_phase_cells(
    broad: pd.DataFrame,
    extreme_values: Mapping[str, list[str]],
) -> pd.DataFrame:
    required = {
        "scenario",
        "community",
        "volume_ratio",
        "leaf_depth_delta",
    }
    missing = required - set(broad.columns)
    if missing:
        raise ValueError(f"Broad stress table is missing columns: {sorted(missing)}")
    cells = broad.copy()
    parsed = cells["scenario"].astype(str).map(parse_scenario)
    cells["design"] = parsed.map(lambda row: row["design"])
    cells["factor"] = parsed.map(lambda row: row["factor"])
    cells["factor_value"] = parsed.map(lambda row: row["value"])
    for column in ("trait", "ranking", "viewport"):
        cells[column] = parsed.map(lambda row: row.get(column, ""))
    cells["stress_tier"] = cells["scenario"].astype(str).map(
        lambda value: classify_stress_tier(value, extreme_values)
    )
    cells["effect_region"] = [
        classify_effect(float(volume), float(depth))
        for volume, depth in zip(
            cells["volume_ratio"],
            cells["leaf_depth_delta"],
            strict=True,
        )
    ]
    cells["shallow_swarm"] = cells["effect_region"].eq("shallow_swarm")
    cells["evidence_tier"] = "global_stress_boundary_map"
    return cells


def prepare_local_phase_scenarios(local: pd.DataFrame) -> pd.DataFrame:
    required = {
        "scenario",
        "geometric_volume_ratio",
        "mean_leaf_depth_delta",
        "scenario_supports_shallow_swarm",
    }
    missing = required - set(local.columns)
    if missing:
        raise ValueError(
            f"Local sensitivity table is missing columns: {sorted(missing)}"
        )
    scenarios = local.copy()
    scenarios["volume_ratio"] = scenarios["geometric_volume_ratio"].astype(float)
    scenarios["leaf_depth_delta"] = scenarios["mean_leaf_depth_delta"].astype(
        float
    )
    scenarios["effect_region"] = [
        classify_effect(float(volume), float(depth))
        for volume, depth in zip(
            scenarios["volume_ratio"],
            scenarios["leaf_depth_delta"],
            strict=True,
        )
    ]
    scenarios["shallow_swarm"] = scenarios["effect_region"].eq("shallow_swarm")
    scenarios["evidence_tier"] = "confirmatory_local_uncertainty"
    return scenarios


def summarize_broad_tiers(cells: pd.DataFrame) -> pd.DataFrame:
    return (
        cells.groupby("stress_tier", sort=False)
        .agg(
            n_cells=("scenario", "size"),
            n_scenarios=("scenario", "nunique"),
            n_communities=("community", "nunique"),
            shallow_swarm_share=("shallow_swarm", "mean"),
            median_volume_ratio=("volume_ratio", "median"),
            median_leaf_depth_delta=("leaf_depth_delta", "median"),
            min_volume_ratio=("volume_ratio", "min"),
            max_volume_ratio=("volume_ratio", "max"),
            min_leaf_depth_delta=("leaf_depth_delta", "min"),
            max_leaf_depth_delta=("leaf_depth_delta", "max"),
        )
        .reset_index()
    )


def summarize_factor_blocks(cells: pd.DataFrame) -> pd.DataFrame:
    single = cells[cells["design"].isin(["reference", "one_factor"])].copy()
    return (
        single.groupby(["factor", "stress_tier"], sort=False)
        .agg(
            n_cells=("scenario", "size"),
            n_scenarios=("scenario", "nunique"),
            shallow_swarm_share=("shallow_swarm", "mean"),
            median_volume_ratio=("volume_ratio", "median"),
            median_leaf_depth_delta=("leaf_depth_delta", "median"),
        )
        .reset_index()
    )


def build_mechanism_phase_map(
    root: str | Path,
    config: Mapping[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    project = Path(root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    inputs = config["inputs"]
    broad_path = project / inputs["broad_tradeoff_regions"]
    local_path = project / inputs["local_sensitivity_scenarios"]
    broad = pd.read_csv(broad_path)
    local = pd.read_csv(local_path)

    broad_cells = prepare_broad_phase_cells(
        broad,
        config.get("classification", {}).get("extreme_values", {}),
    )
    local_scenarios = prepare_local_phase_scenarios(local)
    tier_summary = summarize_broad_tiers(broad_cells)
    factor_summary = summarize_factor_blocks(broad_cells)
    factorial_summary = _factorial_summary(broad_cells)

    broad_cells.to_csv(output / "broad_phase_cells.csv", index=False)
    local_scenarios.to_csv(
        output / "confirmatory_local_scenarios.csv",
        index=False,
    )
    tier_summary.to_csv(output / "stress_tier_summary.csv", index=False)
    factor_summary.to_csv(output / "factor_block_summary.csv", index=False)
    factorial_summary.to_csv(output / "factorial_phase_summary.csv", index=False)

    _plot_overview(
        broad_cells,
        local_scenarios,
        output / "mechanism_phase_overview",
    )
    _plot_factor_support(
        factor_summary,
        output / "one_factor_support",
    )
    _plot_factorial_phase(
        factorial_summary,
        output / "trait_ranking_viewport_phase",
    )

    local_support = int(local_scenarios["shallow_swarm"].sum())
    broad_support = int(broad_cells["shallow_swarm"].sum())
    report = _report(
        local_scenarios,
        broad_cells,
        tier_summary,
        factor_summary,
        config,
    )
    (output / "MECHANISM_PHASE_MAP_REPORT.md").write_text(
        report,
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "confirmatory_local": {
            "n_scenarios": int(len(local_scenarios)),
            "supporting_scenarios": local_support,
            "support_share": float(local_scenarios["shallow_swarm"].mean()),
            "unit": "paired scenario summaries",
        },
        "global_stress": {
            "n_cells": int(len(broad_cells)),
            "n_scenarios": int(broad_cells["scenario"].nunique()),
            "supporting_cells": broad_support,
            "support_share": float(broad_cells["shallow_swarm"].mean()),
            "unit": "scenario-community cells",
        },
        "classification": dict(config.get("classification", {})),
        "claim_boundary": dict(config["evidence_scope"]),
        "inputs": {
            "broad_tradeoff_regions": {
                "path": str(broad_path),
                "sha256": sha256_file(broad_path),
            },
            "local_sensitivity_scenarios": {
                "path": str(local_path),
                "sha256": sha256_file(local_path),
            },
        },
        "outputs": {
            path.name: sha256_file(path)
            for path in sorted(output.iterdir())
            if path.is_file() and path.name != "mechanism_phase_map_manifest.json"
        },
    }
    write_json(output / "mechanism_phase_map_manifest.json", manifest)
    return manifest


def _factorial_summary(cells: pd.DataFrame) -> pd.DataFrame:
    factorial = cells[cells["design"].eq("factorial")].copy()
    if factorial.empty:
        return pd.DataFrame(
            columns=[
                "trait",
                "ranking",
                "viewport",
                "n_communities",
                "shallow_swarm_share",
                "median_volume_ratio",
                "median_leaf_depth_delta",
            ]
        )
    return (
        factorial.groupby(["trait", "ranking", "viewport"], sort=False)
        .agg(
            n_communities=("community", "nunique"),
            shallow_swarm_share=("shallow_swarm", "mean"),
            median_volume_ratio=("volume_ratio", "median"),
            median_leaf_depth_delta=("leaf_depth_delta", "median"),
        )
        .reset_index()
    )


def _plot_overview(
    broad: pd.DataFrame,
    local: pd.DataFrame,
    output_base: Path,
) -> None:
    colors = {
        "reference": "#222222",
        "bounded_one_factor": "#2f6b9a",
        "boundary_one_factor": "#d17a22",
        "multifactor_stress": "#6b8e23",
        "multifactor_boundary": "#9c3d54",
    }
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for tier, group in broad.groupby("stress_tier", sort=False):
        axes[0].scatter(
            group["volume_ratio"],
            group["leaf_depth_delta"],
            s=28,
            alpha=0.65,
            color=colors.get(tier, "#777777"),
            label=tier.replace("_", " "),
        )
    _phase_axes(axes[0], "Global stress boundary map")
    axes[0].legend(fontsize=7, loc="best")

    axes[1].scatter(
        local["volume_ratio"],
        local["leaf_depth_delta"],
        s=48,
        color="#176b3a",
        alpha=0.85,
    )
    _phase_axes(axes[1], "Confirmatory local uncertainty")
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output_base.with_suffix(f".{suffix}"), dpi=220)
    plt.close(figure)


def _phase_axes(ax: Any, title: str) -> None:
    ax.axvline(1.0, color="#777777", linestyle="--", linewidth=1)
    ax.axhline(0.0, color="#777777", linestyle="--", linewidth=1)
    ax.set_xlabel("Conflict-to-baseline comment-volume ratio")
    ax.set_ylabel("Change in mean leaf depth")
    ax.set_title(title)
    ax.grid(alpha=0.15)


def _plot_factor_support(summary: pd.DataFrame, output_base: Path) -> None:
    ordered = summary.sort_values(
        ["shallow_swarm_share", "factor"],
        ascending=[True, True],
    )
    labels = [
        (
            str(row.factor).replace("_", " ")
            if row.stress_tier == "bounded_one_factor"
            else f"{str(row.factor).replace('_', ' ')} ({str(row.stress_tier).replace('_', ' ')})"
        )
        for row in ordered.itertuples(index=False)
    ]
    colors = [
        "#2f6b9a" if tier == "bounded_one_factor" else "#d17a22"
        for tier in ordered["stress_tier"]
    ]
    figure, ax = plt.subplots(figsize=(10, max(5.5, 0.42 * len(ordered))))
    ax.barh(labels, ordered["shallow_swarm_share"], color=colors)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Share of scenario-community cells in Shallow Swarm region")
    ax.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output_base.with_suffix(f".{suffix}"), dpi=220)
    plt.close(figure)


def _plot_factorial_phase(summary: pd.DataFrame, output_base: Path) -> None:
    if summary.empty:
        return
    traits = list(dict.fromkeys(summary["trait"].astype(str)))
    rankings = list(dict.fromkeys(summary["ranking"].astype(str)))
    viewports = list(dict.fromkeys(summary["viewport"].astype(str)))
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), squeeze=False)
    for ax, trait in zip(axes.flat, traits, strict=False):
        subset = summary[summary["trait"].eq(trait)]
        pivot = subset.pivot(
            index="ranking",
            columns="viewport",
            values="shallow_swarm_share",
        ).reindex(index=rankings, columns=viewports)
        image = ax.imshow(
            pivot.to_numpy(dtype=float),
            vmin=0.0,
            vmax=1.0,
            cmap="viridis",
            aspect="auto",
        )
        ax.set_xticks(range(len(viewports)), viewports)
        ax.set_yticks(range(len(rankings)), rankings)
        ax.set_xlabel("Viewport")
        ax.set_ylabel("Ranking")
        ax.set_title(f"Trait: {trait}")
        for row_index in range(len(rankings)):
            for column_index in range(len(viewports)):
                value = pivot.iloc[row_index, column_index]
                if pd.notna(value):
                    ax.text(
                        column_index,
                        row_index,
                        f"{100 * value:.0f}%",
                        ha="center",
                        va="center",
                        color="white" if value < 0.55 else "black",
                        fontsize=8,
                    )
    figure.colorbar(
        image,
        ax=axes.ravel().tolist(),
        label="Shallow Swarm share across communities",
        shrink=0.82,
    )
    figure.subplots_adjust(
        left=0.08,
        right=0.9,
        bottom=0.08,
        top=0.94,
        wspace=0.28,
        hspace=0.3,
    )
    for suffix in ("png", "pdf"):
        figure.savefig(output_base.with_suffix(f".{suffix}"), dpi=220)
    plt.close(figure)


def _report(
    local: pd.DataFrame,
    broad: pd.DataFrame,
    tiers: pd.DataFrame,
    factors: pd.DataFrame,
    config: Mapping[str, Any],
) -> str:
    tier_lines = [
        "| Tier | Cells | Scenarios | Shallow Swarm | Median volume ratio | Median leaf-depth change |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in tiers.itertuples(index=False):
        tier_lines.append(
            f"| {row.stress_tier} | {row.n_cells} | {row.n_scenarios} | "
            f"{100 * row.shallow_swarm_share:.1f}% | "
            f"{row.median_volume_ratio:.3f} | "
            f"{row.median_leaf_depth_delta:+.3f} |"
        )
    factor_lines = [
        "| Factor block | Tier | Cells | Shallow Swarm | Median volume ratio | Median leaf-depth change |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in factors.itertuples(index=False):
        factor_lines.append(
            f"| {row.factor} | {row.stress_tier} | {row.n_cells} | "
            f"{100 * row.shallow_swarm_share:.1f}% | "
            f"{row.median_volume_ratio:.3f} | "
            f"{row.median_leaf_depth_delta:+.3f} |"
        )
    allowed = config["evidence_scope"]["allowed_claim"]
    forbidden = config["evidence_scope"]["forbidden_claim"]
    return "\n".join(
        [
            "# Mechanism Phase Map",
            "",
            "## Confirmatory Local Region",
            "",
            f"- Scenarios: {len(local)}",
            f"- Supporting scenarios: {int(local['shallow_swarm'].sum())}/{len(local)}",
            (
                "- Volume-ratio range: "
                f"{local['volume_ratio'].min():.3f} to "
                f"{local['volume_ratio'].max():.3f}"
            ),
            (
                "- Mean leaf-depth change: "
                f"{local['leaf_depth_delta'].min():+.3f} to "
                f"{local['leaf_depth_delta'].max():+.3f}"
            ),
            "",
            "This tier is the frozen local uncertainty analysis. It is not an "
            "independent empirical parameter-identification result.",
            "",
            "## Global Stress Boundary Map",
            "",
            f"- Scenario-community cells: {len(broad)}",
            f"- Distinct scenarios: {broad['scenario'].nunique()}",
            f"- Shallow Swarm cells: {int(broad['shallow_swarm'].sum())}/{len(broad)}",
            "",
            *tier_lines,
            "",
            "## One-Factor Blocks",
            "",
            *factor_lines,
            "",
            "## Interpretation",
            "",
            "The global stress map is a phase-boundary analysis, not an "
            "unweighted estimate of how often the original mechanism is true. "
            "Its role is to identify regions where the volume-depth tradeoff "
            "appears, disappears, or reverses.",
            "",
            f"Allowed claim: {allowed}",
            "",
            f"Forbidden claim: {forbidden}",
            "",
        ]
    )
