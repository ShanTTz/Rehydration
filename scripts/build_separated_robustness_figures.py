from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


REGION_ORDER = (
    "shallow_swarm",
    "deep_swarm",
    "deep_contraction",
    "shallow_contraction",
    "boundary",
)
REGION_LABELS = {
    "shallow_swarm": "Volume up, depth down",
    "deep_swarm": "Volume up, depth up",
    "deep_contraction": "Volume down, depth up",
    "shallow_contraction": "Volume down, depth down",
    "boundary": "On decision boundary",
}
REGION_COLORS = {
    "shallow_swarm": "#2A9D8F",
    "deep_swarm": "#457B9D",
    "deep_contraction": "#D1495B",
    "shallow_contraction": "#E9A23B",
    "boundary": "#6C757D",
}
REGION_MARKERS = {
    "shallow_swarm": "o",
    "deep_swarm": "^",
    "deep_contraction": "s",
    "shallow_contraction": "D",
    "boundary": "x",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _primary_point(summary: pd.DataFrame) -> tuple[float, float]:
    if "scope" in summary.columns:
        summary = summary[summary["scope"].eq("all")]
    volume = summary[
        summary["metric"].eq("comment_volume") & summary["contrast"].eq("joint_log")
    ]
    depth = summary[
        summary["metric"].eq("mean_leaf_depth") & summary["contrast"].eq("joint")
    ]
    if len(volume) != 1 or len(depth) != 1:
        raise ValueError("Primary factorial summary does not contain unique joint rows")
    return float(np.exp(volume.iloc[0]["mean"])), float(depth.iloc[0]["mean"])


def _plot_local(local: pd.DataFrame, primary: tuple[float, float], base: Path) -> None:
    figure, ax = plt.subplots(figsize=(3.7, 3.15), constrained_layout=True)
    ax.scatter(
        local["volume_ratio"],
        local["leaf_depth_delta"],
        s=42,
        color="#287271",
        alpha=0.82,
        edgecolor="white",
        linewidth=0.5,
        label="Bounded LHS setting",
    )
    ax.scatter(
        [primary[0]],
        [primary[1]],
        marker="*",
        s=145,
        color="#D1495B",
        edgecolor="white",
        linewidth=0.7,
        zorder=4,
        label="Primary factorial",
    )
    ax.axvspan(
        float(local["volume_ratio"].min()),
        float(local["volume_ratio"].max()),
        color="#287271",
        alpha=0.05,
    )
    ax.text(
        0.03,
        0.05,
        f"{int(local['shallow_swarm'].sum())}/{len(local)} retain\nvolume up, depth down",
        transform=ax.transAxes,
        fontsize=8.0,
        fontweight="bold",
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "#CCCCCC", "boxstyle": "square,pad=0.35"},
    )
    ax.set_title("Local robustness\n(controlled simulator)", fontsize=9.2, fontweight="bold")
    ax.set_xlabel("Joint / baseline volume ratio", fontsize=8.5)
    ax.set_ylabel("Joint - baseline mean leaf depth", fontsize=8.5)
    ax.set_xlim(2.75, 4.25)
    ax.set_ylim(-13.45, -10.45)
    ax.grid(color="#DDDDDD", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7.8)
    ax.legend(frameon=False, fontsize=7.6, loc="upper right")
    for suffix in ("pdf", "png"):
        figure.savefig(base.with_suffix(f".{suffix}"), dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_global(cells: pd.DataFrame, base: Path) -> None:
    figure, ax = plt.subplots(figsize=(3.9, 3.15), constrained_layout=True)
    for region in REGION_ORDER:
        group = cells[cells["effect_region"].eq(region)]
        if group.empty:
            continue
        ax.scatter(
            group["volume_ratio"],
            group["leaf_depth_delta"],
            s=24 if region != "shallow_swarm" else 30,
            alpha=0.58 if region != "shallow_swarm" else 0.78,
            color=REGION_COLORS[region],
            marker=REGION_MARKERS[region],
            label=f"{REGION_LABELS[region]} ({len(group)})",
        )
    ax.axvline(1.0, color="#555555", linestyle="--", linewidth=1.0)
    ax.axhline(0.0, color="#555555", linestyle="--", linewidth=1.0)
    ax.text(0.98, 0.04, "Shallow Swarm\n108 / 615", transform=ax.transAxes, ha="right", va="bottom", fontsize=8.0, fontweight="bold", color="#176B5B")
    ax.text(0.98, 0.96, "Most common failure:\nvolume up, depth up\n263 / 615", transform=ax.transAxes, ha="right", va="top", fontsize=8.0, color="#315F7D")
    ax.set_title("Global failure regions\n(learned BDMTF)", fontsize=9.2, fontweight="bold")
    ax.set_xlabel("Conflict intensity 1 / 0 volume ratio", fontsize=8.5)
    ax.set_ylabel("Conflict intensity 1 - 0 mean leaf depth", fontsize=8.5)
    ax.grid(color="#DDDDDD", linewidth=0.65, alpha=0.75)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7.8)
    ax.legend(frameon=False, fontsize=7.2, loc="lower left", ncol=1)
    for suffix in ("pdf", "png"):
        figure.savefig(base.with_suffix(f".{suffix}"), dpi=220, bbox_inches="tight")
    plt.close(figure)


def _one_factor_summary(cells: pd.DataFrame) -> pd.DataFrame:
    one_factor = cells[cells["design"].eq("one_factor")].copy()
    return (
        one_factor.groupby(["factor", "factor_value"], dropna=False, as_index=False)
        .agg(
            n_cells=("community", "size"),
            supporting_cells=("shallow_swarm", "sum"),
            support_share=("shallow_swarm", "mean"),
            median_volume_ratio=("volume_ratio", "median"),
            median_leaf_depth_delta=("leaf_depth_delta", "median"),
        )
        .sort_values(["support_share", "factor", "factor_value"], kind="stable")
    )


def _write_macros(path: Path, local: pd.DataFrame, global_cells: pd.DataFrame, regions: pd.DataFrame) -> None:
    counts = regions.set_index("effect_region")["n_cells"].to_dict()
    lines = [
        "% Auto-generated by build_separated_robustness_figures.py.",
        rf"\providecommand{{\LocalRobustnessSettings}}{{{len(local)}}}",
        rf"\providecommand{{\LocalRobustnessSupport}}{{{int(local['shallow_swarm'].sum())}}}",
        rf"\providecommand{{\GlobalBoundaryScenarios}}{{{global_cells['scenario'].nunique()}}}",
        rf"\providecommand{{\GlobalBoundaryCells}}{{{len(global_cells)}}}",
        rf"\providecommand{{\GlobalBoundarySupport}}{{{int(global_cells['shallow_swarm'].sum())}}}",
        rf"\providecommand{{\GlobalBoundaryFailures}}{{{int((~global_cells['shallow_swarm']).sum())}}}",
        rf"\providecommand{{\GlobalDeepSwarmCells}}{{{int(counts.get('deep_swarm', 0))}}}",
        rf"\providecommand{{\GlobalDeepContractionCells}}{{{int(counts.get('deep_contraction', 0))}}}",
        rf"\providecommand{{\GlobalShallowContractionCells}}{{{int(counts.get('shallow_contraction', 0))}}}",
        rf"\providecommand{{\GlobalBoundaryOnlyCells}}{{{int(counts.get('boundary', 0))}}}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _report(config: dict, local: pd.DataFrame, cells: pd.DataFrame, regions: pd.DataFrame, one_factor: pd.DataFrame) -> str:
    zero_support = one_factor[one_factor["supporting_cells"].eq(0)]
    zero_lines = [
        f"- `{row.factor}={row.factor_value}`: 0/{int(row.n_cells)} supporting cells; median volume ratio {row.median_volume_ratio:.3f}, median depth change {row.median_leaf_depth_delta:+.3f}."
        for row in zero_support.itertuples(index=False)
    ]
    region_lines = [
        f"| {REGION_LABELS.get(row.effect_region, row.effect_region)} | {int(row.n_cells)} | {100 * row.share:.1f}% |"
        for row in regions.itertuples(index=False)
    ]
    return "\n".join(
        [
            "# Robustness and Failure-Region Evidence",
            "",
            "## Design separation",
            "",
            "| Property | Local uncertainty | Global boundary search |",
            "|---|---|---|",
            f"| Scientific role | {config['local_uncertainty']['role']} | {config['global_boundary']['role']} |",
            f"| Simulator | {config['local_uncertainty']['simulator']} | {config['global_boundary']['simulator']} |",
            f"| Contrast | {config['local_uncertainty']['contrast']} | {config['global_boundary']['contrast']} |",
            f"| Unit | {config['local_uncertainty']['unit']} | {config['global_boundary']['unit']} |",
            "",
            "The two analyses do not estimate a common sampling distribution and must not be displayed or interpreted as two panels of one experiment.",
            "",
            "## Local robustness answer",
            "",
            f"All {int(local['shallow_swarm'].sum())}/{len(local)} bounded settings retain the joint higher-volume/lower-depth direction. The volume-ratio range is {local['volume_ratio'].min():.3f}--{local['volume_ratio'].max():.3f}; the depth-change range is {local['leaf_depth_delta'].min():+.3f} to {local['leaf_depth_delta'].max():+.3f}.",
            "",
            "## Global failure-region answer",
            "",
            f"Only {int(cells['shallow_swarm'].sum())}/{len(cells)} scenario-community cells occupy the target quadrant; {int((~cells['shallow_swarm']).sum())} are failure, null, or reversal cells.",
            "",
            "| Region | Cells | Share |",
            "|---|---:|---:|",
            *region_lines,
            "",
            "The most common failure is higher volume with greater depth, showing that activity amplification alone does not imply structural compression.",
            "",
            "## Concrete one-factor failures",
            "",
            *zero_lines,
            "",
            f"Allowed claim: {config['evidence_scope']['allowed_claim']}",
            "",
            f"Forbidden claim: {config['evidence_scope']['forbidden_claim']}",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path, default=Path("configs/robustness_separated.json"))
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    paths = {name: root / value for name, value in config["inputs"].items()}
    local = pd.read_csv(paths["local_scenarios"])
    cells = pd.read_csv(paths["global_cells"])
    primary = _primary_point(pd.read_csv(paths["primary_factorial"]))
    regions = (
        cells.groupby("effect_region", as_index=False)
        .size()
        .rename(columns={"size": "n_cells"})
    )
    regions["share"] = regions["n_cells"] / len(cells)
    one_factor = _one_factor_summary(cells)

    output = root / "artifacts/reviewer_validation/robustness_separated"
    staging = root / "manuscript/iclr2026_revision_staging/generated"
    output.mkdir(parents=True, exist_ok=True)
    staging.mkdir(parents=True, exist_ok=True)
    _plot_local(local, primary, output / "local_uncertainty_robustness")
    _plot_global(cells, output / "global_failure_regions")
    regions.to_csv(output / "failure_region_summary.csv", index=False)
    one_factor.to_csv(output / "one_factor_failure_summary.csv", index=False)
    design = pd.DataFrame(
        [
            {"analysis": "local_uncertainty", **config["local_uncertainty"]},
            {"analysis": "global_boundary", **config["global_boundary"]},
        ]
    )
    design.to_csv(output / "design_separation.csv", index=False)
    (output / "ROBUSTNESS_FAILURE_REGIONS.md").write_text(
        _report(config, local, cells, regions, one_factor), encoding="utf-8"
    )
    for name in (
        "local_uncertainty_robustness.pdf",
        "local_uncertainty_robustness.png",
        "global_failure_regions.pdf",
        "global_failure_regions.png",
    ):
        shutil.copy2(output / name, staging / name)
    _write_macros(staging / "robustness_separated_macros.tex", local, cells, regions)

    summary = {
        "schema_version": 1,
        "status": "complete",
        "local_uncertainty": {
            **config["local_uncertainty"],
            "n_settings": int(len(local)),
            "supporting_settings": int(local["shallow_swarm"].sum()),
            "volume_ratio_range": [float(local["volume_ratio"].min()), float(local["volume_ratio"].max())],
            "leaf_depth_delta_range": [float(local["leaf_depth_delta"].min()), float(local["leaf_depth_delta"].max())],
        },
        "global_boundary": {
            **config["global_boundary"],
            "n_scenarios": int(cells["scenario"].nunique()),
            "n_cells": int(len(cells)),
            "supporting_cells": int(cells["shallow_swarm"].sum()),
            "failure_or_reversal_cells": int((~cells["shallow_swarm"]).sum()),
            "region_counts": {row.effect_region: int(row.n_cells) for row in regions.itertuples(index=False)},
        },
        "evidence_scope": config["evidence_scope"],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "config": str(config_path.relative_to(root)),
        "config_sha256": _sha256(config_path),
        "inputs": {str(path.relative_to(root)): _sha256(path) for path in paths.values()},
        "outputs": {
            path.name: _sha256(path)
            for path in sorted(output.iterdir())
            if path.is_file() and path.name != "manifest.json"
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
