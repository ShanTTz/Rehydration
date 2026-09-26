"""Regenerate imported mechanism figures with embedded TrueType fonts."""
from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd


matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts/reviewer_validation/mechanism_storyline_20260901"
GENERATED = ROOT / "manuscript/iclr2027_overleaf_package_integrated_20260907/generated"


def plot_interaction_grid(summary_path: Path, output_path: Path) -> None:
    summary = pd.read_csv(summary_path)
    scales = sorted(summary["conflict_channel_scale"].unique())
    viewports = sorted(summary["viewport_k"].unique())
    volume = summary.pivot(
        index="conflict_channel_scale",
        columns="viewport_k",
        values="volume_ratio_of_ratios",
    ).loc[scales, viewports]
    depth = summary.pivot(
        index="conflict_channel_scale",
        columns="viewport_k",
        values="depth_interaction",
    ).loc[scales, viewports]
    support = summary.pivot(
        index="conflict_channel_scale",
        columns="viewport_k",
        values="strict_support",
    ).loc[scales, viewports]

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4), constrained_layout=True)
    panels = [
        (axes[0], volume.to_numpy(), "Volume ratio of ratios", "viridis"),
        (axes[1], depth.to_numpy(), "Leaf-depth difference in differences", "coolwarm"),
    ]
    for axis, values, title, cmap in panels:
        image = axis.imshow(values, origin="lower", aspect="auto", cmap=cmap)
        axis.set_xticks(range(len(viewports)), viewports)
        axis.set_yticks(range(len(scales)), [f"{value:g}" for value in scales])
        axis.set_xlabel("Ranking viewport $k$")
        axis.set_ylabel("Conflict-response scale")
        axis.set_title(title)
        for row in range(len(scales)):
            for column in range(len(viewports)):
                marker = "*" if bool(support.iloc[row, column]) else ""
                axis.text(
                    column,
                    row,
                    f"{values[row, column]:.2f}{marker}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if abs(values[row, column]) > 1.0 else "black",
                )
        fig.colorbar(image, ax=axis, shrink=0.8)
    fig.suptitle("Four-cell interaction boundary (* = both 95% intervals exclude null)")
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    jobs = [
        (
            ARTIFACTS / "interaction_boundary_map/boundary_cells.csv",
            GENERATED / "interaction_boundary_map.pdf",
        ),
        (
            ARTIFACTS / "active_regime_confirmation_capacity512/active_regime_cells.csv",
            GENERATED / "active_regime_confirmation.pdf",
        ),
        (
            ARTIFACTS / "core_phase_confirmation_capacity512/active_regime_cells.csv",
            GENERATED / "core_phase_confirmation.pdf",
        ),
    ]
    for source, destination in jobs:
        plot_interaction_grid(source, destination)
        print(destination)


if __name__ == "__main__":
    main()
