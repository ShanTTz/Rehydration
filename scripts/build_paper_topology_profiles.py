from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


def _pad(values: list[float], length: int) -> np.ndarray:
    result = np.zeros(length, dtype=float)
    result[: len(values)] = np.asarray(values, dtype=float)
    return result


def build(root: Path) -> None:
    result_dir = (
        root
        / "artifacts"
        / "reviewer_validation"
        / "topology_fidelity_learned_interaction"
    )
    summary = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
    stage_a = summary["stage_a"]
    length = max(
        len(stage_a["real_depth_profile"]),
        len(stage_a["simulated_depth_profile"]),
    )
    real = _pad(stage_a["real_depth_profile"], length)
    learned = _pad(stage_a["simulated_depth_profile"], length)
    depths = np.arange(1, length + 1)
    observed_color = "#202020"
    learned_color = "#167D8D"

    figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.0), constrained_layout=True)
    axes[0].step(
        depths,
        np.cumsum(real),
        where="post",
        label="Observed Reddit",
        color=observed_color,
        linewidth=1.8,
    )
    axes[0].step(
        depths,
        np.cumsum(learned),
        where="post",
        label="Learned BDMTF",
        color=learned_color,
        linewidth=1.8,
    )
    axes[0].set_xlabel("Reply depth")
    axes[0].set_ylabel("Cumulative node share")
    axes[0].set_ylim(0.0, 1.02)
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].plot(
        depths,
        real,
        marker="o",
        markersize=3,
        label="Observed Reddit",
        color=observed_color,
        linewidth=1.5,
    )
    axes[1].plot(
        depths,
        learned,
        marker="s",
        markersize=3,
        label="Learned BDMTF",
        color=learned_color,
        linewidth=1.5,
    )
    axes[1].set_xlabel("Reply depth")
    axes[1].set_ylabel("Mean node share")
    axes[1].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(color="#DDDDDD", linewidth=0.6)

    output_pdf = result_dir / "topology_profiles_paper.pdf"
    output_png = result_dir / "topology_profiles_paper.png"
    figure.savefig(output_pdf)
    figure.savefig(output_png, dpi=220)
    plt.close(figure)

    staging = root / "manuscript" / "iclr2026_revision_staging" / "generated"
    staging.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output_pdf, staging / "topology_profiles.pdf")
    shutil.copy2(output_png, staging / "topology_profiles.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    build(args.root.resolve())


if __name__ == "__main__":
    main()
