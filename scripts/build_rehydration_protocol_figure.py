from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    ROOT
    / "manuscript"
    / "iclr2026_overleaf_package_next_revision_20260819"
    / "generated"
    / "rehydration_protocol.pdf"
)


def box(
    ax,
    x,
    y,
    w,
    h,
    title,
    subtitle,
    face,
    edge="#263238",
    title_size=9,
    subtitle_size=7.2,
):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.2,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h * 0.64,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        weight="bold",
    )
    ax.text(
        x + w / 2,
        y + h * 0.30,
        subtitle,
        ha="center",
        va="center",
        fontsize=subtitle_size,
        color="#37474f",
    )


def arrow(ax, start, end, color="#455a64", style="-"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.25,
            linestyle=style,
            color=color,
            shrinkA=2,
            shrinkB=2,
        )
    )


def main() -> None:
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "DejaVu Sans",
            "axes.linewidth": 0,
        }
    )
    fig, ax = plt.subplots(figsize=(7.15, 2.45))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    box(ax, 0.02, 0.39, 0.15, 0.23, "Initial context", "$X$: post, agents,\nseed thread", "#e8f1f8")
    box(ax, 0.22, 0.39, 0.17, 0.23, "Condition-blind\ngenerator", "$Q_\\theta(Z\\mid X)$", "#fff3d9")
    box(ax, 0.44, 0.39, 0.16, 0.23, "Frozen pool", "abstract frames $Z$\n+ sealed hash", "#e8f5e9")

    arrow(ax, (0.17, 0.505), (0.22, 0.505))
    arrow(ax, (0.39, 0.505), (0.44, 0.505))

    ax.plot([0.625, 0.625], [0.17, 0.83], color="#607d8b", linewidth=1.0)
    ax.text(0.625, 0.91, "Policy assignment", ha="center", va="center", fontsize=7.5, color="#455a64")

    box(ax, 0.66, 0.64, 0.18, 0.21, "Control replay", "$M(\\pi_0, Z, \\epsilon)$", "#e3f2fd")
    box(ax, 0.66, 0.15, 0.18, 0.21, "Treatment replay", "$M(\\pi_1, Z, \\epsilon)$", "#fce8e6")
    box(
        ax,
        0.875,
        0.39,
        0.105,
        0.23,
        "Paired effect",
        "$M_1-M_0$",
        "#f3e5f5",
        title_size=7.6,
    )

    arrow(ax, (0.60, 0.54), (0.66, 0.74), color="#1976d2")
    arrow(ax, (0.60, 0.47), (0.66, 0.26), color="#c62828")
    arrow(ax, (0.84, 0.74), (0.875, 0.54), color="#1976d2")
    arrow(ax, (0.84, 0.26), (0.875, 0.47), color="#c62828")

    ax.text(0.50, 0.26, "same $Z$", ha="center", va="center", fontsize=7.5, color="#2e7d32", weight="bold")
    ax.text(0.50, 0.18, "keyed common random numbers", ha="center", va="center", fontsize=6.8, color="#455a64")
    ax.text(0.50, 0.76, "no Phase-II LLM calls", ha="center", va="center", fontsize=7.5, color="#2e7d32", weight="bold")

    ax.annotate(
        "Information barrier",
        xy=(0.625, 0.50),
        xytext=(0.625, 0.06),
        ha="center",
        va="center",
        fontsize=7.3,
        color="#37474f",
        arrowprops=dict(arrowstyle="-[,widthB=3.0,lengthB=0.6", lw=1.0, color="#607d8b"),
    )

    fig.tight_layout(pad=0.15)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    main()
