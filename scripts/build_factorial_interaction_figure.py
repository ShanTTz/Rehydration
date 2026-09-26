from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


CELL_ORDER = (
    "baseline_best",
    "baseline_controversial",
    "conflict_best",
    "conflict_controversial",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cluster_bootstrap(
    frame: pd.DataFrame,
    value_columns: list[str],
    samples: int,
    seed: int,
) -> dict[str, tuple[float, float]]:
    grouped = [group[value_columns].to_numpy(dtype=float) for _, group in frame.groupby("cluster", sort=True)]
    rng = np.random.default_rng(seed)
    draws = np.empty((samples, len(value_columns)), dtype=float)
    for index in range(samples):
        selected = rng.integers(0, len(grouped), size=len(grouped))
        values = np.concatenate([grouped[item] for item in selected], axis=0)
        draws[index] = np.nanmean(values, axis=0)
    return {
        column: (float(np.quantile(draws[:, idx], 0.025)), float(np.quantile(draws[:, idx], 0.975)))
        for idx, column in enumerate(value_columns)
    }


def _volume_cells(
    runs: pd.DataFrame,
    cells: dict[str, str],
    samples: int,
    seed: int,
) -> pd.DataFrame:
    pivot = runs.pivot_table(
        index=["community", "post_id", "seed"],
        columns="condition",
        values="comment_volume",
        aggfunc="first",
    ).dropna(subset=list(cells.values()))
    base = np.log1p(pivot[cells["baseline_best"]].astype(float))
    effects = pd.DataFrame(index=pivot.index)
    for name in CELL_ORDER:
        effects[name] = np.log1p(pivot[cells[name]].astype(float)) - base
    effects = effects.reset_index()
    effects["cluster"] = effects["community"].astype(str) + "/" + effects["post_id"].astype(str)
    intervals = _cluster_bootstrap(effects, list(CELL_ORDER), samples, seed)
    records = []
    for name in CELL_ORDER:
        estimate_log = float(effects[name].mean())
        low_log, high_log = intervals[name]
        records.append(
            {
                "panel": "controlled_volume_ratio",
                "cell": name,
                "estimate": float(np.exp(estimate_log)),
                "ci_low": float(np.exp(low_log)),
                "ci_high": float(np.exp(high_log)),
                "n_blocks": int(len(effects)),
                "n_posts": int(effects["cluster"].nunique()),
            }
        )
    return pd.DataFrame(records)


def _depth_cells(
    runs: pd.DataFrame,
    empirical: pd.DataFrame,
    cells: dict[str, str],
    development_splits: list[str],
    test_split: str,
    samples: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    split_columns = empirical[["community", "post_id", "split"]].drop_duplicates()
    merged = runs.merge(split_columns, on=["community", "post_id"], how="inner", validate="many_to_one")
    baseline = merged[merged["condition"].eq(cells["baseline_best"])].copy()
    baseline_post = (
        baseline.groupby(["community", "post_id", "split"], as_index=False)["mean_leaf_depth"]
        .mean()
        .rename(columns={"mean_leaf_depth": "simulated_baseline_depth"})
    )
    development_real = empirical[empirical["split"].isin(development_splits)][
        ["community", "post_id", "split", "mean_leaf_depth"]
    ].rename(columns={"mean_leaf_depth": "real_depth"})
    development = baseline_post[baseline_post["split"].isin(development_splits)].merge(
        development_real,
        on=["community", "post_id", "split"],
        how="inner",
        validate="one_to_one",
    )
    scales = (
        development.groupby("community", as_index=False)
        .agg(
            simulated_development_depth=("simulated_baseline_depth", "mean"),
            real_development_depth=("real_depth", "mean"),
            n_development_posts=("post_id", "nunique"),
        )
    )
    scales["depth_scale"] = scales["real_development_depth"] / scales["simulated_development_depth"]

    test = merged[merged["split"].eq(test_split)].merge(
        scales[["community", "depth_scale"]], on="community", how="inner", validate="many_to_one"
    )
    test["calibrated_leaf_depth"] = test["mean_leaf_depth"] * test["depth_scale"]
    pivot = test.pivot_table(
        index=["community", "post_id", "seed"],
        columns="condition",
        values="calibrated_leaf_depth",
        aggfunc="first",
    ).dropna(subset=list(cells.values()))
    values = pd.DataFrame(index=pivot.index)
    for name in CELL_ORDER:
        values[name] = pivot[cells[name]].astype(float)
    values = values.reset_index()
    values["cluster"] = values["community"].astype(str) + "/" + values["post_id"].astype(str)
    intervals = _cluster_bootstrap(values, list(CELL_ORDER), samples, seed + 1)

    records = []
    for name in CELL_ORDER:
        low, high = intervals[name]
        records.append(
            {
                "panel": "heldout_empirical_scale_depth",
                "cell": name,
                "estimate": float(values[name].mean()),
                "ci_low": low,
                "ci_high": high,
                "n_blocks": int(len(values)),
                "n_posts": int(values["cluster"].nunique()),
            }
        )
    cell_frame = pd.DataFrame(records)

    contrast_frame = values[["cluster", *CELL_ORDER]].copy()
    contrast_frame["joint"] = (
        contrast_frame["conflict_controversial"] - contrast_frame["baseline_best"]
    )
    contrast_frame["ranking_baseline"] = (
        contrast_frame["baseline_controversial"] - contrast_frame["baseline_best"]
    )
    contrast_frame["ranking_conflict"] = (
        contrast_frame["conflict_controversial"] - contrast_frame["conflict_best"]
    )
    contrast_frame["interaction"] = (
        contrast_frame["ranking_conflict"] - contrast_frame["ranking_baseline"]
    )
    contrast_names = ["joint", "ranking_baseline", "ranking_conflict", "interaction"]
    contrast_intervals = _cluster_bootstrap(contrast_frame, contrast_names, samples, seed + 2)
    contrasts = {
        name: {
            "estimate": float(contrast_frame[name].mean()),
            "ci_low": contrast_intervals[name][0],
            "ci_high": contrast_intervals[name][1],
        }
        for name in contrast_names
    }
    observed_test = empirical[empirical["split"].eq(test_split)]["mean_leaf_depth"].astype(float)
    audit = {
        "development_posts": int(development["post_id"].nunique()),
        "test_posts": int(values["cluster"].nunique()),
        "test_blocks": int(len(values)),
        "observed_test_mean_leaf_depth": float(observed_test.mean()),
        "calibrated_test_baseline_leaf_depth": float(values["baseline_best"].mean()),
        "baseline_mean_absolute_gap": float(abs(values["baseline_best"].mean() - observed_test.mean())),
        "baseline_mean_relative_gap": float(abs(values["baseline_best"].mean() - observed_test.mean()) / observed_test.mean()),
        "contrasts": contrasts,
    }
    return cell_frame, scales, audit


def _plot(cells: pd.DataFrame, observed_depth: float, output_pdf: Path, output_png: Path) -> None:
    volume = cells[cells["panel"].eq("controlled_volume_ratio")].set_index("cell")
    depth = cells[cells["panel"].eq("heldout_empirical_scale_depth")].set_index("cell")
    colors = {"baseline": "#287271", "conflict": "#D1495B"}
    markers = {"baseline": "o", "conflict": "s"}
    x = np.asarray([0.0, 1.0])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), constrained_layout=True)

    for policy, names in {
        "baseline": ["baseline_best", "baseline_controversial"],
        "conflict": ["conflict_best", "conflict_controversial"],
    }.items():
        panel = volume.loc[names]
        values = panel["estimate"].to_numpy(dtype=float)
        errors = np.vstack(
            [values - panel["ci_low"].to_numpy(dtype=float), panel["ci_high"].to_numpy(dtype=float) - values]
        )
        axes[0].errorbar(
            x,
            values,
            yerr=errors,
            color=colors[policy],
            marker=markers[policy],
            linewidth=2.0,
            markersize=6.5,
            capsize=3,
            label="Baseline behavioral policy" if policy == "baseline" else "Conflict-responsive policy",
        )
        for xpos, value in zip(x, values):
            axes[0].annotate(f"{value:.3f}x", (xpos, value), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)

    for policy, names in {
        "baseline": ["baseline_best", "baseline_controversial"],
        "conflict": ["conflict_best", "conflict_controversial"],
    }.items():
        panel = depth.loc[names]
        values = panel["estimate"].to_numpy(dtype=float)
        errors = np.vstack(
            [values - panel["ci_low"].to_numpy(dtype=float), panel["ci_high"].to_numpy(dtype=float) - values]
        )
        axes[1].errorbar(
            x,
            values,
            yerr=errors,
            color=colors[policy],
            marker=markers[policy],
            linewidth=2.0,
            markersize=6.5,
            capsize=3,
        )
        for xpos, value in zip(x, values):
            axes[1].annotate(f"{value:.2f}", (xpos, value), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)

    axes[1].axhline(observed_depth, color="#555555", linestyle="--", linewidth=1.1)
    axes[1].text(0.5, observed_depth - 0.07, f"Observed test mean = {observed_depth:.2f}", ha="center", va="top", fontsize=7.5, color="#444444")
    axes[0].set_title("(a) Controlled volume interaction", fontsize=10, fontweight="bold")
    axes[1].set_title("(b) Held-out depth on empirical scale", fontsize=10, fontweight="bold")
    axes[0].set_ylabel("Comment volume (ratio to baseline)")
    axes[1].set_ylabel("Mean leaf depth (calibrated units)")
    for axis in axes:
        axis.set_xticks(x, ["Best", "Controversial"])
        axis.set_xlabel("Platform ranking policy")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.7)
        axis.set_xlim(-0.18, 1.18)
    axes[0].set_ylim(0.75, 3.85)
    axes[1].set_ylim(0.35, 2.65)
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.06), ncol=2, frameon=False, fontsize=8.5)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _write_macros(path: Path, cells: pd.DataFrame, audit: dict[str, float]) -> None:
    volume = cells[cells["panel"].eq("controlled_volume_ratio")].set_index("cell")
    depth = cells[cells["panel"].eq("heldout_empirical_scale_depth")].set_index("cell")
    joint = audit["contrasts"]["joint"]
    interaction = audit["contrasts"]["interaction"]
    text = "\n".join(
        [
            "% Auto-generated by build_factorial_interaction_figure.py.",
            rf"\providecommand{{\FactorialPlotBaselineRankingVolume}}{{{volume.loc['baseline_controversial', 'estimate']:.3f}}}",
            rf"\providecommand{{\FactorialPlotConflictBestVolume}}{{{volume.loc['conflict_best', 'estimate']:.3f}}}",
            rf"\providecommand{{\FactorialPlotJointVolume}}{{{volume.loc['conflict_controversial', 'estimate']:.3f}}}",
            rf"\providecommand{{\CalibratedDevelopmentPosts}}{{{audit['development_posts']}}}",
            rf"\providecommand{{\CalibratedTestPosts}}{{{audit['test_posts']}}}",
            rf"\providecommand{{\CalibratedTestBlocks}}{{{audit['test_blocks']}}}",
            rf"\providecommand{{\ObservedTestLeafDepth}}{{{audit['observed_test_mean_leaf_depth']:.3f}}}",
            rf"\providecommand{{\CalibratedBaselineLeafDepth}}{{{depth.loc['baseline_best', 'estimate']:.3f}}}",
            rf"\providecommand{{\CalibratedJointLeafDepth}}{{{depth.loc['conflict_controversial', 'estimate']:.3f}}}",
            rf"\providecommand{{\CalibratedJointDepthChange}}{{{joint['estimate']:.3f}}}",
            rf"\providecommand{{\CalibratedJointDepthLow}}{{{joint['ci_low']:.3f}}}",
            rf"\providecommand{{\CalibratedJointDepthHigh}}{{{joint['ci_high']:.3f}}}",
            rf"\providecommand{{\CalibratedDepthInteraction}}{{{interaction['estimate']:.3f}}}",
            rf"\providecommand{{\CalibratedDepthInteractionLow}}{{{interaction['ci_low']:.3f}}}",
            rf"\providecommand{{\CalibratedDepthInteractionHigh}}{{{interaction['ci_high']:.3f}}}",
            rf"\providecommand{{\CalibratedBaselineRelativeGap}}{{{100.0 * audit['baseline_mean_relative_gap']:.1f}\%}}",
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path, default=Path("configs/factorial_interaction_real_scale.json"))
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    runs_path = root / config["inputs"]["factorial_runs"]
    empirical_path = root / config["inputs"]["empirical_metrics"]
    runs = pd.read_json(runs_path, lines=True)
    empirical = pd.read_csv(empirical_path)
    samples = int(config["bootstrap_samples"])
    seed = int(config["seed"])
    cells = config["factorial_cells"]

    volume = _volume_cells(runs, cells, samples, seed)
    depth, scales, audit = _depth_cells(
        runs,
        empirical,
        cells,
        list(config["real_scale_calibration"]["development_splits"]),
        str(config["real_scale_calibration"]["test_split"]),
        samples,
        seed,
    )
    combined = pd.concat([volume, depth], ignore_index=True)
    output = root / "artifacts/reviewer_validation/factorial_interaction_real_scale"
    staging = root / "manuscript/iclr2026_revision_staging/generated"
    output.mkdir(parents=True, exist_ok=True)
    staging.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output / "factorial_interaction_cells.csv", index=False)
    scales.to_csv(output / "community_depth_scales.csv", index=False)

    summary = {
        "schema_version": 1,
        "status": "complete",
        "controlled_volume": volume.to_dict(orient="records"),
        "heldout_empirical_scale_depth": depth.to_dict(orient="records"),
        "calibration_audit": audit,
        "evidence_scope": config["evidence_scope"],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _plot(
        combined,
        float(audit["observed_test_mean_leaf_depth"]),
        output / "factorial_interaction_real_scale.pdf",
        output / "factorial_interaction_real_scale.png",
    )
    _plot(
        combined,
        float(audit["observed_test_mean_leaf_depth"]),
        staging / "factorial_interaction_real_scale.pdf",
        staging / "factorial_interaction_real_scale.png",
    )
    _write_macros(staging / "factorial_interaction_real_scale_macros.tex", combined, audit)

    manifest = {
        "schema_version": 1,
        "config": str(config_path.relative_to(root)),
        "config_sha256": _sha256(config_path),
        "inputs": {
            str(runs_path.relative_to(root)): _sha256(runs_path),
            str(empirical_path.relative_to(root)): _sha256(empirical_path),
        },
        "outputs": {
            path.name: _sha256(path)
            for path in sorted(output.iterdir())
            if path.is_file() and path.name != "manifest.json"
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "audit": audit}, indent=2))


if __name__ == "__main__":
    main()
