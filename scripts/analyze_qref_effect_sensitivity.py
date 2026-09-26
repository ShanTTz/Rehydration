"""Estimate Q_ref-specific fixed-semantics effects from cached family replays."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STRICT = ROOT / "artifacts" / "api" / "replay_metrics.csv"
DEFAULT_COUPLED = ROOT / "artifacts" / "api" / "coupled_replay_metrics.csv"
DEFAULT_OUTPUT = ROOT / "artifacts" / "reviewer_validation" / "qref_effect_sensitivity"
METRICS = ("log_volume_effect", "leaf_depth_effect", "toxicity_effect")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def analyze(
    strict_path: Path,
    coupled_path: Path,
    output_dir: Path,
    *,
    bootstrap_samples: int = 5_000,
    seed: int = 30371,
) -> dict[str, object]:
    strict = pd.read_csv(strict_path)
    coupled = pd.read_csv(coupled_path)
    keys = ["family", "model", "community", "post_id", "seed"]
    paired = coupled.merge(strict, on=keys, suffixes=("_linked", "_sealed"), validate="one_to_one")
    paired["cluster_id"] = paired["community"].astype(str) + "::" + paired["post_id"].astype(str)
    paired["log_volume_effect"] = np.log1p(paired["size_linked"]) - np.log1p(
        paired["size_sealed"]
    )
    paired["leaf_depth_effect"] = (
        paired["mean_leaf_depth_linked"] - paired["mean_leaf_depth_sealed"]
    )
    paired["toxicity_effect"] = (
        paired["toxicity_density_linked"] - paired["toxicity_density_sealed"]
    )

    clusters = sorted(paired["cluster_id"].unique())
    families = sorted(paired["family"].unique())
    if paired.groupby("cluster_id")["family"].nunique().min() != len(families):
        raise ValueError("every post cluster must contain all Q_ref families")

    rng = np.random.default_rng(seed)
    family_draws = {
        (family, metric): np.empty(bootstrap_samples, dtype=float)
        for family in families
        for metric in METRICS
    }
    range_draws = {
        metric: np.empty(bootstrap_samples, dtype=float) for metric in METRICS
    }
    grouped = {cluster: frame for cluster, frame in paired.groupby("cluster_id", sort=False)}
    for draw in range(bootstrap_samples):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        frame = pd.concat([grouped[cluster] for cluster in sampled], ignore_index=True)
        means = frame.groupby("family")[list(METRICS)].mean()
        for family in families:
            for metric in METRICS:
                family_draws[(family, metric)][draw] = float(means.loc[family, metric])
        for metric in METRICS:
            range_draws[metric][draw] = float(means[metric].max() - means[metric].min())

    rows = []
    family_means = paired.groupby(["family", "model"])[list(METRICS)].mean()
    for (family, model), values in family_means.iterrows():
        for metric in METRICS:
            draws = family_draws[(family, metric)]
            estimate = float(values[metric])
            rows.append(
                {
                    "family": family,
                    "model": model,
                    "metric": metric,
                    "estimate": estimate,
                    "ci_low": float(np.quantile(draws, 0.025)),
                    "ci_high": float(np.quantile(draws, 0.975)),
                    "posts": len(clusters),
                    "display_estimate": float(np.exp(estimate))
                    if metric == "log_volume_effect"
                    else estimate,
                }
            )
    summary_frame = pd.DataFrame(rows)

    ranges = {}
    for metric in METRICS:
        values = family_means[metric]
        draws = range_draws[metric]
        ranges[metric] = {
            "observed": float(values.max() - values.min()),
            "ci_low": float(np.quantile(draws, 0.025)),
            "ci_high": float(np.quantile(draws, 0.975)),
        }
    volume_ratios = np.exp(
        summary_frame[summary_frame["metric"] == "log_volume_effect"]
        .set_index("family")["estimate"]
    )
    depth_effects = summary_frame[
        summary_frame["metric"] == "leaf_depth_effect"
    ].set_index("family")["estimate"]
    toxicity_effects = summary_frame[
        summary_frame["metric"] == "toxicity_effect"
    ].set_index("family")["estimate"]

    manifest = {
        "status": "complete",
        "estimand": (
            "paired effect of enabling the prespecified semantic-linked transition "
            "while holding each model family's frozen Q_ref realization fixed"
        ),
        "families": families,
        "matched_posts": len(clusters),
        "cached_intent_responses": 1500,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_cluster": "community x post_id",
        "family_effect_ranges": ranges,
        "all_family_mean_directions": {
            "volume_positive": bool((volume_ratios > 1.0).all()),
            "leaf_depth_positive": bool((depth_effects > 0.0).all()),
            "toxicity_positive": bool((toxicity_effects > 0.0).all()),
        },
        "scope": (
            "Q_ref sensitivity of the semantic-linked diagnostic policy; not the "
            "controlled Shallow-Swarm factorial"
        ),
        "inputs": {
            str(strict_path): _sha256(strict_path),
            str(coupled_path): _sha256(coupled_path),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    paired.to_csv(output_dir / "paired_qref_effects.csv", index=False)
    summary_frame.to_csv(output_dir / "qref_family_effect_summary.csv", index=False)
    (output_dir / "qref_effect_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    table = summary_frame.pivot(index=["family", "model"], columns="metric")
    lines = [
        "\\begin{tabular}{llrrr}",
        "\\toprule",
        "$Q_{\\mathrm{ref}}$ family & Model & Volume ratio & $\\Delta$ leaf depth & $\\Delta$ toxicity \\\\",
        "\\midrule",
    ]
    for family, model in table.index:
        row = summary_frame[
            (summary_frame["family"] == family) & (summary_frame["model"] == model)
        ].set_index("metric")
        lines.append(
            f"{family.title()} & {model} & "
            f"{np.exp(row.loc['log_volume_effect', 'estimate']):.3f}$\\times$ & "
            f"{row.loc['leaf_depth_effect', 'estimate']:+.3f} & "
            f"{row.loc['toxicity_effect', 'estimate']:+.3f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (output_dir / "table_qref_effect_sensitivity.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    macros = [
        "% Auto-generated by analyze_qref_effect_sensitivity.py.",
        f"\\newcommand{{\\QRefSensitivityPosts}}{{{len(clusters)}}}",
        f"\\newcommand{{\\QRefVolumeMin}}{{{volume_ratios.min():.3f}}}",
        f"\\newcommand{{\\QRefVolumeMax}}{{{volume_ratios.max():.3f}}}",
        f"\\newcommand{{\\QRefDepthMin}}{{{depth_effects.min():+.3f}}}",
        f"\\newcommand{{\\QRefDepthMax}}{{{depth_effects.max():+.3f}}}",
        f"\\newcommand{{\\QRefToxicityMin}}{{{toxicity_effects.min():+.3f}}}",
        f"\\newcommand{{\\QRefToxicityMax}}{{{toxicity_effects.max():+.3f}}}",
    ]
    (output_dir / "qref_effect_macros.tex").write_text(
        "\n".join(macros) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", type=Path, default=DEFAULT_STRICT)
    parser.add_argument("--coupled", type=Path, default=DEFAULT_COUPLED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-samples", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=30371)
    args = parser.parse_args()
    result = analyze(
        args.strict,
        args.coupled,
        args.output_dir,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
