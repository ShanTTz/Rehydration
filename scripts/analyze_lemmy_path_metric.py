"""Document and stress-test the held-out Lemmy event-path composite metric."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCORES = (
    ROOT / "artifacts" / "interventions" / "path_fidelity" / "path_draw_scores.csv"
)
DEFAULT_MANIFEST = (
    ROOT / "artifacts" / "interventions" / "path_fidelity" / "path_fidelity_manifest.json"
)
DEFAULT_OUTPUT = (
    ROOT / "artifacts" / "interventions" / "path_fidelity" / "metric_definition"
)
COMPONENTS = (
    "cumulative_count_nmae",
    "arrival_time_nwd",
    "depth_nwd",
    "root_share_ae",
    "parent_hhi_ae",
)
BDMTF = "bdmtf_training_path"
EMPIRICAL = "empirical_nearest_path"


def _weighted_gain(pivot: pd.DataFrame, weights: np.ndarray) -> tuple[float, float, float]:
    bdmtf = sum(weights[index] * pivot[(BDMTF, metric)] for index, metric in enumerate(COMPONENTS))
    empirical = sum(
        weights[index] * pivot[(EMPIRICAL, metric)] for index, metric in enumerate(COMPONENTS)
    )
    gain = empirical - bdmtf
    return float(gain.mean()), float(gain.min()), float((gain > 0).mean())


def analyze(
    scores_path: Path,
    manifest_path: Path,
    output_dir: Path,
    *,
    random_draws: int = 10_000,
    seed: int = 30371,
) -> dict[str, object]:
    scores = pd.read_csv(scores_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    component_scores = scores[scores["metric"].isin(COMPONENTS)].copy()
    component_scores["error"] = component_scores["error"].clip(0.0, 1.0)
    component_scores = component_scores.groupby(
        ["intervention_id", "intervention_type", "model", "metric"],
        as_index=False,
    )["error"].mean()
    pivot = component_scores.pivot(
        index=["intervention_id", "intervention_type"],
        columns=["model", "metric"],
        values="error",
    ).dropna()

    rows: list[dict[str, object]] = []
    equal = np.full(len(COMPONENTS), 1.0 / len(COMPONENTS))
    equal_bdmtf = sum(
        equal[index] * pivot[(BDMTF, metric)] for index, metric in enumerate(COMPONENTS)
    )
    equal_empirical = sum(
        equal[index] * pivot[(EMPIRICAL, metric)]
        for index, metric in enumerate(COMPONENTS)
    )
    mean_gain, min_intervention_gain, positive_intervention_share = _weighted_gain(pivot, equal)
    rows.append(
        {
            "scheme": "equal",
            "draw": 0,
            "mean_gain": mean_gain,
            "min_intervention_gain": min_intervention_gain,
            "positive_intervention_share": positive_intervention_share,
            **{f"weight_{metric}": equal[index] for index, metric in enumerate(COMPONENTS)},
        }
    )
    for held_out in range(len(COMPONENTS)):
        weights = np.full(len(COMPONENTS), 1.0 / (len(COMPONENTS) - 1))
        weights[held_out] = 0.0
        gain, minimum, positive = _weighted_gain(pivot, weights)
        rows.append(
            {
                "scheme": f"leave_out_{COMPONENTS[held_out]}",
                "draw": held_out,
                "mean_gain": gain,
                "min_intervention_gain": minimum,
                "positive_intervention_share": positive,
                **{f"weight_{metric}": weights[index] for index, metric in enumerate(COMPONENTS)},
            }
        )
    for active in range(len(COMPONENTS)):
        weights = np.zeros(len(COMPONENTS))
        weights[active] = 1.0
        gain, minimum, positive = _weighted_gain(pivot, weights)
        rows.append(
            {
                "scheme": f"only_{COMPONENTS[active]}",
                "draw": active,
                "mean_gain": gain,
                "min_intervention_gain": minimum,
                "positive_intervention_share": positive,
                **{f"weight_{metric}": weights[index] for index, metric in enumerate(COMPONENTS)},
            }
        )

    rng = np.random.default_rng(seed)
    random_gains = np.empty(random_draws, dtype=float)
    for draw, weights in enumerate(rng.dirichlet(np.ones(len(COMPONENTS)), size=random_draws)):
        gain, minimum, positive = _weighted_gain(pivot, weights)
        random_gains[draw] = gain
        rows.append(
            {
                "scheme": "dirichlet_1",
                "draw": draw,
                "mean_gain": gain,
                "min_intervention_gain": minimum,
                "positive_intervention_share": positive,
                **{f"weight_{metric}": weights[index] for index, metric in enumerate(COMPONENTS)},
            }
        )

    equal_by_type = []
    for intervention_type, subset in pivot.groupby(level="intervention_type"):
        gain, minimum, positive = _weighted_gain(subset, equal)
        equal_by_type.append(
            {
                "intervention_type": intervention_type,
                "interventions": int(len(subset)),
                "mean_gain": gain,
                "min_intervention_gain": minimum,
                "positive_intervention_share": positive,
            }
        )

    summary = {
        "status": "complete",
        "test_interventions": int(len(pivot)),
        "components": list(COMPONENTS),
        "primary_weights": {metric: 0.2 for metric in COMPONENTS},
        "normalization_scales_fitted_on_training_only": manifest[
            "scales_fitted_on_training_only"
        ],
        "component_clipping": [0.0, 1.0],
        "equal_weight_bdmtf_error": float(equal_bdmtf.mean()),
        "equal_weight_empirical_error": float(equal_empirical.mean()),
        "equal_weight_mean_gain": mean_gain,
        "equal_weight_relative_error_reduction": float(
            mean_gain / equal_empirical.mean()
        ),
        "equal_weight_positive_intervention_share": positive_intervention_share,
        "random_weight_draws": random_draws,
        "random_weight_positive_mean_gain_share": float((random_gains > 0).mean()),
        "random_weight_mean_gain_range": [
            float(random_gains.min()),
            float(random_gains.max()),
        ],
        "equal_weight_by_intervention_type": equal_by_type,
        "test_outcomes_used_to_choose_primary_weights": False,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / "composite_weight_sensitivity.csv", index=False)
    pd.DataFrame(equal_by_type).to_csv(output_dir / "equal_weight_by_type.csv", index=False)
    (output_dir / "path_metric_definition.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    by_type = {row["intervention_type"]: row for row in equal_by_type}
    macros = [
        "% Auto-generated by analyze_lemmy_path_metric.py.",
        f"\\newcommand{{\\LemmyPathWeightDraws}}{{{random_draws:,}}}",
        f"\\newcommand{{\\LemmyPathWeightPositivePct}}{{{100 * summary['random_weight_positive_mean_gain_share']:.1f}}}",
        f"\\newcommand{{\\LemmyPathWeightMinGain}}{{{random_gains.min():.3f}}}",
        f"\\newcommand{{\\LemmyPathWeightMaxGain}}{{{random_gains.max():.3f}}}",
        f"\\newcommand{{\\LemmyPathRelativeGainPct}}{{{100 * summary['equal_weight_relative_error_reduction']:.1f}}}",
    ]
    for intervention_type, command in (("lock_post", "Lock"), ("remove_post", "Removal")):
        row = by_type.get(intervention_type)
        if row:
            macros.extend(
                [
                    f"\\newcommand{{\\LemmyPath{command}N}}{{{row['interventions']}}}",
                    f"\\newcommand{{\\LemmyPath{command}Gain}}{{{row['mean_gain']:.3f}}}",
                ]
            )
    (output_dir / "lemmy_path_metric_macros.tex").write_text(
        "\n".join(macros) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--random-draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=30371)
    args = parser.parse_args()
    result = analyze(
        args.scores,
        args.manifest,
        args.output_dir,
        random_draws=args.random_draws,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
