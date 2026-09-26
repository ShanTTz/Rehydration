from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from bdmtf.revision.provenance import sha256_file, write_json


TARGET_BASELINES = (
    "target_empirical_bootstrap",
    "target_hawkes",
    "target_branching_process",
)


def _paired_metric_comparison(
    fidelity: pd.DataFrame,
    left: str,
    right: str,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    left_rows = fidelity[fidelity["model"].eq(left)][
        ["community", "metric", "normalized_wasserstein"]
    ].rename(columns={"normalized_wasserstein": "left"})
    right_rows = fidelity[fidelity["model"].eq(right)][
        ["community", "metric", "normalized_wasserstein"]
    ].rename(columns={"normalized_wasserstein": "right"})
    paired = left_rows.merge(
        right_rows,
        on=["community", "metric"],
        validate="one_to_one",
    )
    paired["gain"] = paired["right"] - paired["left"]
    values = paired["gain"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    draws = np.asarray(
        [
            float(rng.choice(values, size=len(values), replace=True).mean())
            for _ in range(samples)
        ]
    )
    return {
        "left_model": left,
        "right_model": right,
        "paired_metrics": int(len(paired)),
        "mean_nwd_improvement": float(values.mean()),
        "ci": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "left_better_metric_share": float(np.mean(values > 0)),
    }


def run_cross_platform_budget_audit(
    root: str | Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    project = Path(root)
    source = project / str(config["source_dir"])
    output = project / str(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    curve_path = source / "adapter_budget_curve.csv"
    ranking_path = source / "evaluation" / "model_ranking.csv"
    fidelity_path = source / "evaluation" / "fidelity_distances.csv"
    manifest_path = source / "platform_adapter_curve_manifest.json"
    curve = pd.read_csv(curve_path)
    ranking = pd.read_csv(ranking_path).set_index("model")
    fidelity = pd.read_csv(fidelity_path)
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    zero_budget = curve[curve["validation_budget"].eq(0)].iloc[0]
    best = curve.sort_values(
        ["all_metric_median_distance", "validation_budget"]
    ).iloc[0]
    equal_budget_rows = [
        {
            "model": "target_default_bdmtf_common_seed",
            "target_train_cascades": int(source_manifest["n_train"]),
            "target_validation_cascades": 0,
            "median_nwd": float(zero_budget["all_metric_median_distance"]),
            "contract": "BDMTF",
        }
    ]
    for model in TARGET_BASELINES:
        equal_budget_rows.append(
            {
                "model": model,
                "target_train_cascades": int(source_manifest["n_train"]),
                "target_validation_cascades": 0,
                "median_nwd": float(ranking.loc[model, "median"]),
                "contract": "none",
            }
        )
    equal_budget = pd.DataFrame(equal_budget_rows).sort_values("median_nwd")
    equal_budget.to_csv(output / "equal_target_budget_models.csv", index=False)

    comparisons = []
    for offset, baseline in enumerate(TARGET_BASELINES):
        comparisons.append(
            _paired_metric_comparison(
                fidelity,
                "platform_adapted_n0",
                baseline,
                int(config.get("bootstrap_samples", 5000)),
                int(config.get("seed", 30371)) + offset,
            )
        )
    pd.DataFrame(comparisons).to_csv(
        output / "paired_equal_budget_comparisons.csv",
        index=False,
    )

    zero_shot = float(zero_budget["reddit_zero_shot_median_distance"])
    target_default = float(zero_budget["all_metric_median_distance"])
    adapted = float(best["all_metric_median_distance"])
    result = {
        "status": "complete",
        "platform": "HackerNews",
        "target_train_cascades": int(source_manifest["n_train"]),
        "target_test_cascades": int(source_manifest["n_test"]),
        "fair_target_default": {
            "model": "platform_adapted_n0",
            "reason": (
                "It uses the same target profile and base structure configuration "
                "as target-default, with the common adapter seed schedule."
            ),
            "median_nwd": target_default,
        },
        "reddit_zero_shot_median_nwd": zero_shot,
        "target_data_gain": float((zero_shot - target_default) / zero_shot),
        "best_adapter_validation_budget": int(best["validation_budget"]),
        "best_adapter_median_nwd": adapted,
        "incremental_adapter_gain": float((target_default - adapted) / target_default),
        "equal_budget_models": equal_budget.to_dict(orient="records"),
        "paired_comparisons": comparisons,
        "interpretation": (
            "The portable state/action contract supports target adaptation, but "
            "unconditional cascade fidelity is not a BDMTF performance advantage."
        ),
        "inputs": {
            str(path.relative_to(project)): sha256_file(path)
            for path in (curve_path, ranking_path, fidelity_path, manifest_path)
        },
    }
    write_json(output / "manifest.json", result)
    _write_report(output / "EQUAL_TARGET_BUDGET_REPORT.md", result)
    return result


def _write_report(path: Path, result: Mapping[str, Any]) -> None:
    lines = [
        "# Equal-Target-Budget Cross-Platform Audit",
        "",
        "## Target-Default Definition",
        "",
        str(result["fair_target_default"]["reason"]),
        "",
        "## Same Target-Training Budget",
        "",
        "| Model | Contract | Target train | Validation | Median NWD |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in result["equal_budget_models"]:
        lines.append(
            f"| {row['model']} | {row['contract']} | "
            f"{row['target_train_cascades']:,} | "
            f"{row['target_validation_cascades']:,} | {row['median_nwd']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Transport Decomposition",
            "",
            (
                f"Replacing the Reddit profile with the HN training profile reduces "
                f"median NWD by {100 * result['target_data_gain']:.1f}%. The best "
                f"structure adapter uses {result['best_adapter_validation_budget']} "
                f"validation cascades and adds {100 * result['incremental_adapter_gain']:.1f}% "
                "relative improvement over the common-seed target default."
            ),
            "",
            str(result["interpretation"]),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")

