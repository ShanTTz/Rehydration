from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from bdmtf.revision.provenance import sha256_file, write_json


@dataclass(frozen=True)
class OutcomeSpec:
    name: str
    treatment_effect: float
    semantic_main: float
    semantic_interaction: float
    structural_noise: float
    structural_noise_interaction: float


METHODS = (
    "independent_live",
    "common_seed_live",
    "rehydration",
)


def _outcome(
    policy: int,
    semantics: np.ndarray,
    structural_noise: np.ndarray,
    spec: OutcomeSpec,
) -> np.ndarray:
    return (
        spec.treatment_effect * policy
        + spec.semantic_main * semantics
        + spec.semantic_interaction * policy * semantics
        + (
            spec.structural_noise
            + spec.structural_noise_interaction * policy
        )
        * structural_noise
    )


def _estimate(diff: np.ndarray, truth: float) -> dict[str, float]:
    estimate = float(diff.mean())
    standard_error = float(diff.std(ddof=1) / np.sqrt(len(diff)))
    low = estimate - 1.96 * standard_error
    high = estimate + 1.96 * standard_error
    return {
        "estimate": estimate,
        "standard_error": standard_error,
        "ci_low": low,
        "ci_high": high,
        "covered": float(low <= truth <= high),
    }


def simulate_trial(
    rng: np.random.Generator,
    units: int,
    semantic_shift: float,
    proposal_noise: float,
    spec: OutcomeSpec,
) -> dict[str, dict[str, float]]:
    base_semantics = rng.normal(size=units)

    independent_proposal_0 = rng.normal(scale=proposal_noise, size=units)
    independent_proposal_1 = rng.normal(scale=proposal_noise, size=units)
    independent_noise_0 = rng.normal(size=units)
    independent_noise_1 = rng.normal(size=units)
    independent_y0 = _outcome(
        0,
        base_semantics + independent_proposal_0,
        independent_noise_0,
        spec,
    )
    independent_y1 = _outcome(
        1,
        base_semantics + semantic_shift + independent_proposal_1,
        independent_noise_1,
        spec,
    )

    common_proposal = rng.normal(scale=proposal_noise, size=units)
    common_noise = rng.normal(size=units)
    common_y0 = _outcome(
        0,
        base_semantics + common_proposal,
        common_noise,
        spec,
    )
    common_y1 = _outcome(
        1,
        base_semantics + semantic_shift + common_proposal,
        common_noise,
        spec,
    )

    frozen_noise = rng.normal(size=units)
    frozen_y0 = _outcome(0, base_semantics, frozen_noise, spec)
    frozen_y1 = _outcome(1, base_semantics, frozen_noise, spec)

    truth = spec.treatment_effect
    return {
        "independent_live": _estimate(independent_y1 - independent_y0, truth),
        "common_seed_live": _estimate(common_y1 - common_y0, truth),
        "rehydration": _estimate(frozen_y1 - frozen_y0, truth),
    }


def _summarize(replicates: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["outcome", "semantic_shift", "method"]
    for values, group in replicates.groupby(keys, sort=True):
        outcome, shift, method = values
        truth = float(group["truth"].iloc[0])
        errors = group["estimate"].to_numpy(dtype=float) - truth
        rows.append(
            {
                "outcome": str(outcome),
                "semantic_shift": float(shift),
                "method": str(method),
                "truth": truth,
                "mean_estimate": float(group["estimate"].mean()),
                "bias": float(errors.mean()),
                "absolute_bias": float(abs(errors.mean())),
                "sampling_variance": float(group["estimate"].var(ddof=1)),
                "mse": float(np.mean(errors**2)),
                "coverage_95": float(group["covered"].mean()),
                "mean_ci_width": float(
                    (group["ci_high"] - group["ci_low"]).mean()
                ),
                "replications": int(len(group)),
            }
        )
    summary = pd.DataFrame(rows)
    references = summary[summary["method"].eq("rehydration")].set_index(
        ["outcome", "semantic_shift"]
    )
    summary["variance_ratio_vs_rehydration"] = summary.apply(
        lambda row: (
            row["sampling_variance"]
            / max(
                float(
                    references.loc[
                        (row["outcome"], row["semantic_shift"]),
                        "sampling_variance",
                    ]
                ),
                1e-12,
            )
        ),
        axis=1,
    )
    return summary


def _write_report(
    path: Path,
    summary: pd.DataFrame,
    manifest: Mapping[str, Any],
) -> None:
    lines = [
        "# Known-Ground-Truth Counterfactual Recovery",
        "",
        "## Estimand",
        "",
        (
            "The target is the population controlled structural effect "
            "E[M(1, Z) - M(0, Z)] with the pre-treatment semantic potential "
            "Z held fixed. Its analytic value is the prespecified policy "
            "coefficient in each outcome equation."
        ),
        "",
        "## Design",
        "",
        (
            f"Each cell contains {manifest['replications']:,} Monte Carlo "
            f"experiments with {manifest['units_per_replication']:,} units. "
            "Independent live replay redraws both semantic and structural "
            "innovations; common-seed live replay shares innovations but still "
            "allows treatment to shift semantics; Rehydration fixes semantics "
            "before either branch is run."
        ),
        "",
        "## Recovery Results",
        "",
        "| Outcome | Semantic shift | Method | Bias | MSE | Coverage | Variance / Rehydration |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.outcome} | {row.semantic_shift:.2f} | "
            f"{row.method.replace('_', ' ')} | {row.bias:.3f} | "
            f"{row.mse:.4f} | {100 * row.coverage_95:.1f}% | "
            f"{row.variance_ratio_vs_rehydration:.2f}x |"
        )
    lines.extend(
        [
            "",
            "A zero semantic shift is the negative control. Positive shifts "
            "encode the failure mode in which the intervention changes the live "
            "proposal distribution. Common random seeds can reduce dispersion, "
            "but they cannot remove that post-treatment semantic path.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_ground_truth_scm(
    root: str | Path,
    config: Mapping[str, Any],
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    project = Path(root)
    output = project / str(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    seed = int(config.get("seed", 30371))
    units = int(config.get("units_per_replication", 250))
    replications = int(config.get("replications", 2000))
    proposal_noise = float(config.get("proposal_noise", 0.8))
    shifts = [float(value) for value in config["semantic_shifts"]]
    specs = [OutcomeSpec(**raw) for raw in config["outcomes"]]
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []

    for spec in specs:
        for shift in shifts:
            for replication in range(replications):
                estimates = simulate_trial(
                    rng,
                    units,
                    shift,
                    proposal_noise,
                    spec,
                )
                for method in METHODS:
                    rows.append(
                        {
                            "outcome": spec.name,
                            "semantic_shift": shift,
                            "replication": replication,
                            "method": method,
                            "truth": spec.treatment_effect,
                            **estimates[method],
                        }
                    )

    replicates = pd.DataFrame(rows)
    summary = _summarize(replicates)
    replicates.to_csv(output / "replicate_estimates.csv", index=False)
    summary.to_csv(output / "recovery_summary.csv", index=False)
    replicates.to_parquet(output / "replicate_estimates.parquet", index=False)
    manifest = {
        "status": "complete",
        "seed": seed,
        "units_per_replication": units,
        "replications": replications,
        "semantic_shifts": shifts,
        "proposal_noise": proposal_noise,
        "methods": list(METHODS),
        "outcomes": [spec.__dict__ for spec in specs],
        "estimand": "E[M(1,Z)-M(0,Z)] with pre-treatment Z fixed",
        "ground_truth": "analytic policy coefficient",
    }
    if config_path is not None:
        source = Path(config_path)
        manifest["config"] = {
            "path": str(source.resolve()),
            "sha256": sha256_file(source),
        }
    write_json(output / "manifest.json", manifest)
    _write_report(output / "GROUND_TRUTH_SCM_REPORT.md", summary, manifest)
    return manifest

