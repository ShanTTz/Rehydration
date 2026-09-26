"""Generate paired mechanism attenuation and Lemmy sparsity diagnostics.

This script performs no new simulation. It recomputes reviewer-facing
statistics from frozen run and event-score artifacts and writes traceable
tables/macros for the review-fixed manuscript copy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> pd.DataFrame:
    return pd.DataFrame(
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def block_interactions(runs: pd.DataFrame) -> pd.DataFrame:
    keys = ["scenario", "community", "post_id", "seed"]
    rows: list[dict[str, object]] = []
    for key, group in runs.groupby(keys, sort=True):
        cells = group.set_index("cell")
        if not {"B0_P0", "B0_P1", "B1_P0", "B1_P1"}.issubset(cells.index):
            continue
        log_volume = {
            cell: math.log1p(float(cells.loc[cell, "comment_volume"]))
            for cell in ("B0_P0", "B0_P1", "B1_P0", "B1_P1")
        }
        depth = {
            cell: float(cells.loc[cell, "mean_leaf_depth"])
            for cell in ("B0_P0", "B0_P1", "B1_P0", "B1_P1")
        }
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "volume_log_interaction": (
                    log_volume["B1_P1"]
                    - log_volume["B1_P0"]
                    - log_volume["B0_P1"]
                    + log_volume["B0_P0"]
                ),
                "depth_interaction": (
                    depth["B1_P1"]
                    - depth["B1_P0"]
                    - depth["B0_P1"]
                    + depth["B0_P0"]
                ),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("No complete four-cell blocks")
    return result


def paired_attenuation(
    blocks: pd.DataFrame, comparison: str, draws: int, seed: int
) -> dict[str, float | int | str]:
    post = (
        blocks.groupby(["scenario", "community", "post_id"], as_index=False)[
            ["volume_log_interaction", "depth_interaction"]
        ]
        .mean()
    )
    reference = post[post["scenario"] == "reference"]
    deleted = post[post["scenario"] == comparison]
    paired = reference.merge(
        deleted,
        on=["community", "post_id"],
        suffixes=("_reference", "_deleted"),
        validate="one_to_one",
    )
    paired["volume_log_attenuation"] = (
        paired["volume_log_interaction_reference"]
        - paired["volume_log_interaction_deleted"]
    )
    paired["depth_attenuation"] = (
        paired["depth_interaction_reference"]
        - paired["depth_interaction_deleted"]
    )
    rng = np.random.default_rng(seed)
    volume_samples = np.empty(draws)
    depth_samples = np.empty(draws)
    groups = list(paired.groupby("community", sort=True))
    for index in range(draws):
        sample = pd.concat(
            [
                group.iloc[rng.integers(0, len(group), size=len(group))]
                for _, group in groups
            ],
            ignore_index=True,
        )
        volume_samples[index] = sample["volume_log_attenuation"].mean()
        depth_samples[index] = sample["depth_attenuation"].mean()
    volume = float(paired["volume_log_attenuation"].mean())
    depth = float(paired["depth_attenuation"].mean())
    volume_ci = np.quantile(volume_samples, [0.025, 0.975])
    depth_ci = np.quantile(depth_samples, [0.025, 0.975])
    return {
        "comparison": comparison,
        "posts": int(len(paired)),
        "log_ror_attenuation": volume,
        "log_ror_attenuation_ci_low": float(volume_ci[0]),
        "log_ror_attenuation_ci_high": float(volume_ci[1]),
        "ror_attenuation_factor": math.exp(volume),
        "ror_attenuation_factor_ci_low": math.exp(float(volume_ci[0])),
        "ror_attenuation_factor_ci_high": math.exp(float(volume_ci[1])),
        "depth_interaction_difference": depth,
        "depth_interaction_difference_ci_low": float(depth_ci[0]),
        "depth_interaction_difference_ci_high": float(depth_ci[1]),
    }


def write_mechanism_outputs(
    records_path: Path, output_dir: Path, manuscript_generated: Path
) -> dict[str, object]:
    runs = read_jsonl(records_path)
    blocks = block_interactions(runs)
    comparisons = [
        paired_attenuation(blocks, scenario, 10_000, 20270908 + index)
        for index, scenario in enumerate(
            ("no_conflict", "no_heat", "no_depth_preference", "no_depth_targeting")
        )
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(comparisons)
    output_path = output_dir / "mechanism_paired_attenuation.csv"
    frame.to_csv(output_path, index=False)
    conflict = frame.set_index("comparison").loc["no_conflict"]
    macro_lines = [
        "% Generated from paired post-level channel-deletion contrasts.",
        f"\\newcommand{{\\ConflictAttenuationFactor}}{{{conflict.ror_attenuation_factor:.3f}}}",
        f"\\newcommand{{\\ConflictAttenuationFactorLow}}{{{conflict.ror_attenuation_factor_ci_low:.3f}}}",
        f"\\newcommand{{\\ConflictAttenuationFactorHigh}}{{{conflict.ror_attenuation_factor_ci_high:.3f}}}",
        f"\\newcommand{{\\ConflictAttenuationLog}}{{{conflict.log_ror_attenuation:.3f}}}",
        f"\\newcommand{{\\ConflictAttenuationLogLow}}{{{conflict.log_ror_attenuation_ci_low:.3f}}}",
        f"\\newcommand{{\\ConflictAttenuationLogHigh}}{{{conflict.log_ror_attenuation_ci_high:.3f}}}",
    ]
    macro_path = manuscript_generated / "review_response_macros.tex"
    macro_path.write_text("\n".join(macro_lines) + "\n", encoding="utf-8")
    return {
        "records": str(records_path),
        "records_sha256": sha256(records_path),
        "output": str(output_path),
        "output_sha256": sha256(output_path),
        "comparisons": comparisons,
    }


def write_lemmy_outputs(
    diagnostic_path: Path, output_dir: Path, manuscript_generated: Path
) -> dict[str, object]:
    diagnostic = pd.read_csv(diagnostic_path)
    models = {
        "BDMTF": "bdmtf_training_path",
        "Empirical nearest": "empirical_nearest_path",
        "Always empty": "always_empty_composite_error",
    }
    rows: list[dict[str, object]] = []
    subsets = [("All", diagnostic)] + [
        ("Locks", diagnostic[diagnostic["intervention_type"] == "lock_post"]),
        ("Removals", diagnostic[diagnostic["intervention_type"] == "remove_post"]),
    ]
    for label, subset in subsets:
        row: dict[str, object] = {
            "subset": label,
            "interventions": int(len(subset)),
            "observed_nonempty": int((subset["observed_events_in_saved_window"] > 0).sum()),
            "observed_events": int(subset["observed_events_in_saved_window"].sum()),
        }
        for model, column in models.items():
            row[model] = float(subset[column].mean())
        rows.append(row)
    table_frame = pd.DataFrame(rows)
    output_path = output_dir / "lemmy_sparse_path_baselines.csv"
    table_frame.to_csv(output_path, index=False)

    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Subset & $N$ & Nonempty & Events & BDMTF & Empirical & Empty \\",
        r"\midrule",
    ]
    for row in table_frame.itertuples(index=False):
        lines.append(
            f"{row.subset} & {row.interventions} & {row.observed_nonempty} & "
            f"{row.observed_events} & {row.BDMTF:.3f} & "
            f"{getattr(row, '_5'):.3f} & {getattr(row, '_6'):.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    table_path = manuscript_generated / "table_lemmy_sparse_path_audit.tex"
    table_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    overall = table_frame.iloc[0]
    macros = [
        "% Generated from the saved Lemmy eight-day event windows.",
        f"\\newcommand{{\\LemmyObservedNonempty}}{{{int(overall['observed_nonempty'])}}}",
        f"\\newcommand{{\\LemmyObservedEvents}}{{{int(overall['observed_events'])}}}",
        f"\\newcommand{{\\LemmyAlwaysEmptyError}}{{{overall['Always empty']:.3f}}}",
        f"\\newcommand{{\\LemmyBDMTFSparseError}}{{{overall['BDMTF']:.3f}}}",
        f"\\newcommand{{\\LemmyEmpiricalSparseError}}{{{overall['Empirical nearest']:.3f}}}",
    ]
    macro_path = manuscript_generated / "lemmy_sparse_audit_macros.tex"
    macro_path.write_text("\n".join(macros) + "\n", encoding="utf-8")
    return {
        "diagnostic": str(diagnostic_path),
        "diagnostic_sha256": sha256(diagnostic_path),
        "output": str(output_path),
        "output_sha256": sha256(output_path),
        "observed_nonempty": int(overall["observed_nonempty"]),
        "observed_events": int(overall["observed_events"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manuscript",
        type=Path,
        default=ROOT
        / "manuscript"
        / "iclr2027_overleaf_package_review_fixed_20260908",
    )
    args = parser.parse_args()
    output = ROOT / "artifacts" / "reviewer_validation" / "review_20260908"
    generated = args.manuscript / "generated"
    mechanism = write_mechanism_outputs(
        ROOT
        / "artifacts"
        / "reviewer_validation"
        / "mechanism_storyline_20260901"
        / "channel_ablation_4cell"
        / "runs.jsonl",
        output,
        generated,
    )
    lemmy = write_lemmy_outputs(
        output / "lemmy_empty_path_diagnostic.csv", output, generated
    )
    manifest = {
        "status": "complete",
        "kind": "reanalysis_of_frozen_artifacts",
        "mechanism": mechanism,
        "lemmy": lemmy,
    }
    manifest_path = output / "review_response_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
