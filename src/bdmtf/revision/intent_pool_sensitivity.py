from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from bdmtf.experiments import (
    config_from_dict,
    interventions_from_config,
    run_batch,
)
from bdmtf.revision.provenance import sha256_file, write_json


def _factorial_contrasts(
    runs: pd.DataFrame,
    cells: Mapping[str, str],
) -> pd.DataFrame:
    index = [
        "intent_pool_capacity_multiplier",
        "community",
        "post_id",
        "seed",
    ]
    metrics = ["comment_volume", "mean_leaf_depth", "max_depth"]
    pivot = runs.pivot(index=index, columns="condition", values=metrics)
    required = list(cells.values())
    complete = np.ones(len(pivot), dtype=bool)
    for metric in metrics:
        for condition in required:
            complete &= pivot[(metric, condition)].notna().to_numpy()
    pivot = pivot.loc[complete]

    def values(metric: str, name: str) -> np.ndarray:
        return pivot[(metric, cells[name])].to_numpy(dtype=float)

    bb_volume = values("comment_volume", "baseline_best")
    bc_volume = values("comment_volume", "baseline_controversial")
    tb_volume = values("comment_volume", "toxic_best")
    tc_volume = values("comment_volume", "toxic_controversial")
    bb_depth = values("mean_leaf_depth", "baseline_best")
    bc_depth = values("mean_leaf_depth", "baseline_controversial")
    tb_depth = values("mean_leaf_depth", "toxic_best")
    tc_depth = values("mean_leaf_depth", "toxic_controversial")

    out = pivot.reset_index()[index]
    out["joint_volume_ratio"] = (tc_volume + 1.0) / (bb_volume + 1.0)
    out["interaction_volume_log"] = (
        np.log1p(tc_volume)
        - np.log1p(tb_volume)
        - np.log1p(bc_volume)
        + np.log1p(bb_volume)
    )
    out["interaction_volume_ratio_of_ratios"] = np.exp(
        out["interaction_volume_log"]
    )
    out["joint_mean_leaf_depth_delta"] = tc_depth - bb_depth
    out["interaction_mean_leaf_depth"] = (
        tc_depth - tb_depth - bc_depth + bb_depth
    )
    return out


def _cluster_interval(
    frame: pd.DataFrame,
    column: str,
    samples: int,
    seed: int,
) -> tuple[float, float, float]:
    clusters = [
        group[column].to_numpy(dtype=float)
        for _, group in frame.groupby(["community", "post_id"], sort=False)
    ]
    estimate = float(frame[column].mean())
    if len(clusters) < 2:
        return estimate, estimate, estimate
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(samples):
        selected = rng.integers(0, len(clusters), size=len(clusters))
        values = np.concatenate([clusters[index] for index in selected])
        draws.append(float(values.mean()))
    return (
        estimate,
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    )


def _effect_summary(
    contrasts: pd.DataFrame,
    bootstrap_samples: int,
    seed: int,
) -> pd.DataFrame:
    columns = [
        "joint_volume_ratio",
        "interaction_volume_ratio_of_ratios",
        "joint_mean_leaf_depth_delta",
        "interaction_mean_leaf_depth",
    ]
    rows: list[dict[str, Any]] = []
    for multiplier, group in contrasts.groupby(
        "intent_pool_capacity_multiplier", sort=True
    ):
        for offset, column in enumerate(columns):
            estimate, low, high = _cluster_interval(
                group,
                column,
                bootstrap_samples,
                seed + int(multiplier) * 101 + offset,
            )
            rows.append(
                {
                    "intent_pool_capacity_multiplier": int(multiplier),
                    "metric": column,
                    "estimate": estimate,
                    "ci_low": low,
                    "ci_high": high,
                    "post_seed_blocks": int(len(group)),
                    "posts": int(
                        group[["community", "post_id"]].drop_duplicates().shape[0]
                    ),
                }
            )
    summary = pd.DataFrame(rows)
    largest = int(summary["intent_pool_capacity_multiplier"].max())
    references = (
        summary[summary["intent_pool_capacity_multiplier"].eq(largest)]
        .set_index("metric")["estimate"]
        .to_dict()
    )
    summary["largest_pool_estimate"] = summary["metric"].map(references)
    summary["absolute_gap_from_largest"] = (
        summary["estimate"] - summary["largest_pool_estimate"]
    ).abs()
    summary["relative_gap_from_largest"] = summary.apply(
        lambda row: (
            abs(row["estimate"] - row["largest_pool_estimate"])
            / max(abs(row["largest_pool_estimate"]), 1e-9)
        ),
        axis=1,
    )
    return summary


def _exhaustion_summary(runs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_columns = ["intent_pool_capacity_multiplier", "condition"]
    for keys, group in runs.groupby(group_columns, sort=True):
        multiplier, condition = keys
        requests = float(group["intent_requests"].sum())
        exhausted = float(group["pool_exhausted"].sum())
        supportive_requests = float(group["supportive_intent_requests"].sum())
        supportive_exhausted = float(group["supportive_pool_exhausted"].sum())
        antagonistic_requests = float(group["antagonistic_intent_requests"].sum())
        antagonistic_exhausted = float(group["antagonistic_pool_exhausted"].sum())
        rows.append(
            {
                "intent_pool_capacity_multiplier": int(multiplier),
                "condition": str(condition),
                "runs": int(len(group)),
                "intent_requests": int(requests),
                "pool_exhausted": int(exhausted),
                "exhaustion_rate": exhausted / requests if requests else 0.0,
                "supportive_exhaustion_rate": (
                    supportive_exhausted / supportive_requests
                    if supportive_requests
                    else 0.0
                ),
                "antagonistic_exhaustion_rate": (
                    antagonistic_exhausted / antagonistic_requests
                    if antagonistic_requests
                    else 0.0
                ),
                "mean_comment_volume": float(group["comment_volume"].mean()),
                "mean_leaf_depth": float(group["mean_leaf_depth"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _write_report(
    path: Path,
    exhaustion: pd.DataFrame,
    effects: pd.DataFrame,
    manifest: Mapping[str, Any],
) -> None:
    lines = [
        "# Frozen-Intent Pool Capacity Audit",
        "",
        "## Design",
        "",
        (
            f"The audit replays {manifest['posts']} posts with "
            f"{manifest['seeds']} seeds and the same four factorial cells at "
            f"capacity multipliers {manifest['capacity_multipliers']}."
        ),
        "Each branch uses an independent no-replacement session. Exhausted reply "
        "requests become logged no-ops and never trigger semantic regeneration.",
        "",
        "## Exhaustion",
        "",
        "| Capacity | Condition | Requests | Exhausted | Rate | Supportive | Antagonistic |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in exhaustion.itertuples(index=False):
        lines.append(
            f"| {row.intent_pool_capacity_multiplier}x | {row.condition} | "
            f"{row.intent_requests:,} | {row.pool_exhausted:,} | "
            f"{100 * row.exhaustion_rate:.2f}% | "
            f"{100 * row.supportive_exhaustion_rate:.2f}% | "
            f"{100 * row.antagonistic_exhaustion_rate:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Factorial Effect Stability",
            "",
            "| Capacity | Metric | Estimate [95% CI] | Relative gap from largest pool |",
            "| ---: | --- | ---: | ---: |",
        ]
    )
    for row in effects.itertuples(index=False):
        lines.append(
            f"| {row.intent_pool_capacity_multiplier}x | {row.metric} | "
            f"{row.estimate:.3f} [{row.ci_low:.3f}, {row.ci_high:.3f}] | "
            f"{100 * row.relative_gap_from_largest:.2f}% |"
        )
    lines.extend(
        [
            "",
            "All estimates are computed from frozen posts and paired seeds. The largest "
            "tested pool is a convergence reference, not a fitted target.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_intent_pool_sensitivity(
    root: str | Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    project = Path(root)
    base_config_path = project / str(config["base_config"])
    base_raw = json.loads(base_config_path.read_text(encoding="utf-8"))
    if config.get("communities"):
        base_raw["communities"] = [str(value) for value in config["communities"]]
    output = project / str(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    capacities = [int(value) for value in config["capacity_multipliers"]]
    all_records: list[pd.DataFrame] = []

    for multiplier in capacities:
        raw = json.loads(json.dumps(base_raw))
        raw["posts_per_community"] = int(config["posts_per_community"])
        raw["seeds"] = [int(value) for value in config["seeds"]]
        raw["simulation"]["intent_pool_capacity_multiplier"] = multiplier
        raw["simulation"]["intent_pool_exhaustion_policy"] = "no_op"
        simulation = config_from_dict(raw)
        run_dir = output / f"capacity_{multiplier:03d}x"
        records = run_batch(
            social_root=project / str(config["social_root"]),
            output_dir=run_dir,
            communities=raw["communities"],
            posts_per_community=int(raw["posts_per_community"]),
            seeds=raw["seeds"],
            config=simulation,
            interventions=interventions_from_config(raw),
            seed_mode=str(raw.get("randomization", {}).get("seed_mode", "condition_specific")),
            resume=True,
        )
        frame = pd.DataFrame(records)
        frame["intent_pool_capacity_multiplier"] = multiplier
        all_records.append(frame)

    runs = pd.concat(all_records, ignore_index=True)
    runs.to_csv(output / "capacity_runs.csv", index=False)
    runs.to_parquet(output / "capacity_runs.parquet", index=False)
    cells = base_raw["analysis"]["factorial_cells"]
    contrasts = _factorial_contrasts(runs, cells)
    contrasts.to_csv(output / "factorial_contrasts.csv", index=False)
    effects = _effect_summary(
        contrasts,
        int(config.get("bootstrap_samples", 2000)),
        int(config.get("seed", 30371)),
    )
    effects.to_csv(output / "effect_stability.csv", index=False)
    exhaustion = _exhaustion_summary(runs)
    exhaustion.to_csv(output / "exhaustion_by_condition.csv", index=False)

    manifest = {
        "status": "complete",
        "capacity_multipliers": capacities,
        "posts": int(runs[["community", "post_id"]].drop_duplicates().shape[0]),
        "communities": sorted(runs["community"].astype(str).unique().tolist()),
        "seeds": int(runs["seed"].nunique()),
        "runs": int(len(runs)),
        "post_seed_blocks_per_capacity": int(
            contrasts.groupby("intent_pool_capacity_multiplier").size().min()
        ),
        "exhaustion_policy": "no_op",
        "base_config": {
            "path": str(base_config_path.resolve()),
            "sha256": sha256_file(base_config_path),
        },
        "convergence_reference_multiplier": max(capacities),
    }
    write_json(output / "manifest.json", manifest)
    _write_report(output / "INTENT_POOL_CAPACITY_REPORT.md", exhaustion, effects, manifest)
    return manifest
