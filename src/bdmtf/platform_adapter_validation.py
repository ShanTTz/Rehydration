from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from bdmtf.revision.evaluation import evaluate_fidelity, summarize_model_ranking
from bdmtf.revision.fitted_models import fit_pooled_model
from bdmtf.revision.provenance import sha256_file, write_json
from bdmtf.revision.simulation import (
    RevisionSimulationConfig,
    event_metrics,
    nodes_to_records,
    simulate_cascade,
)


def chronological_platform_split(
    metrics: pd.DataFrame,
    events: pd.DataFrame,
    platform: str,
    fractions: Mapping[str, float],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    selected = metrics[metrics["platform"].astype(str).eq(platform)].copy()
    if selected.empty:
        raise ValueError(f"No metrics found for platform {platform}")
    roots = events[events["platform"].astype(str).eq(platform)].copy()
    roots["content_id"] = roots["content_id"].astype(str)
    roots["created_at"] = pd.to_datetime(roots["created_at"], utc=True, errors="coerce")
    first_seen = (
        roots.dropna(subset=["created_at"])
        .groupby(["community", "content_id"], as_index=False)["created_at"]
        .min()
        .rename(columns={"content_id": "post_id"})
    )
    selected["post_id"] = selected["post_id"].astype(str)
    selected = selected.merge(
        first_seen,
        on=["community", "post_id"],
        how="inner",
        validate="one_to_one",
    )
    train_fraction = float(fractions["train"])
    validation_fraction = float(fractions["validation"])
    if not np.isclose(sum(map(float, fractions.values())), 1.0):
        raise ValueError("Platform split fractions must sum to 1")
    parts: list[pd.DataFrame] = []
    boundaries: list[dict[str, Any]] = []
    for community, group in selected.groupby("community", sort=True):
        ordered = group.sort_values(["created_at", "post_id"]).reset_index(drop=True)
        n = len(ordered)
        train_end = int(np.floor(n * train_fraction))
        validation_end = int(np.floor(n * (train_fraction + validation_fraction)))
        ordered["split"] = "test"
        ordered.loc[: train_end - 1, "split"] = "train"
        ordered.loc[train_end : validation_end - 1, "split"] = "validation"
        parts.append(ordered)
        boundaries.append(
            {
                "community": str(community),
                "n": int(n),
                "train": int((ordered["split"] == "train").sum()),
                "validation": int((ordered["split"] == "validation").sum()),
                "test": int((ordered["split"] == "test").sum()),
                "train_latest": _iso_max(
                    ordered.loc[ordered["split"] == "train", "created_at"]
                ),
                "validation_latest": _iso_max(
                    ordered.loc[ordered["split"] == "validation", "created_at"]
                ),
                "test_earliest": _iso_min(
                    ordered.loc[ordered["split"] == "test", "created_at"]
                ),
            }
        )
    split_metrics = pd.concat(parts, ignore_index=True)
    leakage = False
    for _, group in split_metrics.groupby("community", sort=True):
        train_max = group.loc[group["split"] == "train", "created_at"].max()
        validation_min = group.loc[
            group["split"] == "validation", "created_at"
        ].min()
        validation_max = group.loc[
            group["split"] == "validation", "created_at"
        ].max()
        test_min = group.loc[group["split"] == "test", "created_at"].min()
        leakage |= bool(train_max > validation_min or validation_max > test_min)
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "platform": platform,
        "fractions": dict(fractions),
        "n_cascades": int(len(split_metrics)),
        "boundaries": boundaries,
        "temporal_leakage_detected": leakage,
    }
    return split_metrics, manifest


def select_platform_adapter(
    platform_metrics: pd.DataFrame,
    profile: Any,
    base_config: RevisionSimulationConfig,
    grid: Mapping[str, Iterable[Any]],
    selection_metrics: Iterable[str],
    max_validation_cascades: int,
    seed: int,
) -> tuple[RevisionSimulationConfig, pd.DataFrame, dict[str, Any]]:
    train = platform_metrics[platform_metrics["split"].eq("train")].copy()
    validation = platform_metrics[
        platform_metrics["split"].eq("validation")
    ].sort_values(["community", "created_at", "post_id"])
    if max_validation_cascades > 0:
        validation = validation.groupby("community", group_keys=False).head(
            max_validation_cascades
        )
    metric_names = list(selection_metrics)
    candidates = [
        dict(zip(grid.keys(), values))
        for values in itertools.product(*(grid[key] for key in grid))
    ]
    rows: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(candidates):
        candidate_config = replace(
            base_config,
            ranking=str(candidate["ranking"]),
            viewport_k=int(candidate["viewport_k"]),
            deep_drill_lambda=float(candidate["deep_drill_lambda"]),
        )
        generated = []
        for target in validation.itertuples(index=False):
            run_seed = seed + _stable_seed(
                str(target.post_id), str(candidate_index), "adapter_selection"
            )
            nodes = simulate_cascade(
                "learned_bdmtf",
                profile,
                train,
                str(target.post_id),
                run_seed,
                candidate_config,
            )
            generated.append(event_metrics(nodes))
        simulated = pd.DataFrame(generated)
        distances = []
        for metric in metric_names:
            real = validation[metric].to_numpy(dtype=float)
            fake = simulated[metric].to_numpy(dtype=float)
            scale = max(float(np.std(real)), abs(float(np.mean(real))) * 0.1, 1e-9)
            distances.append(float(wasserstein_distance(real, fake) / scale))
        rows.append(
            {
                "candidate_index": candidate_index,
                **candidate,
                "mean_normalized_wasserstein": float(np.mean(distances)),
                "median_normalized_wasserstein": float(np.median(distances)),
                "n_validation_cascades": int(len(validation)),
                "metrics": json.dumps(metric_names),
            }
        )
    scores = pd.DataFrame(rows).sort_values(
        [
            "mean_normalized_wasserstein",
            "median_normalized_wasserstein",
            "candidate_index",
        ]
    )
    best = scores.iloc[0]
    selected_config = replace(
        base_config,
        ranking=str(best["ranking"]),
        viewport_k=int(best["viewport_k"]),
        deep_drill_lambda=float(best["deep_drill_lambda"]),
    )
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "selection_split": "validation",
        "test_used_for_selection": False,
        "candidate_count": int(len(scores)),
        "selected": {
            "ranking": selected_config.ranking,
            "viewport_k": selected_config.viewport_k,
            "deep_drill_lambda": selected_config.deep_drill_lambda,
        },
        "selection_score": float(best["mean_normalized_wasserstein"]),
        "n_validation_cascades": int(len(validation)),
        "selection_metrics": metric_names,
    }
    return selected_config, scores, manifest


def run_platform_adapter_test(
    platform_metrics: pd.DataFrame,
    reddit_metrics: pd.DataFrame,
    selected_config: RevisionSimulationConfig,
    default_config: RevisionSimulationConfig,
    seeds: Iterable[int],
    max_test_cascades: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_profile = fit_pooled_model(platform_metrics, (), "target_platform")
    reddit_profile = fit_pooled_model(reddit_metrics, (), "reddit_source")
    target_train = platform_metrics[platform_metrics["split"].eq("train")].copy()
    reddit_train = reddit_metrics[reddit_metrics["split"].eq("train")].copy()
    test = platform_metrics[platform_metrics["split"].eq("test")].sort_values(
        ["community", "created_at", "post_id"]
    )
    if max_test_cascades > 0:
        test = test.groupby("community", group_keys=False).head(max_test_cascades)
    model_specs = (
        (
            "reddit_zero_shot_learned_bdmtf",
            "learned_bdmtf",
            reddit_profile,
            reddit_train,
            default_config,
        ),
        (
            "target_default_learned_bdmtf",
            "learned_bdmtf",
            target_profile,
            target_train,
            default_config,
        ),
        (
            "platform_adapted_learned_bdmtf",
            "learned_bdmtf",
            target_profile,
            target_train,
            selected_config,
        ),
        (
            "target_empirical_bootstrap",
            "empirical_bootstrap",
            target_profile,
            target_train,
            default_config,
        ),
        (
            "target_branching_process",
            "branching_process",
            target_profile,
            target_train,
            default_config,
        ),
        (
            "target_hawkes",
            "hawkes",
            target_profile,
            target_train,
            default_config,
        ),
    )
    metric_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    for target in test.itertuples(index=False):
        for label, model_name, profile, train_rows, config in model_specs:
            for seed in map(int, seeds):
                seed_family = (
                    "platform_adapted_learned_bdmtf"
                    if label
                    in {
                        "target_default_learned_bdmtf",
                        "platform_adapted_learned_bdmtf",
                    }
                    else label
                )
                run_seed = seed + _stable_seed(
                    str(target.post_id), seed_family, "platform_test"
                )
                nodes = simulate_cascade(
                    model_name,
                    profile,
                    train_rows,
                    str(target.post_id),
                    run_seed,
                    config,
                )
                context = {
                    "platform": str(target.platform),
                    "community": str(target.community),
                    "post_id": str(target.post_id),
                    "model": label,
                    "seed": seed,
                    "split": "test",
                }
                metric_rows.append({**context, **event_metrics(nodes)})
                event_rows.extend(nodes_to_records(nodes, context))
    return pd.DataFrame(metric_rows), pd.DataFrame(event_rows)


def run_platform_adapter_workflow(
    root: str | Path,
    config: Mapping[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    project = Path(root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    inputs = config["inputs"]
    external_metrics_path = project / inputs["external_metrics"]
    external_events_path = project / inputs["external_events"]
    reddit_metrics_path = project / inputs["reddit_metrics"]
    simulation_config_path = project / inputs["simulation_config"]
    external_metrics = pd.read_parquet(external_metrics_path)
    external_events = pd.read_parquet(external_events_path)
    reddit_metrics = pd.read_parquet(reddit_metrics_path)
    raw_simulation = json.loads(simulation_config_path.read_text(encoding="utf-8"))
    base_config = RevisionSimulationConfig.from_dict(raw_simulation["simulation"])

    platform_metrics, split_manifest = chronological_platform_split(
        external_metrics,
        external_events,
        str(config["platform"]),
        config["split_fractions"],
    )
    if len(platform_metrics) < int(config["minimum_cascades"]):
        raise ValueError("Target platform does not meet the minimum cascade count")
    platform_metrics_path = output / "platform_temporal_metrics.parquet"
    platform_metrics.to_parquet(platform_metrics_path, index=False)
    write_json(output / "platform_split_manifest.json", split_manifest)

    profile = fit_pooled_model(platform_metrics, (), "target_platform")
    selected_config, scores, selection_manifest = select_platform_adapter(
        platform_metrics,
        profile,
        base_config,
        config["adapter_grid"],
        config["selection"]["metrics"],
        int(config["selection"]["max_validation_cascades"]),
        int(config["selection"]["seed"]),
    )
    score_path = output / "adapter_validation_scores.csv"
    scores.to_csv(score_path, index=False)
    write_json(output / "adapter_selection_manifest.json", selection_manifest)

    simulations, events = run_platform_adapter_test(
        platform_metrics,
        reddit_metrics,
        selected_config,
        base_config,
        config["test"]["seeds"],
        int(config["test"]["max_test_cascades"]),
    )
    simulations_path = output / "simulated_metrics.parquet"
    events_path = output / "events.parquet"
    simulations.to_parquet(simulations_path, index=False)
    simulations.to_csv(output / "simulated_metrics.csv", index=False)
    events.to_parquet(events_path, index=False)
    fidelity_dir = output / "evaluation"
    fidelity = evaluate_fidelity(
        platform_metrics,
        simulations,
        fidelity_dir,
        bootstrap_samples=int(config["test"]["bootstrap_samples"]),
    )
    ranking = summarize_model_ranking(fidelity)
    ranking.to_csv(fidelity_dir / "model_ranking.csv", index=False)
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "platform": str(config["platform"]),
        "n_train": int((platform_metrics["split"] == "train").sum()),
        "n_validation": int((platform_metrics["split"] == "validation").sum()),
        "n_test": int((platform_metrics["split"] == "test").sum()),
        "n_simulations": int(len(simulations)),
        "selected_adapter": asdict(selected_config),
        "best_model_by_median_normalized_wasserstein": (
            str(ranking.iloc[0]["model"]) if not ranking.empty else None
        ),
        "test_used_for_adapter_selection": False,
        "zero_shot_failure_retained": True,
        "evidence_scope": dict(config["evidence_scope"]),
        "inputs": {
            "external_metrics_sha256": sha256_file(external_metrics_path),
            "external_events_sha256": sha256_file(external_events_path),
            "reddit_metrics_sha256": sha256_file(reddit_metrics_path),
            "simulation_config_sha256": sha256_file(simulation_config_path),
        },
        "outputs": {
            "temporal_metrics_sha256": sha256_file(platform_metrics_path),
            "adapter_scores_sha256": sha256_file(score_path),
            "simulated_metrics_sha256": sha256_file(simulations_path),
            "events_sha256": sha256_file(events_path),
            "model_ranking_sha256": sha256_file(
                fidelity_dir / "model_ranking.csv"
            ),
        },
    }
    write_json(output / "platform_adapter_manifest.json", manifest)
    return manifest


def _stable_seed(*parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big", signed=False)


def _iso_max(values: pd.Series) -> str | None:
    value = values.max()
    return value.isoformat() if pd.notna(value) else None


def _iso_min(values: pd.Series) -> str | None:
    value = values.min()
    return value.isoformat() if pd.notna(value) else None
