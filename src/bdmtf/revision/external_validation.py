from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from bdmtf.revision.evaluation import evaluate_fidelity, summarize_model_ranking
from bdmtf.revision.fitted_models import fit_pooled_model
from bdmtf.revision.provenance import run_manifest, write_json
from bdmtf.revision.simulation import RevisionSimulationConfig, event_metrics, nodes_to_records, simulate_cascade


def _stable_seed(*parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def run_leave_one_community_out(
    empirical_metrics: pd.DataFrame,
    output_dir: Path,
    simulation: RevisionSimulationConfig,
    model_names: Iterable[str] = ("empirical_bootstrap", "branching_process", "hawkes", "legacy_heuristic", "learned_bdmtf"),
    seeds: Iterable[int] = (0, 1, 2),
    max_posts_per_community: int = 0,
    bootstrap_samples: int = 400,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate true zero-shot transfer to each entirely excluded community."""
    output_dir.mkdir(parents=True, exist_ok=True)
    communities = sorted(empirical_metrics["community"].dropna().astype(str).unique())
    metric_records: list[dict[str, Any]] = []
    event_records: list[dict[str, Any]] = []
    fold_records: list[dict[str, Any]] = []
    seed_values = [int(seed) for seed in seeds]
    names = [str(name) for name in model_names]

    for held_out in communities:
        target_test = empirical_metrics[
            empirical_metrics["community"].astype(str).eq(held_out) & empirical_metrics["split"].eq("test")
        ].sort_values("post_id")
        if max_posts_per_community > 0:
            target_test = target_test.head(max_posts_per_community)
        source_train = empirical_metrics[
            ~empirical_metrics["community"].astype(str).eq(held_out) & empirical_metrics["split"].eq("train")
        ]
        if target_test.empty or source_train.empty:
            continue
        profile = fit_pooled_model(empirical_metrics, (held_out,), f"pooled_without_{held_out}")
        fold_records.append(
            {
                "held_out_community": held_out,
                "source_communities": sorted(source_train["community"].astype(str).unique().tolist()),
                "n_source_train_cascades": int(len(source_train)),
                "n_target_test_cascades": int(len(target_test)),
                "target_cascades_used_for_fit": 0,
            }
        )
        for test_row in target_test.itertuples(index=False):
            for model_name in names:
                for seed in seed_values:
                    run_seed = seed + _stable_seed(held_out, str(test_row.post_id), model_name)
                    nodes = simulate_cascade(model_name, profile, source_train, str(test_row.post_id), run_seed, simulation)
                    context = {
                        "community": held_out,
                        "post_id": str(test_row.post_id),
                        "model": f"zero_shot_{model_name}",
                        "seed": seed,
                        "split": "test",
                        "fit_scope": f"all_train_except_{held_out}",
                    }
                    metric_records.append({**context, **event_metrics(nodes)})
                    event_records.extend(nodes_to_records(nodes, context))

    metrics = pd.DataFrame(metric_records)
    events = pd.DataFrame(event_records)
    if metrics.empty:
        raise ValueError("No leave-one-community-out folds could be evaluated")
    metrics.to_csv(output_dir / "simulated_metrics.csv", index=False)
    metrics.to_parquet(output_dir / "simulated_metrics.parquet", index=False)
    events.to_parquet(output_dir / "events.parquet", index=False)
    write_json(output_dir / "folds.json", fold_records)
    write_json(
        output_dir / "run_manifest.json",
        {
            **run_manifest(
                "run-leave-one-community-out",
                {
                    "simulation": simulation.__dict__,
                    "models": names,
                    "seeds": seed_values,
                    "max_posts_per_community": max_posts_per_community,
                },
            ),
            "fit_rule": "Each target community is entirely excluded from model fitting.",
            "folds": fold_records,
        },
    )

    fidelity_dir = output_dir / "evaluation"
    fidelity = evaluate_fidelity(empirical_metrics, metrics, fidelity_dir, bootstrap_samples=bootstrap_samples)
    ranking = summarize_model_ranking(fidelity)
    ranking.to_csv(fidelity_dir / "model_ranking.csv", index=False)
    summary = summarize_cross_community_fidelity(fidelity, fidelity_dir, len(metrics))
    write_json(output_dir / "summary.json", summary)
    return metrics, fidelity


def run_cross_platform_transfer(
    reddit_metrics: pd.DataFrame,
    external_metrics: pd.DataFrame,
    output_dir: Path,
    simulation: RevisionSimulationConfig,
    model_names: Iterable[str] = ("empirical_bootstrap", "branching_process", "hawkes", "legacy_heuristic", "learned_bdmtf"),
    seeds: Iterable[int] = (0, 1, 2),
    max_cascades: int = 0,
    bootstrap_samples: int = 400,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit on Reddit only and evaluate zero-shot cascade distributions on another platform."""
    output_dir.mkdir(parents=True, exist_ok=True)
    source_train = reddit_metrics[reddit_metrics["split"].eq("train")].copy()
    if source_train.empty or external_metrics.empty:
        raise ValueError("Cross-platform transfer requires Reddit training cascades and real external cascades")
    profile = fit_pooled_model(reddit_metrics, (), "pooled_reddit")
    targets = external_metrics.copy()
    targets["split"] = "test"
    targets["source_community"] = targets["community"].astype(str)
    targets["community"] = targets["platform"].astype(str) + ":" + targets["community"].astype(str)
    targets = targets.sort_values(["community", "post_id"])
    if max_cascades > 0:
        targets = targets.groupby("community", group_keys=False).head(max_cascades)

    metric_records: list[dict[str, Any]] = []
    event_records: list[dict[str, Any]] = []
    names = [str(name) for name in model_names]
    seed_values = [int(seed) for seed in seeds]
    for target in targets.itertuples(index=False):
        platform = str(target.platform)
        for model_name in names:
            for seed in seed_values:
                run_seed = seed + _stable_seed(platform, str(target.post_id), model_name)
                nodes = simulate_cascade(model_name, profile, source_train, str(target.post_id), run_seed, simulation)
                context = {
                    "platform": platform,
                    "community": str(target.community),
                    "post_id": str(target.post_id),
                    "model": f"reddit_zero_shot_{model_name}",
                    "seed": seed,
                    "split": "test",
                    "fit_scope": "reddit_train_only",
                }
                metric_records.append({**context, **event_metrics(nodes)})
                event_records.extend(nodes_to_records(nodes, context))

    metrics = pd.DataFrame(metric_records)
    events = pd.DataFrame(event_records)
    metrics.to_csv(output_dir / "simulated_metrics.csv", index=False)
    metrics.to_parquet(output_dir / "simulated_metrics.parquet", index=False)
    events.to_parquet(output_dir / "events.parquet", index=False)
    targets.to_csv(output_dir / "empirical_metrics.csv", index=False)
    targets.to_parquet(output_dir / "empirical_metrics.parquet", index=False)
    write_json(
        output_dir / "run_manifest.json",
        {
            **run_manifest(
                "run-cross-platform-transfer",
                {
                    "simulation": simulation.__dict__,
                    "models": names,
                    "seeds": seed_values,
                    "max_cascades": max_cascades,
                },
            ),
            "source_platform": "Reddit",
            "target_platforms": sorted(targets["platform"].astype(str).unique().tolist()),
            "target_cascades_used_for_fit": 0,
            "n_source_train_cascades": int(len(source_train)),
            "n_target_cascades": int(len(targets)),
        },
    )
    fidelity_dir = output_dir / "evaluation"
    fidelity = evaluate_fidelity(targets, metrics, fidelity_dir, bootstrap_samples=bootstrap_samples)
    ranking = summarize_model_ranking(fidelity)
    ranking.to_csv(fidelity_dir / "model_ranking.csv", index=False)
    summary = summarize_cross_platform_fidelity(fidelity, targets, len(metrics))
    write_json(output_dir / "summary.json", summary)
    return metrics, fidelity


def summarize_cross_platform_fidelity(
    fidelity: pd.DataFrame,
    targets: pd.DataFrame,
    n_simulations: int,
) -> dict[str, Any]:
    ranking = summarize_model_ranking(fidelity)
    return {
        "status": "complete",
        "reviewer_external_validity_status": "partially_addressed",
        "source_platform": "Reddit",
        "target_platforms": sorted(targets["platform"].astype(str).unique().tolist()),
        "n_target_cascades": int(len(targets)),
        "n_target_cascades_by_platform": {
            str(platform): int(count)
            for platform, count in targets.groupby("platform").size().items()
        },
        "n_simulations": int(n_simulations),
        "best_model_by_mean_normalized_wasserstein": str(ranking.sort_values("mean").iloc[0]["model"]),
        "best_model_by_median_normalized_wasserstein": str(ranking.sort_values("median").iloc[0]["model"]),
        "causal_intervention_validated": False,
        "claim_boundary": (
            "Bounded official-API samples from the listed threaded platforms do not establish "
            "universal cross-platform transportability or intervention causality."
        ),
    }


def summarize_cross_community_fidelity(fidelity: pd.DataFrame, output_dir: Path, n_simulations: int) -> dict[str, Any]:
    ranking = summarize_model_ranking(fidelity)
    by_community = (
        fidelity.groupby(["community", "model"])["normalized_wasserstein"]
        .agg(["mean", "median", "std", "count"])
        .reset_index()
    )
    by_community["mean_rank"] = by_community.groupby("community")["mean"].rank(method="min")
    by_community["median_rank"] = by_community.groupby("community")["median"].rank(method="min")
    by_community.to_csv(output_dir / "ranking_by_community.csv", index=False)
    by_metric = (
        fidelity.groupby(["metric", "model"])["normalized_wasserstein"]
        .agg(["mean", "median", "std", "count"])
        .reset_index()
    )
    by_metric["mean_rank"] = by_metric.groupby("metric")["mean"].rank(method="min")
    by_metric["median_rank"] = by_metric.groupby("metric")["median"].rank(method="min")
    by_metric.to_csv(output_dir / "ranking_by_metric.csv", index=False)
    mean_winners = by_community[by_community["mean_rank"].eq(1)]["model"].value_counts().to_dict()
    median_winners = by_community[by_community["median_rank"].eq(1)]["model"].value_counts().to_dict()
    best_mean = ranking.sort_values("mean").iloc[0] if not ranking.empty else None
    best_median = ranking.sort_values("median").iloc[0] if not ranking.empty else None
    return {
        "status": "complete",
        "evidence_level": "cross-community zero-shot within five Reddit communities",
        "reviewer_external_validity_status": "partially_addressed",
        "n_communities": int(fidelity["community"].nunique()),
        "n_simulations": int(n_simulations),
        "best_model_by_mean_normalized_wasserstein": str(best_mean["model"]) if best_mean is not None else None,
        "best_mean_normalized_wasserstein": float(best_mean["mean"]) if best_mean is not None else None,
        "best_model_by_median_normalized_wasserstein": str(best_median["model"]) if best_median is not None else None,
        "best_median_normalized_wasserstein": float(best_median["median"]) if best_median is not None else None,
        "community_winners_by_mean": {str(key): int(value) for key, value in mean_winners.items()},
        "community_winners_by_median": {str(key): int(value) for key, value in median_winners.items()},
        "cross_platform_validated": False,
        "real_intervention_validated": False,
        "broad_cross_community_claim_ready": False,
        "claim_boundary": "This result does not establish transfer to unseen platforms or causal intervention effects.",
    }
