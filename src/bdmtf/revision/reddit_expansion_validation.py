from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from bdmtf.revision.data_pipeline import _text_flags
from bdmtf.revision.evaluation import evaluate_fidelity, summarize_model_ranking
from bdmtf.revision.fitted_models import (
    CommunityModel,
    fit_community_model,
    fit_pooled_model,
)
from bdmtf.revision.provenance import run_manifest, sha256_file, write_json
from bdmtf.revision.simulation import (
    MODEL_NAMES,
    RevisionSimulationConfig,
    event_metrics,
    nodes_to_records,
    simulate_cascade,
)


def _stable_seed(*parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def _gini(values: np.ndarray) -> float:
    values = np.sort(np.maximum(np.asarray(values, dtype=float), 0.0))
    if len(values) == 0 or values.sum() == 0:
        return 0.0
    index = np.arange(1, len(values) + 1)
    return float(
        np.sum((2 * index - len(values) - 1) * values)
        / (len(values) * values.sum())
    )


def _empty_metrics() -> dict[str, float]:
    return {
        "size": 0.0,
        "max_depth": 0.0,
        "mean_leaf_depth": 0.0,
        "mean_depth": 0.0,
        "mean_branching_factor": 0.0,
        "root_reply_share": 0.0,
        "width_gini": 0.0,
        "time_to_50_minutes": 0.0,
        "time_to_90_minutes": 0.0,
        "duration_minutes": 0.0,
        "mean_interarrival_minutes": 0.0,
        "repeat_author_share": 0.0,
        "toxicity_density": 0.0,
        "counterspeech_rate": 0.0,
        "removed_rate": 0.0,
        "n_leaf_nodes": 0,
    }


def _cascade_metrics(group: pd.DataFrame) -> dict[str, Any]:
    roots = group[group["event_type"].eq("post")]
    if len(roots) != 1:
        raise ValueError("Each imported cascade must contain exactly one root post")
    root = roots.iloc[0]
    comments = group[group["event_type"].eq("comment")].copy()
    result: dict[str, Any] = {
        "community": str(root["community"]),
        "post_id": str(root["content_id"]),
        "created_at": pd.to_datetime(root["created_at"], utc=True),
        "is_viral": int(bool(root.get("is_viral", False))),
    }
    if comments.empty:
        return {**result, **_empty_metrics()}

    depths = (
        pd.to_numeric(comments["depth"], errors="coerce")
        .fillna(1)
        .clip(lower=2)
        .astype(int)
        - 1
    )
    comments["_metric_depth"] = depths
    comment_ids = set(comments["event_id"].astype(str))
    child_counts = comments["parent_event_id"].astype(str).value_counts()
    leaves = comments.loc[~comments["event_id"].astype(str).isin(child_counts.index)]
    root_replies = comments["parent_event_id"].astype(str).eq(str(root["event_id"]))
    width = comments.groupby("_metric_depth").size().to_numpy(dtype=float)
    non_leaf_children = child_counts[
        child_counts.index.astype(str).isin(comment_ids)
    ].to_numpy(dtype=float)

    created = pd.to_datetime(comments["created_at"], utc=True, errors="coerce")
    minutes = (
        (created - pd.to_datetime(root["created_at"], utc=True))
        .dt.total_seconds()
        .div(60.0)
    )
    minutes = np.sort(
        minutes[(minutes >= 0) & np.isfinite(minutes)].to_numpy(dtype=float)
    )
    authors = comments["author_id"].fillna("").astype(str)
    author_counts = authors[
        ~authors.isin({"", "[deleted]", "None", "nan"})
    ].value_counts()
    toxic, counterspeech, text_removed = _text_flags(comments["text"])
    removed = comments["removed"].fillna(False).astype(bool) | text_removed
    size = len(comments)
    return {
        **result,
        "size": float(size),
        "max_depth": float(depths.max()),
        "mean_leaf_depth": (
            float(leaves["_metric_depth"].mean()) if len(leaves) else 0.0
        ),
        "mean_depth": float(depths.mean()),
        "mean_branching_factor": (
            float(np.mean(non_leaf_children)) if len(non_leaf_children) else 0.0
        ),
        "root_reply_share": float(root_replies.mean()),
        "width_gini": _gini(width),
        "time_to_50_minutes": (
            float(np.quantile(minutes, 0.5)) if len(minutes) else 0.0
        ),
        "time_to_90_minutes": (
            float(np.quantile(minutes, 0.9)) if len(minutes) else 0.0
        ),
        "duration_minutes": (
            float(minutes.max() - minutes.min()) if len(minutes) > 1 else 0.0
        ),
        "mean_interarrival_minutes": (
            float(np.diff(minutes).mean()) if len(minutes) > 1 else 0.0
        ),
        "repeat_author_share": (
            float(author_counts[author_counts > 1].sum() / size) if size else 0.0
        ),
        "toxicity_density": float(toxic.mean()),
        "counterspeech_rate": float(counterspeech.mean()),
        "removed_rate": float(removed.mean()),
        "n_leaf_nodes": int(len(leaves)),
    }


def _assign_splits(
    metrics: pd.DataFrame,
    train_fraction: float,
    validation_fraction: float,
) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    for (_, _), group in metrics.groupby(
        ["community", "is_viral"],
        dropna=False,
        sort=True,
    ):
        ordered = group.sort_values(["created_at", "post_id"]).copy()
        count = len(ordered)
        train_end = int(np.floor(count * train_fraction))
        validation_end = train_end + int(np.floor(count * validation_fraction))
        assignments = np.full(count, "test", dtype=object)
        assignments[:train_end] = "train"
        assignments[train_end:validation_end] = "validation"
        ordered["split"] = assignments
        records.append(ordered)
    split = pd.concat(records, ignore_index=True)
    if split.duplicated(["community", "post_id"]).any():
        raise ValueError("Expanded Reddit split contains duplicate cascade assignments")
    return split.sort_values(["community", "created_at", "post_id"])


def build_expanded_reddit_metrics(
    community_dir: Path,
    output_dir: Path,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    per_community_dir = output_dir / "metrics_by_community"
    per_community_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    inventory: list[dict[str, Any]] = []
    sources = sorted(community_dir.glob("*.parquet"))
    if not sources:
        raise FileNotFoundError(f"No normalized community Parquet files in {community_dir}")

    for index, source in enumerate(sources, start=1):
        community = source.stem
        target = per_community_dir / f"{community}.parquet"
        source_hash = sha256_file(source)
        identity_path = target.with_suffix(".source.json")
        cached_hash = ""
        if identity_path.is_file():
            try:
                import json

                cached_hash = str(
                    json.loads(identity_path.read_text(encoding="utf-8")).get(
                        "source_sha256",
                        "",
                    )
                )
            except (OSError, ValueError):
                cached_hash = ""
        if target.is_file() and cached_hash == source_hash:
            frame = pd.read_parquet(target)
        else:
            events = pd.read_parquet(source)
            rows = [
                _cascade_metrics(group)
                for _, group in events.groupby("content_id", sort=False)
            ]
            frame = pd.DataFrame(rows)
            partial = target.with_suffix(".parquet.part")
            frame.to_parquet(partial, index=False)
            partial.replace(target)
            write_json(identity_path, {"source_sha256": source_hash})
        frames.append(frame)
        inventory.append(
            {
                "community": community,
                "cascades": int(len(frame)),
                "comments": int(frame["size"].sum()),
                "source_sha256": source_hash,
            }
        )
        write_json(
            output_dir / "metric_progress.json",
            {
                "status": "running",
                "completed": index,
                "total": len(sources),
                "last_community": community,
                "cascades_so_far": sum(item["cascades"] for item in inventory),
            },
        )

    metrics = pd.concat(frames, ignore_index=True)
    metrics = _assign_splits(metrics, train_fraction, validation_fraction)
    metrics_path = output_dir / "empirical_cascade_metrics.parquet"
    metrics.to_parquet(metrics_path, index=False)
    metrics.to_csv(output_dir / "empirical_cascade_metrics.csv", index=False)
    split_columns = ["community", "post_id", "is_viral", "created_at", "split"]
    metrics[split_columns].to_parquet(output_dir / "post_splits.parquet", index=False)
    metrics[split_columns].to_csv(output_dir / "post_splits.csv", index=False)

    split_counts = (
        metrics.groupby(["community", "split"]).size().rename("n").reset_index()
    )
    manifest = {
        "status": "complete",
        "method": "chronological within community and viral label",
        "fractions": {
            "train": train_fraction,
            "validation": validation_fraction,
            "test": 1.0 - train_fraction - validation_fraction,
        },
        "n_communities": int(metrics["community"].nunique()),
        "n_cascades": int(len(metrics)),
        "n_cascades_with_comments": int((metrics["size"] > 0).sum()),
        "n_comments": int(metrics["size"].sum()),
        "duplicate_assignments": 0,
        "inventory": inventory,
        "split_counts": split_counts.to_dict(orient="records"),
        "metrics_sha256": sha256_file(metrics_path),
        "metric_definition": "Top-level Reddit comments have depth 1.",
    }
    write_json(output_dir / "metrics_manifest.json", manifest)
    write_json(
        output_dir / "metric_progress.json",
        {
            "status": "complete",
            "completed": len(sources),
            "total": len(sources),
            "cascades_so_far": int(len(metrics)),
        },
    )
    return metrics, manifest


class _EventWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.partial = path.with_suffix(".parquet.part")
        self.partial.unlink(missing_ok=True)
        self.writer: pq.ParquetWriter | None = None
        self.records: list[dict[str, Any]] = []

    def add(self, records: list[dict[str, Any]]) -> None:
        self.records.extend(records)
        if len(self.records) >= 100_000:
            self.flush()

    def flush(self) -> None:
        if not self.records:
            return
        table = pa.Table.from_pylist(self.records)
        if self.writer is None:
            self.writer = pq.ParquetWriter(
                self.partial,
                table.schema,
                compression="snappy",
            )
        self.writer.write_table(table)
        self.records.clear()

    def close(self) -> None:
        self.flush()
        if self.writer is not None:
            self.writer.close()
            self.partial.replace(self.path)


def _profiles_for_protocol(
    empirical: pd.DataFrame,
    protocol: str,
) -> dict[str, CommunityModel]:
    communities = sorted(empirical["community"].astype(str).unique())
    if protocol == "target_adapted":
        return {
            community: fit_community_model(empirical, community)
            for community in communities
        }
    if protocol == "leave_one_community_out":
        return {
            community: fit_pooled_model(
                empirical,
                (community,),
                f"pooled_without_{community}",
            )
            for community in communities
        }
    raise ValueError(f"Unknown Reddit expansion protocol: {protocol}")


def run_expanded_reddit_protocol(
    empirical: pd.DataFrame,
    output_dir: Path,
    protocol: str,
    simulation: RevisionSimulationConfig,
    model_names: Iterable[str] = MODEL_NAMES,
    seeds: Iterable[int] = (0, 1, 2),
    bootstrap_samples: int = 400,
    max_test_cascades_per_community: int = 0,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    names = [str(name) for name in model_names]
    seed_values = [int(seed) for seed in seeds]
    profiles = _profiles_for_protocol(empirical, protocol)
    train = empirical[empirical["split"].eq("train")]
    test = empirical[empirical["split"].eq("test")]
    metric_records: list[dict[str, Any]] = []
    event_writer = _EventWriter(output_dir / "events.parquet")
    completed = 0
    total = sum(
        min(len(group), max_test_cascades_per_community)
        if max_test_cascades_per_community > 0
        else len(group)
        for _, group in test.groupby("community")
    )

    try:
        for community, target_rows in test.groupby("community", sort=True):
            target_rows = target_rows.sort_values(["created_at", "post_id"])
            if max_test_cascades_per_community > 0:
                target_rows = target_rows.head(max_test_cascades_per_community)
            if protocol == "target_adapted":
                train_rows = train[train["community"].eq(community)]
                fit_scope = f"{community}_train_only"
            else:
                train_rows = train[~train["community"].eq(community)]
                fit_scope = f"all_train_except_{community}"
            for target in target_rows.itertuples(index=False):
                for model_name in names:
                    for seed in seed_values:
                        run_seed = seed + _stable_seed(
                            protocol,
                            str(community),
                            str(target.post_id),
                            model_name,
                        )
                        nodes = simulate_cascade(
                            model_name,
                            profiles[str(community)],
                            train_rows,
                            str(target.post_id),
                            run_seed,
                            simulation,
                        )
                        context = {
                            "protocol": protocol,
                            "community": str(community),
                            "post_id": str(target.post_id),
                            "model": model_name,
                            "seed": seed,
                            "split": "test",
                            "fit_scope": fit_scope,
                        }
                        metric_records.append({**context, **event_metrics(nodes)})
                        event_writer.add(nodes_to_records(nodes, context))
                completed += 1
                if completed % 25 == 0:
                    write_json(
                        output_dir / "progress.json",
                        {
                            "status": "running",
                            "protocol": protocol,
                            "completed_test_cascades": completed,
                            "total_test_cascades": total,
                            "simulations_completed": len(metric_records),
                        },
                    )
    finally:
        event_writer.close()

    simulated = pd.DataFrame(metric_records)
    simulated.to_parquet(output_dir / "simulated_metrics.parquet", index=False)
    simulated.to_csv(output_dir / "simulated_metrics.csv", index=False)
    fidelity_dir = output_dir / "evaluation"
    fidelity = evaluate_fidelity(
        empirical,
        simulated,
        fidelity_dir,
        bootstrap_samples=bootstrap_samples,
    )
    ranking = summarize_model_ranking(fidelity)
    ranking.to_csv(fidelity_dir / "model_ranking.csv", index=False)
    ranking_records = ranking.to_dict(orient="records")
    best_mean = ranking.sort_values("mean").iloc[0]
    best_median = ranking.sort_values("median").iloc[0]
    bdmtf = ranking[ranking["model"].eq("learned_bdmtf")]
    manifest = {
        **run_manifest(
            f"run-expanded-reddit-{protocol}",
            {
                "models": names,
                "seeds": seed_values,
                "simulation": simulation.__dict__,
                "bootstrap_samples": bootstrap_samples,
                "max_test_cascades_per_community": max_test_cascades_per_community,
            },
        ),
        "status": "complete",
        "protocol": protocol,
        "n_communities": int(test["community"].nunique()),
        "n_test_cascades": int(completed),
        "n_simulations": int(len(simulated)),
        "best_model_by_mean_normalized_wasserstein": str(best_mean["model"]),
        "best_mean_normalized_wasserstein": float(best_mean["mean"]),
        "best_model_by_median_normalized_wasserstein": str(best_median["model"]),
        "best_median_normalized_wasserstein": float(best_median["median"]),
        "learned_bdmtf_mean_normalized_wasserstein": (
            float(bdmtf.iloc[0]["mean"]) if not bdmtf.empty else None
        ),
        "learned_bdmtf_median_normalized_wasserstein": (
            float(bdmtf.iloc[0]["median"]) if not bdmtf.empty else None
        ),
        "ranking": ranking_records,
        "claim_boundary": (
            "This test broadens within-Reddit community coverage. It does not by "
            "itself establish cross-platform or causal-intervention validity."
        ),
    }
    write_json(output_dir / "summary.json", manifest)
    write_json(
        output_dir / "progress.json",
        {
            "status": "complete",
            "protocol": protocol,
            "completed_test_cascades": completed,
            "total_test_cascades": total,
            "simulations_completed": len(simulated),
        },
    )
    return manifest


def run_reddit_expansion_validation(
    expansion_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    import json

    import_manifest_path = expansion_dir / "expansion_manifest.json"
    if not import_manifest_path.is_file():
        raise FileNotFoundError("Run expand-reddit before Reddit expansion validation")
    import_manifest = json.loads(import_manifest_path.read_text(encoding="utf-8"))
    if not import_manifest.get("target_met"):
        raise ValueError("Reddit expansion data did not meet the frozen coverage target")

    metrics_dir = expansion_dir / "metrics"
    empirical, metrics_manifest = build_expanded_reddit_metrics(
        expansion_dir / "communities",
        metrics_dir,
        float(config.get("train_fraction", 0.6)),
        float(config.get("validation_fraction", 0.2)),
    )
    simulation = RevisionSimulationConfig.from_dict(config.get("simulation", {}))
    protocols = config.get(
        "protocols",
        ["target_adapted", "leave_one_community_out"],
    )
    summaries: dict[str, dict[str, Any]] = {}
    for protocol in protocols:
        summaries[str(protocol)] = run_expanded_reddit_protocol(
            empirical,
            expansion_dir / "validation" / str(protocol),
            str(protocol),
            simulation,
            config.get("models", MODEL_NAMES),
            config.get("seeds", [0, 1, 2]),
            int(config.get("bootstrap_samples", 400)),
            int(config.get("max_test_cascades_per_community", 0)),
        )
    result = {
        "status": "complete",
        "reviewer_issue": "Original evidence was limited to five selected Reddit communities.",
        "n_communities": metrics_manifest["n_communities"],
        "n_cascades": metrics_manifest["n_cascades"],
        "n_comments": metrics_manifest["n_comments"],
        "protocols": summaries,
        "claim_allowed": bool(metrics_manifest["n_communities"] >= 20)
        and bool(metrics_manifest["n_cascades"] >= 10_000),
        "claim_boundary": (
            "The expanded evidence supports breadth across the sampled Reddit "
            "community types; platform-level generalization remains a separate test."
        ),
    }
    write_json(expansion_dir / "validation_summary.json", result)
    return result
