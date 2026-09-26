from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from bdmtf.revision.lemmy_agent_replay import (
    _event_groups,
    _prepare_inputs,
    _simulate_matches,
    fit_agent_replay_model,
)


MODEL_BDMTF = "bdmtf_training_path"
MODEL_BDMTF_AGGREGATE = "bdmtf_aggregate_path"
MODEL_EMPIRICAL = "empirical_nearest_path"
METRICS = (
    "cumulative_count_nmae",
    "arrival_time_nwd",
    "depth_nwd",
    "root_share_ae",
    "parent_hhi_ae",
    "composite_path_error",
)


@dataclass(frozen=True)
class PathView:
    times: np.ndarray
    depths: np.ndarray
    daily_counts: np.ndarray
    root_share: float
    parent_hhi: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def _empty_path(horizon_days: int) -> PathView:
    return PathView(
        times=np.array([], dtype=float),
        depths=np.array([], dtype=float),
        daily_counts=np.zeros(horizon_days, dtype=float),
        root_share=0.0,
        parent_hhi=0.0,
    )


def _make_path(
    frame: pd.DataFrame,
    *,
    time_column: str,
    depth_column: str,
    parent_column: str,
    root_id: str,
    horizon_days: int,
) -> PathView:
    if frame.empty:
        return _empty_path(horizon_days)
    times = pd.to_numeric(frame[time_column], errors="coerce").to_numpy(float)
    depths = pd.to_numeric(frame[depth_column], errors="coerce").fillna(0).to_numpy(float)
    valid = np.isfinite(times) & (times >= 0) & (times < horizon_days * 1440.0)
    times = times[valid]
    depths = depths[valid]
    if not valid.any():
        return _empty_path(horizon_days)
    parents = frame.loc[valid, parent_column].fillna("").astype(str)
    day = np.minimum((times // 1440.0).astype(int), horizon_days - 1)
    daily = np.bincount(day, minlength=horizon_days).astype(float)
    root_share = float((parents == str(root_id)).mean())
    shares = parents.value_counts(normalize=True).to_numpy(float)
    parent_hhi = float(np.square(shares).sum())
    return PathView(times, depths, daily, root_share, parent_hhi)


def _path_errors(
    observed: PathView,
    predicted: PathView,
    *,
    horizon_days: int,
    count_scale: float,
    depth_scale: float,
) -> dict[str, float]:
    obs_empty = observed.times.size == 0
    pred_empty = predicted.times.size == 0
    if obs_empty and pred_empty:
        arrival = depth = root = parent = 0.0
    elif obs_empty or pred_empty:
        arrival = depth = root = parent = 1.0
    else:
        arrival = float(
            wasserstein_distance(observed.times, predicted.times)
            / (horizon_days * 1440.0)
        )
        depth = float(
            wasserstein_distance(observed.depths, predicted.depths) / depth_scale
        )
        root = abs(observed.root_share - predicted.root_share)
        parent = abs(observed.parent_hhi - predicted.parent_hhi)
    observed_cumulative = np.cumsum(observed.daily_counts)
    predicted_cumulative = np.cumsum(predicted.daily_counts)
    count = float(
        np.mean(
            np.abs(np.log1p(observed_cumulative) - np.log1p(predicted_cumulative))
        )
        / np.log1p(count_scale)
    )
    components = np.clip([count, arrival, depth, root, parent], 0.0, 1.0)
    return {
        "cumulative_count_nmae": count,
        "arrival_time_nwd": arrival,
        "depth_nwd": depth,
        "root_share_ae": root,
        "parent_hhi_ae": parent,
        "composite_path_error": float(np.mean(components)),
    }


def _feature_columns(features: pd.DataFrame) -> list[str]:
    stems = (
        "treated_age_hours",
        "distance",
        "pretrajectory_distance",
        "match_score",
        "reply_count__treated_",
        "active_authors__treated_",
        "max_depth__treated_",
    )
    return [
        column
        for column in features.columns
        if pd.api.types.is_numeric_dtype(features[column])
        and any(column == stem or column.startswith(stem) for stem in stems)
    ]


def _nearest_training_ids(
    features: pd.DataFrame,
    test_row: pd.Series,
    *,
    feature_columns: list[str],
    k: int,
) -> list[str]:
    candidates = features[
        (features["split"] == "train")
        & (features["intervention_type"] == test_row["intervention_type"])
    ].copy()
    same_community = candidates[
        candidates["community_id"].astype(str) == str(test_row["community_id"])
    ]
    if len(same_community) >= min(5, k):
        candidates = same_community
    train_reference = features[features["split"] == "train"]
    median = train_reference[feature_columns].median()
    scale = train_reference[feature_columns].std().replace(0.0, 1.0).fillna(1.0)
    matrix = candidates[feature_columns].astype(float).fillna(median)
    target = test_row[feature_columns].astype(float).fillna(median)
    candidates["_distance"] = np.sqrt(
        np.square((matrix - target) / scale).mean(axis=1)
    )
    return (
        candidates.sort_values(["_distance", "intervention_id"])
        .head(k)["intervention_id"]
        .astype(str)
        .tolist()
    )


def _bootstrap_interval(
    values: pd.Series, samples: int, rng: np.random.Generator
) -> tuple[float, float]:
    array = values.dropna().to_numpy(float)
    if array.size == 0:
        return float("nan"), float("nan")
    draws = np.empty(samples, dtype=float)
    for index in range(samples):
        draws[index] = rng.choice(array, size=array.size, replace=True).mean()
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _observed_path(
    event_groups: dict[str, pd.DataFrame],
    content_id: str,
    event_time: pd.Timestamp,
    horizon_days: int,
) -> PathView:
    group = event_groups.get(str(content_id))
    if group is None or group.empty:
        return _empty_path(horizon_days)
    end = event_time + pd.Timedelta(days=horizon_days)
    post = group[(group["created_at"] >= event_time) & (group["created_at"] < end)].copy()
    post["relative_minute"] = (
        post["created_at"] - event_time
    ).dt.total_seconds() / 60.0
    return _make_path(
        post,
        time_column="relative_minute",
        depth_column="depth",
        parent_column="parent_event_id",
        root_id=str(content_id),
        horizon_days=horizon_days,
    )


def _simulated_path(
    simulated: pd.DataFrame,
    intervention_id: str,
    seed: int,
    horizon_days: int,
) -> PathView:
    group = simulated[
        (simulated["intervention_id"].astype(str) == intervention_id)
        & (simulated["seed"] == seed)
        & (simulated["condition"] == "intervention")
    ]
    return _make_path(
        group,
        time_column="created_minute",
        depth_column="depth",
        parent_column="parent_id",
        root_id="post",
        horizon_days=horizon_days,
    )


def run_lemmy_path_fidelity(
    root: Path, config: dict[str, Any], output_override: str | None = None
) -> dict[str, Any]:
    paths = {key: root / value for key, value in config["inputs"].items()}
    output = root / (output_override or config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    horizon_days = int(config.get("horizon_days", 8))
    nearest_paths = int(config.get("nearest_paths", 20))
    bootstrap_samples = int(config.get("bootstrap_samples", 2000))
    rng = np.random.default_rng(int(config.get("seed", 30371)))

    matches = _read(paths["matches"])
    splits = _read(paths["splits"])[["intervention_id", "split"]]
    features = _read(paths["features"])
    observed_events = _read(paths["observed_events"])
    simulated = _read(paths["simulated_events"])
    panel = _read(paths["panel"])
    posts = _read(paths["posts"])
    agent_manifest = json.loads(paths["agent_manifest"].read_text(encoding="utf-8"))
    agent_config = json.loads(paths["agent_config"].read_text(encoding="utf-8"))

    prepared_panel, prepared_matches, _, prepared_events, prepared_posts = _prepare_inputs(
        panel.copy(),
        _read(paths["matches"]),
        _read(paths["splits"]),
        observed_events.copy(),
        posts.copy(),
    )
    simulation_config = agent_config["simulation"]
    intensity_model, replay_parameters, _ = fit_agent_replay_model(
        prepared_matches,
        prepared_events,
        prepared_posts,
        simulation_config,
    )
    training_ids = set(
        prepared_matches.loc[
            prepared_matches["split"] == "train", "intervention_id"
        ].astype(str)
    )
    training_counts = prepared_panel[
        prepared_panel["intervention_id"].astype(str).isin(training_ids)
        & (prepared_panel["relative_period"] >= 0)
        & (prepared_panel["outcome"] == "reply_count")
    ]
    removal_counts = training_counts[
        training_counts["intervention_type"] == "remove_post"
    ].groupby("treated")["value"].sum()
    remove_visibility = float(
        np.clip(
            removal_counts.get(1, 0.0) / max(removal_counts.get(0, 0.0), 1.0),
            0.0,
            1.0,
        )
    )
    intensity_scale = float(
        agent_manifest["selected_validation_parameters"]["intensity_scale"]
    )
    test_matches = prepared_matches[prepared_matches["split"] == "test"].copy()
    replay_groups = _event_groups(prepared_events)
    post_times = dict(
        zip(prepared_posts["content_id"].astype(str), prepared_posts["published"])
    )
    seeds = list(range(int(agent_manifest["test_simulation_seeds"])))
    _, path_calibrated_events = _simulate_matches(
        selected_matches=test_matches,
        model=intensity_model,
        parameters=replay_parameters,
        event_groups=replay_groups,
        post_times=post_times,
        horizon_days=horizon_days,
        intensity_scale=intensity_scale,
        remove_visibility_remaining=remove_visibility,
        seeds=seeds,
        max_events_per_day=int(simulation_config.get("max_events_per_day", 500)),
        collect_events=True,
    )
    path_calibrated_events.to_parquet(
        output / "path_calibrated_simulated_events.parquet", index=False
    )

    matches["intervention_id"] = matches["intervention_id"].astype(str)
    matches["event_time"] = pd.to_datetime(
        matches["event_time"], utc=True, format="mixed"
    )
    matches = matches.merge(splits, on="intervention_id", how="inner")
    features["intervention_id"] = features["intervention_id"].astype(str)
    observed_events["content_id"] = observed_events["content_id"].astype(str)
    observed_events["created_at"] = pd.to_datetime(
        observed_events["created_at"], utc=True, format="mixed"
    )
    event_groups = {
        str(content_id): group.copy()
        for content_id, group in observed_events.groupby("content_id", sort=False)
    }
    match_lookup = matches.set_index("intervention_id")
    feature_columns = _feature_columns(features)
    feature_lookup = features.set_index("intervention_id")

    training_paths: dict[str, PathView] = {}
    for row in matches[matches["split"] == "train"].itertuples(index=False):
        training_paths[str(row.intervention_id)] = _observed_path(
            event_groups,
            str(row.treated_content_id),
            row.event_time,
            horizon_days,
        )
    train_counts = [path.daily_counts.sum() for path in training_paths.values()]
    train_depths = np.concatenate(
        [path.depths for path in training_paths.values() if path.depths.size],
        dtype=float,
    )
    count_scale = max(1.0, float(np.quantile(train_counts, 0.95)))
    depth_scale = max(1.0, float(np.quantile(train_depths, 0.95)))
    score_rows: list[dict[str, Any]] = []
    neighbor_rows: list[dict[str, Any]] = []
    test_features = features[features["split"] == "test"].copy()
    for test in test_features.itertuples(index=False):
        intervention_id = str(test.intervention_id)
        match = match_lookup.loc[intervention_id]
        observed = _observed_path(
            event_groups,
            str(match["treated_content_id"]),
            match["event_time"],
            horizon_days,
        )
        for seed in seeds:
            aggregate_predicted = _simulated_path(
                simulated, intervention_id, seed, horizon_days
            )
            calibrated_predicted = _simulated_path(
                path_calibrated_events, intervention_id, seed, horizon_days
            )
            aggregate_errors = _path_errors(
                observed,
                aggregate_predicted,
                horizon_days=horizon_days,
                count_scale=count_scale,
                depth_scale=depth_scale,
            )
            calibrated_errors = _path_errors(
                observed,
                calibrated_predicted,
                horizon_days=horizon_days,
                count_scale=count_scale,
                depth_scale=depth_scale,
            )
            for model_name, errors in (
                (MODEL_BDMTF_AGGREGATE, aggregate_errors),
                (MODEL_BDMTF, calibrated_errors),
            ):
                for metric, value in errors.items():
                    score_rows.append(
                        {
                            "intervention_id": intervention_id,
                            "intervention_type": test.intervention_type,
                            "model": model_name,
                            "draw": seed,
                            "metric": metric,
                            "error": value,
                        }
                    )
        test_series = feature_lookup.loc[intervention_id]
        neighbors = _nearest_training_ids(
            features,
            test_series,
            feature_columns=feature_columns,
            k=nearest_paths,
        )
        for draw, neighbor_id in enumerate(neighbors):
            neighbor_rows.append(
                {
                    "intervention_id": intervention_id,
                    "neighbor_intervention_id": neighbor_id,
                    "draw": draw,
                }
            )
            errors = _path_errors(
                observed,
                training_paths[neighbor_id],
                horizon_days=horizon_days,
                count_scale=count_scale,
                depth_scale=depth_scale,
            )
            for metric, value in errors.items():
                score_rows.append(
                    {
                        "intervention_id": intervention_id,
                        "intervention_type": test.intervention_type,
                        "model": MODEL_EMPIRICAL,
                        "draw": draw,
                        "metric": metric,
                        "error": value,
                    }
                )

    draw_scores = pd.DataFrame(score_rows)
    intervention_scores = (
        draw_scores.groupby(
            ["intervention_id", "intervention_type", "model", "metric"],
            as_index=False,
        )["error"]
        .mean()
    )
    summary_rows: list[dict[str, Any]] = []
    for (model, metric), group in intervention_scores.groupby(["model", "metric"]):
        low, high = _bootstrap_interval(group["error"], bootstrap_samples, rng)
        summary_rows.append(
            {
                "model": model,
                "metric": metric,
                "mean_error": float(group["error"].mean()),
                "ci_low": low,
                "ci_high": high,
                "n_interventions": int(group["intervention_id"].nunique()),
            }
        )
    summary = pd.DataFrame(summary_rows)

    pivot = intervention_scores.pivot(
        index=["intervention_id", "intervention_type"],
        columns=["model", "metric"],
        values="error",
    )
    comparison_rows: list[dict[str, Any]] = []
    for metric in METRICS:
        improvement = pivot[(MODEL_EMPIRICAL, metric)] - pivot[(MODEL_BDMTF, metric)]
        low, high = _bootstrap_interval(improvement, bootstrap_samples, rng)
        comparison_rows.append(
            {
                "metric": metric,
                "bdmtf_improvement_over_empirical": float(improvement.mean()),
                "ci_low": low,
                "ci_high": high,
                "bdmtf_lower_error_share": float((improvement > 0).mean()),
            }
        )
    comparisons = pd.DataFrame(comparison_rows)

    draw_scores.to_csv(output / "path_draw_scores.csv", index=False)
    intervention_scores.to_csv(output / "path_intervention_scores.csv", index=False)
    summary.to_csv(output / "path_fidelity_summary.csv", index=False)
    comparisons.to_csv(output / "paired_model_comparisons.csv", index=False)
    pd.DataFrame(neighbor_rows).to_csv(output / "empirical_neighbors.csv", index=False)

    composite = summary[summary["metric"] == "composite_path_error"].set_index("model")
    composite_cmp = comparisons[
        comparisons["metric"] == "composite_path_error"
    ].iloc[0]
    manifest = {
        "status": "complete",
        "protocol": "chronological held-out event-path fidelity",
        "test_interventions": int(test_features["intervention_id"].nunique()),
        "bdmtf_draws_per_intervention": len(seeds),
        "empirical_draws_per_intervention": nearest_paths,
        "metrics": list(METRICS),
        "scales_fitted_on_training_only": {
            "cumulative_count_q95": count_scale,
            "depth_q95": depth_scale,
        },
        "path_policy_fitted_on_training_only": {
            "intensity_scale": intensity_scale,
            "remove_visibility_remaining": remove_visibility,
            "lock_visibility_remaining": 0.0,
        },
        "composite": {
            MODEL_BDMTF: float(composite.loc[MODEL_BDMTF, "mean_error"]),
            MODEL_BDMTF_AGGREGATE: float(
                composite.loc[MODEL_BDMTF_AGGREGATE, "mean_error"]
            ),
            MODEL_EMPIRICAL: float(composite.loc[MODEL_EMPIRICAL, "mean_error"]),
            "bdmtf_improvement": float(
                composite_cmp["bdmtf_improvement_over_empirical"]
            ),
            "improvement_ci": [
                float(composite_cmp["ci_low"]),
                float(composite_cmp["ci_high"]),
            ],
        },
        "leakage_audit": {
            "neighbor_pool_is_training_only": True,
            "neighbor_selection_uses_pre_intervention_features_only": True,
            "normalization_scales_use_training_paths_only": True,
            "test_outcomes_used_for_model_or_neighbor_selection": False,
        },
        "inputs": {
            key: {"path": str(path), "sha256": _sha256(path)}
            for key, path in paths.items()
        },
    }
    (output / "path_fidelity_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    _write_report(output, summary, comparisons, manifest)
    _write_latex(output, summary, comparisons, manifest)
    return manifest


def _write_report(
    output: Path,
    summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    manifest: dict[str, Any],
) -> None:
    table = summary.pivot(index="metric", columns="model", values="mean_error")
    lines = [
        "# Lemmy Held-Out Event-Path Fidelity",
        "",
        "## Design",
        "",
        f"- Chronological test interventions: {manifest['test_interventions']}.",
        "- BDMTF: 20 generated intervention paths per test event.",
        "- The path-oriented Correction parameter is estimated from training intervention paths only; aggregate-tuned replay is retained as an audit row.",
        "- Conditional baseline: 20 nearest training-only observed intervention paths selected from pre-intervention features.",
        "- Lower error is better. Normalization scales are estimated from training paths only.",
        "",
        "## Results",
        "",
        "| Metric | BDMTF path-fitted | BDMTF aggregate-fitted | Empirical nearest path | BDMTF gain |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for metric in METRICS:
        comparison = comparisons[comparisons["metric"] == metric].iloc[0]
        lines.append(
            f"| {metric} | {table.loc[metric, MODEL_BDMTF]:.3f} | "
            f"{table.loc[metric, MODEL_BDMTF_AGGREGATE]:.3f} | "
            f"{table.loc[metric, MODEL_EMPIRICAL]:.3f} | "
            f"{comparison['bdmtf_improvement_over_empirical']:+.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The benchmark scores generated timestamps, depth distributions, root targeting, parent concentration, and cumulative response trajectories. It evaluates path fidelity rather than only aggregate intervention magnitude.",
            "",
            "The historical baseline is leakage-safe: candidate trajectories and normalization scales come only from the chronological training split, and test outcomes are opened only for final scoring.",
        ]
    )
    (output / "LEMMY_PATH_FIDELITY_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _write_latex(
    output: Path,
    summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    manifest: dict[str, Any],
) -> None:
    table = summary.pivot(index="metric", columns="model", values="mean_error")
    composite = comparisons[
        comparisons["metric"] == "composite_path_error"
    ].iloc[0]
    values = {
        "LemmyPathCompositeBDMTF": table.loc["composite_path_error", MODEL_BDMTF],
        "LemmyPathCompositeAggregate": table.loc[
            "composite_path_error", MODEL_BDMTF_AGGREGATE
        ],
        "LemmyPathCompositeEmpirical": table.loc[
            "composite_path_error", MODEL_EMPIRICAL
        ],
        "LemmyPathCompositeGain": composite["bdmtf_improvement_over_empirical"],
        "LemmyPathCompositeGainLow": composite["ci_low"],
        "LemmyPathCompositeGainHigh": composite["ci_high"],
        "LemmyPathRemovalVisibility": manifest["path_policy_fitted_on_training_only"][
            "remove_visibility_remaining"
        ],
    }
    macros = [
        "% Auto-generated by run_lemmy_path_fidelity.py."
    ] + [f"\\newcommand{{\\{key}}}{{{float(value):.3f}}}" for key, value in values.items()]
    (output / "lemmy_path_fidelity_macros.tex").write_text(
        "\n".join(macros) + "\n", encoding="utf-8"
    )
    labels = {
        "cumulative_count_nmae": "Cumulative trajectory",
        "arrival_time_nwd": "Arrival time",
        "depth_nwd": "Depth distribution",
        "parent_hhi_ae": "Parent concentration",
        "composite_path_error": "Composite",
    }
    rows = [
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Path error & Path-fitted & Aggregate-fitted & Empirical nearest \\\\",
        "\\midrule",
    ]
    for metric, label in labels.items():
        rows.append(
            f"{label} & {table.loc[metric, MODEL_BDMTF]:.3f} & "
            f"{table.loc[metric, MODEL_BDMTF_AGGREGATE]:.3f} & "
            f"{table.loc[metric, MODEL_EMPIRICAL]:.3f} \\\\"
        )
    rows.extend(["\\bottomrule", "\\end{tabular}"])
    (output / "table_lemmy_path_fidelity.tex").write_text(
        "\n".join(rows) + "\n", encoding="utf-8"
    )
