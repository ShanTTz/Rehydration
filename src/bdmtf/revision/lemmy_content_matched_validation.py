from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer

from bdmtf.platform_adapter_validation import select_platform_adapter
from bdmtf.revision.external_data import (
    external_cascade_metrics,
    normalize_events,
)
from bdmtf.revision.external_sources import canonicalize_url
from bdmtf.revision.fitted_models import fit_pooled_model
from bdmtf.revision.provenance import sha256_file, write_json
from bdmtf.revision.simulation import (
    RevisionSimulationConfig,
    event_metrics,
    simulate_cascade,
)


STRUCTURAL_METRICS = (
    "size",
    "max_depth",
    "mean_leaf_depth",
    "mean_branching_factor",
    "root_reply_share",
    "time_to_90_minutes",
    "repeat_author_share",
)


def _stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**32 - 1)


def _load_resolution_records(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records", payload)
    return records if isinstance(records, dict) else {}


def _resolved_canonical(
    value: object,
    records: Mapping[str, Mapping[str, Any]],
) -> str:
    raw = "" if value is None or pd.isna(value) else str(value)
    cached = records.get(raw, {})
    resolved = str(cached.get("resolved_url", "") or raw)
    return canonicalize_url(resolved)


def _greedy_one_to_one(
    candidates: pd.DataFrame,
    left_column: str = "hn_content_id",
    right_column: str = "lemmy_content_id",
) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    ordered = candidates.sort_values(
        [
            "time_distance_hours",
            "title_similarity",
            left_column,
            right_column,
        ],
        ascending=[True, False, True, True],
    )
    left_used: set[str] = set()
    right_used: set[str] = set()
    rows: list[pd.Series] = []
    for _, row in ordered.iterrows():
        left = str(row[left_column])
        right = str(row[right_column])
        if left in left_used or right in right_used:
            continue
        rows.append(row)
        left_used.add(left)
        right_used.add(right)
    return pd.DataFrame(rows).reset_index(drop=True)


def build_exact_url_pairs(
    hn_events: pd.DataFrame,
    lemmy_posts: pd.DataFrame,
    resolution_records: Mapping[str, Mapping[str, Any]],
    max_hours: float,
    complete_lemmy_ids: set[str] | None = None,
) -> pd.DataFrame:
    hn_roots = hn_events[
        hn_events["event_type"].astype(str).str.lower().isin({"post", "root"})
    ].copy()
    hn_roots["hn_content_id"] = hn_roots["content_id"].astype(str)
    hn_roots["hn_created_at"] = pd.to_datetime(
        hn_roots["created_at"],
        utc=True,
        format="mixed",
        errors="coerce",
    )
    hn_roots["canonical_url"] = hn_roots["content_url"].map(
        lambda value: _resolved_canonical(value, resolution_records)
    )
    hn_roots["hn_title"] = hn_roots["text"].fillna("").astype(str)

    lemmy = lemmy_posts.copy()
    lemmy["lemmy_content_id"] = lemmy["content_id"].astype(str)
    lemmy["lemmy_created_at"] = pd.to_datetime(
        lemmy["published"],
        utc=True,
        format="mixed",
        errors="coerce",
    )
    lemmy["canonical_url"] = lemmy["url"].map(
        lambda value: _resolved_canonical(value, resolution_records)
    )
    lemmy["lemmy_title"] = lemmy["title"].fillna("").astype(str)
    if complete_lemmy_ids is not None:
        lemmy = lemmy[
            lemmy["lemmy_content_id"].isin(set(map(str, complete_lemmy_ids)))
        ]

    candidates = hn_roots[
        [
            "hn_content_id",
            "hn_created_at",
            "hn_title",
            "canonical_url",
        ]
    ].merge(
        lemmy[
            [
                "lemmy_content_id",
                "lemmy_created_at",
                "lemmy_title",
                "reported_comments",
                "canonical_url",
            ]
        ],
        on="canonical_url",
        how="inner",
    )
    candidates = candidates[candidates["canonical_url"].ne("")].copy()
    candidates["time_distance_hours"] = (
        candidates["hn_created_at"] - candidates["lemmy_created_at"]
    ).abs().dt.total_seconds() / 3600.0
    candidates = candidates[
        candidates["time_distance_hours"] <= float(max_hours)
    ].copy()
    candidates["match_type"] = "exact_url"
    candidates["title_similarity"] = 1.0
    candidates["created_at"] = candidates[
        ["hn_created_at", "lemmy_created_at"]
    ].min(axis=1)
    selected = _greedy_one_to_one(candidates)
    if selected.empty:
        return selected
    selected["match_id"] = selected.apply(
        lambda row: hashlib.sha256(
            (
                "exact_url|"
                f"{row['hn_content_id']}|{row['lemmy_content_id']}|"
                f"{row['canonical_url']}"
            ).encode("utf-8")
        ).hexdigest()[:20],
        axis=1,
    )
    return selected


def build_semantic_pairs(
    hn_events: pd.DataFrame,
    lemmy_posts: pd.DataFrame,
    complete_lemmy_ids: set[str],
    exact_pairs: pd.DataFrame,
    semantic_threshold: float,
    max_hours: float,
) -> pd.DataFrame:
    hn = hn_events[
        hn_events["event_type"].astype(str).str.lower().isin({"post", "root"})
    ].copy()
    hn["hn_content_id"] = hn["content_id"].astype(str)
    hn["hn_created_at"] = pd.to_datetime(
        hn["created_at"],
        utc=True,
        format="mixed",
        errors="coerce",
    )
    hn["hn_title"] = hn["text"].fillna("").astype(str)
    hn["canonical_url"] = hn["content_url"].fillna("").map(canonicalize_url)

    lemmy = lemmy_posts[
        lemmy_posts["content_id"].astype(str).isin(complete_lemmy_ids)
    ].copy()
    lemmy["lemmy_content_id"] = lemmy["content_id"].astype(str)
    lemmy["lemmy_created_at"] = pd.to_datetime(
        lemmy["published"],
        utc=True,
        format="mixed",
        errors="coerce",
    )
    lemmy["lemmy_title"] = lemmy["title"].fillna("").astype(str)
    lemmy["canonical_url"] = lemmy["url"].fillna("").map(canonicalize_url)

    exact_hn = set(exact_pairs.get("hn_content_id", pd.Series(dtype=str)).astype(str))
    exact_lemmy = set(
        exact_pairs.get("lemmy_content_id", pd.Series(dtype=str)).astype(str)
    )
    hn = hn[
        hn["hn_title"].str.strip().ne("")
        & ~hn["hn_content_id"].isin(exact_hn)
    ].reset_index(drop=True)
    lemmy = lemmy[
        lemmy["lemmy_title"].str.strip().ne("")
        & ~lemmy["lemmy_content_id"].isin(exact_lemmy)
    ].reset_index(drop=True)
    if hn.empty or lemmy.empty:
        return pd.DataFrame()

    titles = pd.concat(
        [hn["hn_title"], lemmy["lemmy_title"]],
        ignore_index=True,
    )
    matrix = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
    ).fit_transform(titles)
    similarities = (matrix[: len(hn)] @ matrix[len(hn) :].T).tocoo()
    candidate_indices = np.flatnonzero(
        similarities.data >= float(semantic_threshold)
    )
    records: list[dict[str, Any]] = []
    for index in candidate_indices:
        left = hn.iloc[int(similarities.row[index])]
        right = lemmy.iloc[int(similarities.col[index])]
        distance = abs(
            (
                left["hn_created_at"] - right["lemmy_created_at"]
            ).total_seconds()
        ) / 3600.0
        if distance > float(max_hours):
            continue
        if (
            left["canonical_url"]
            and left["canonical_url"] == right["canonical_url"]
        ):
            continue
        records.append(
            {
                "hn_content_id": str(left["hn_content_id"]),
                "lemmy_content_id": str(right["lemmy_content_id"]),
                "hn_created_at": left["hn_created_at"],
                "lemmy_created_at": right["lemmy_created_at"],
                "hn_title": str(left["hn_title"]),
                "lemmy_title": str(right["lemmy_title"]),
                "canonical_url": "",
                "reported_comments": float(
                    pd.to_numeric(
                        right.get("reported_comments", np.nan),
                        errors="coerce",
                    )
                ),
                "time_distance_hours": float(distance),
                "match_type": "semantic_event",
                "title_similarity": float(similarities.data[index]),
                "created_at": min(
                    left["hn_created_at"],
                    right["lemmy_created_at"],
                ),
            }
        )
    selected = _greedy_one_to_one(pd.DataFrame(records))
    if selected.empty:
        return selected
    selected["match_id"] = selected.apply(
        lambda row: hashlib.sha256(
            (
                "semantic_event|"
                f"{row['hn_content_id']}|{row['lemmy_content_id']}"
            ).encode("utf-8")
        ).hexdigest()[:20],
        axis=1,
    )
    return selected


def _bootstrap_interval(
    values: np.ndarray,
    samples: int,
    seed: int,
    statistic: str = "mean",
) -> tuple[float, float, float]:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if not len(clean):
        return float("nan"), float("nan"), float("nan")
    function = np.mean if statistic == "mean" else np.median
    rng = np.random.default_rng(seed)
    draws = [
        float(function(rng.choice(clean, size=len(clean), replace=True)))
        for _ in range(samples)
    ]
    return (
        float(function(clean)),
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    )


def _post_level_pairs(
    exact_pairs: pd.DataFrame,
    hn_events: pd.DataFrame,
    bootstrap_samples: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    comment_mask = ~hn_events["event_type"].astype(str).str.lower().isin(
        {"post", "root"}
    )
    hn_sizes = (
        hn_events[comment_mask]
        .groupby(hn_events.loc[comment_mask, "content_id"].astype(str))
        .size()
        .to_dict()
    )
    pairs = exact_pairs.copy()
    pairs["hn_comment_count"] = (
        pairs["hn_content_id"].astype(str).map(hn_sizes).fillna(0).astype(float)
    )
    pairs["lemmy_comment_count"] = pd.to_numeric(
        pairs["reported_comments"],
        errors="coerce",
    )
    pairs = pairs.dropna(subset=["lemmy_comment_count"]).copy()
    pairs["log_comment_gap"] = np.log1p(
        pairs["lemmy_comment_count"].clip(lower=0)
    ) - np.log1p(pairs["hn_comment_count"].clip(lower=0))
    correlation = spearmanr(
        pairs["hn_comment_count"],
        pairs["lemmy_comment_count"],
    )
    gap = _bootstrap_interval(
        pairs["log_comment_gap"].to_numpy(dtype=float),
        bootstrap_samples,
        seed,
    )
    summary = {
        "n_pairs": int(len(pairs)),
        "n_unique_urls": int(pairs["canonical_url"].nunique()),
        "spearman_comment_count": float(correlation.statistic),
        "spearman_p_value": float(correlation.pvalue),
        "mean_log_comment_gap": gap[0],
        "mean_log_comment_gap_ci": [gap[1], gap[2]],
        "median_hn_comments": float(pairs["hn_comment_count"].median()),
        "median_lemmy_comments": float(pairs["lemmy_comment_count"].median()),
    }
    return pairs, summary


def _normalized_metrics(
    hn_events: pd.DataFrame,
    lemmy_events: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    hn = normalize_events(hn_events)
    lemmy = normalize_events(lemmy_events)
    return external_cascade_metrics(hn), external_cascade_metrics(lemmy)


def _paired_real_metrics(
    pairs: pd.DataFrame,
    hn_metrics: pd.DataFrame,
    lemmy_metrics: pd.DataFrame,
    bootstrap_samples: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if pairs.empty:
        return pd.DataFrame(), pd.DataFrame()
    hn = hn_metrics.copy()
    lemmy = lemmy_metrics.copy()
    hn["post_id"] = hn["post_id"].astype(str)
    lemmy["post_id"] = lemmy["post_id"].astype(str)
    pair_metrics = pairs.merge(
        hn[["post_id", *STRUCTURAL_METRICS]].rename(
            columns={
                "post_id": "hn_content_id",
                **{
                    metric: f"hn_{metric}"
                    for metric in STRUCTURAL_METRICS
                },
            }
        ),
        on="hn_content_id",
        how="inner",
        validate="one_to_one",
    ).merge(
        lemmy[["post_id", *STRUCTURAL_METRICS]].rename(
            columns={
                "post_id": "lemmy_content_id",
                **{
                    metric: f"lemmy_{metric}"
                    for metric in STRUCTURAL_METRICS
                },
            }
        ),
        on="lemmy_content_id",
        how="inner",
        validate="one_to_one",
    )
    summary_rows: list[dict[str, Any]] = []
    for match_type, group in pair_metrics.groupby("match_type", sort=False):
        for metric in STRUCTURAL_METRICS:
            difference = (
                group[f"lemmy_{metric}"] - group[f"hn_{metric}"]
            ).to_numpy(dtype=float)
            estimate = _bootstrap_interval(
                difference,
                bootstrap_samples,
                seed + _stable_seed(match_type, metric),
            )
            correlation = spearmanr(
                group[f"hn_{metric}"],
                group[f"lemmy_{metric}"],
            )
            summary_rows.append(
                {
                    "match_type": match_type,
                    "metric": metric,
                    "n_pairs": int(len(group)),
                    "hn_mean": float(group[f"hn_{metric}"].mean()),
                    "lemmy_mean": float(group[f"lemmy_{metric}"].mean()),
                    "paired_mean_difference": estimate[0],
                    "ci_low": estimate[1],
                    "ci_high": estimate[2],
                    "spearman": float(correlation.statistic),
                    "spearman_p_value": float(correlation.pvalue),
                }
            )
    return pair_metrics, pd.DataFrame(summary_rows)


def _make_historical_target_split(
    lemmy_metrics: pd.DataFrame,
    lemmy_posts: pd.DataFrame,
    held_out_ids: set[str],
    cutoff: pd.Timestamp,
    validation_fraction: float,
) -> pd.DataFrame:
    roots = lemmy_posts[["content_id", "published"]].copy()
    roots["post_id"] = roots["content_id"].astype(str)
    roots["created_at"] = pd.to_datetime(
        roots["published"],
        utc=True,
        format="mixed",
        errors="coerce",
    )
    metrics = lemmy_metrics.copy()
    metrics["post_id"] = metrics["post_id"].astype(str)
    eligible = metrics.merge(
        roots[["post_id", "created_at"]],
        on="post_id",
        how="inner",
        validate="one_to_one",
    )
    eligible = eligible[
        (eligible["created_at"] < cutoff)
        & ~eligible["post_id"].isin(held_out_ids)
    ].sort_values(["created_at", "post_id"])
    if len(eligible) < 100:
        raise ValueError(
            "Fewer than 100 pre-test Lemmy cascades are available for adaptation"
        )
    validation_count = max(
        20,
        int(np.floor(len(eligible) * float(validation_fraction))),
    )
    validation_count = min(validation_count, len(eligible) - 20)
    eligible["split"] = "train"
    eligible.iloc[-validation_count:, eligible.columns.get_loc("split")] = (
        "validation"
    )
    return eligible.reset_index(drop=True)


def _run_matched_simulations(
    pairs: pd.DataFrame,
    actual_lemmy_metrics: pd.DataFrame,
    target_history: pd.DataFrame,
    hn_metrics: pd.DataFrame,
    selected_config: RevisionSimulationConfig,
    default_config: RevisionSimulationConfig,
    seeds: Iterable[int],
) -> pd.DataFrame:
    held_out_hn = set(pairs["hn_content_id"].astype(str))
    hn_train = hn_metrics[
        ~hn_metrics["post_id"].astype(str).isin(held_out_hn)
    ].copy()
    hn_train["split"] = "train"
    target_train = target_history.copy()
    hn_profile = fit_pooled_model(hn_train, (), "hackernews_source")
    target_profile = fit_pooled_model(target_train, (), "lemmy_target")
    actual = actual_lemmy_metrics.copy()
    actual["post_id"] = actual["post_id"].astype(str)
    actual_index = actual.set_index("post_id")
    specs = (
        (
            "hackernews_zero_shot_bdmtf",
            "learned_bdmtf",
            hn_profile,
            hn_train,
            default_config,
        ),
        (
            "lemmy_target_fitted_bdmtf",
            "learned_bdmtf",
            target_profile,
            target_train[target_train["split"].eq("train")],
            default_config,
        ),
        (
            "lemmy_structure_selected_bdmtf",
            "learned_bdmtf",
            target_profile,
            target_train[target_train["split"].eq("train")],
            selected_config,
        ),
        (
            "lemmy_empirical_bootstrap",
            "empirical_bootstrap",
            target_profile,
            target_train[target_train["split"].eq("train")],
            selected_config,
        ),
        (
            "lemmy_branching_process",
            "branching_process",
            target_profile,
            target_train[target_train["split"].eq("train")],
            selected_config,
        ),
    )
    rows: list[dict[str, Any]] = []
    for pair in pairs.itertuples(index=False):
        post_id = str(pair.lemmy_content_id)
        if post_id not in actual_index.index:
            continue
        for label, model_name, profile, train_rows, simulation_config in specs:
            for seed in map(int, seeds):
                nodes = simulate_cascade(
                    model_name,
                    profile,
                    train_rows,
                    post_id,
                    _stable_seed(
                        "content-matched",
                        pair.match_id,
                        seed,
                    ),
                    simulation_config,
                )
                rows.append(
                    {
                        "match_id": str(pair.match_id),
                        "match_type": str(pair.match_type),
                        "hn_content_id": str(pair.hn_content_id),
                        "lemmy_content_id": post_id,
                        "model": label,
                        "seed": seed,
                        **event_metrics(nodes),
                    }
                )
    return pd.DataFrame(rows)


def _evaluate_simulations(
    simulations: pd.DataFrame,
    lemmy_metrics: pd.DataFrame,
    target_history: pd.DataFrame,
    bootstrap_samples: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    actual = lemmy_metrics.copy()
    actual["post_id"] = actual["post_id"].astype(str)
    averaged = (
        simulations.groupby(
            [
                "match_id",
                "match_type",
                "hn_content_id",
                "lemmy_content_id",
                "model",
            ],
            sort=False,
        )[list(STRUCTURAL_METRICS)]
        .mean()
        .reset_index()
    )
    scored = averaged.merge(
        actual[["post_id", *STRUCTURAL_METRICS]].rename(
            columns={
                "post_id": "lemmy_content_id",
                **{
                    metric: f"observed_{metric}"
                    for metric in STRUCTURAL_METRICS
                },
            }
        ),
        on="lemmy_content_id",
        how="inner",
        validate="many_to_one",
    )
    scales = {
        metric: max(
            float(target_history.loc[
                target_history["split"].eq("train"), metric
            ].std()),
            0.1,
        )
        for metric in STRUCTURAL_METRICS
    }
    long_rows: list[dict[str, Any]] = []
    for row in scored.itertuples(index=False):
        base = {
            "match_id": row.match_id,
            "match_type": row.match_type,
            "hn_content_id": row.hn_content_id,
            "lemmy_content_id": row.lemmy_content_id,
            "model": row.model,
        }
        for metric in STRUCTURAL_METRICS:
            predicted = float(getattr(row, metric))
            observed = float(getattr(row, f"observed_{metric}"))
            long_rows.append(
                {
                    **base,
                    "metric": metric,
                    "predicted": predicted,
                    "observed": observed,
                    "absolute_error": abs(predicted - observed),
                    "normalized_absolute_error": abs(predicted - observed)
                    / scales[metric],
                }
            )
    long = pd.DataFrame(long_rows)
    summary = (
        long.groupby(["match_type", "model", "metric"], sort=False)
        .agg(
            n_pairs=("match_id", "nunique"),
            mae=("absolute_error", "mean"),
            normalized_mae=("normalized_absolute_error", "mean"),
            median_normalized_absolute_error=(
                "normalized_absolute_error",
                "median",
            ),
        )
        .reset_index()
    )
    ranking_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    for (match_type, model), group in long.groupby(
        ["match_type", "model"], sort=False
    ):
        cluster = group.groupby("match_id")[
            "normalized_absolute_error"
        ].mean()
        values = cluster.to_numpy(dtype=float)
        draws = [
            float(rng.choice(values, size=len(values), replace=True).mean())
            for _ in range(bootstrap_samples)
        ]
        ranking_rows.append(
            {
                "match_type": match_type,
                "model": model,
                "n_pairs": int(len(values)),
                "mean_normalized_mae": float(values.mean()),
                "ci_low": float(np.quantile(draws, 0.025)),
                "ci_high": float(np.quantile(draws, 0.975)),
            }
        )
    ranking = pd.DataFrame(ranking_rows).sort_values(
        ["match_type", "mean_normalized_mae", "model"]
    )
    return long, summary, ranking


def _paired_model_improvement(
    scored: pd.DataFrame,
    match_type: str,
    left_model: str,
    right_model: str,
    bootstrap_samples: int,
    seed: int,
) -> tuple[float, float, float]:
    selected = scored[scored["match_type"].eq(match_type)]
    left = selected[selected["model"].eq(left_model)][
        ["match_id", "metric", "normalized_absolute_error"]
    ].rename(columns={"normalized_absolute_error": "left"})
    right = selected[selected["model"].eq(right_model)][
        ["match_id", "metric", "normalized_absolute_error"]
    ].rename(columns={"normalized_absolute_error": "right"})
    paired = left.merge(
        right,
        on=["match_id", "metric"],
        how="inner",
        validate="one_to_one",
    )
    cluster = (
        paired.assign(improvement=paired["right"] - paired["left"])
        .groupby("match_id")["improvement"]
        .mean()
    )
    values = cluster.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    draws = [
        float(rng.choice(values, size=len(values), replace=True).mean())
        for _ in range(bootstrap_samples)
    ]
    return (
        float(values.mean()),
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    )


def run_lemmy_content_matched_validation(
    root: str | Path,
    config: Mapping[str, Any],
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    project = Path(root)
    output = (
        Path(output_dir)
        if output_dir is not None
        else project
        / str(
            config.get(
                "output_dir",
                "artifacts/external_validation/lemmy_content_matched",
            )
        )
    )
    output.mkdir(parents=True, exist_ok=True)
    input_paths = {
        name: project / str(relative)
        for name, relative in config["inputs"].items()
    }
    hn_events = pd.read_parquet(input_paths["hackernews_events"])
    lemmy_posts = pd.read_parquet(input_paths["lemmy_posts"])
    lemmy_events = pd.read_parquet(input_paths["lemmy_events"])
    records = _load_resolution_records(input_paths["url_resolution_cache"])
    complete_ids = set(
        lemmy_events.loc[
            lemmy_events["event_type"].astype(str).str.lower().isin(
                {"post", "root"}
            ),
            "content_id",
        ].astype(str)
    )
    max_hours = float(config.get("matching", {}).get("max_hours", 72.0))
    exact_post = build_exact_url_pairs(
        hn_events,
        lemmy_posts,
        records,
        max_hours,
    )
    exact_structural = build_exact_url_pairs(
        hn_events,
        lemmy_posts,
        records,
        max_hours,
        complete_ids,
    )
    semantic = build_semantic_pairs(
        hn_events,
        lemmy_posts,
        complete_ids,
        exact_structural,
        float(config.get("matching", {}).get("semantic_threshold", 0.72)),
        max_hours,
    )
    exact_post.to_csv(output / "exact_url_post_pairs.csv", index=False)
    exact_structural.to_csv(
        output / "exact_url_structural_pairs.csv",
        index=False,
    )
    semantic.to_csv(output / "semantic_event_structural_pairs.csv", index=False)

    bootstrap_samples = int(config.get("bootstrap_samples", 2000))
    seed = int(config.get("seed", 30371))
    post_pairs, post_summary = _post_level_pairs(
        exact_post,
        hn_events,
        bootstrap_samples,
        seed,
    )
    post_pairs.to_csv(output / "post_level_paired_counts.csv", index=False)

    hn_metrics, lemmy_metrics = _normalized_metrics(hn_events, lemmy_events)
    hn_metrics.to_parquet(output / "hackernews_cascade_metrics.parquet", index=False)
    lemmy_metrics.to_parquet(output / "lemmy_cascade_metrics.parquet", index=False)
    structural_pairs = pd.concat(
        [exact_structural, semantic],
        ignore_index=True,
    )
    pair_metrics, gap_summary = _paired_real_metrics(
        structural_pairs,
        hn_metrics,
        lemmy_metrics,
        bootstrap_samples,
        seed,
    )
    pair_metrics.to_csv(output / "matched_real_metric_pairs.csv", index=False)
    gap_summary.to_csv(output / "paired_platform_gap_summary.csv", index=False)

    if exact_structural.empty:
        raise ValueError("No complete exact-URL Lemmy/Hacker News pairs were found")
    held_out_ids = set(structural_pairs["lemmy_content_id"].astype(str))
    cutoff = pd.to_datetime(
        exact_structural["lemmy_created_at"],
        utc=True,
        format="mixed",
    ).min()
    target_history = _make_historical_target_split(
        lemmy_metrics,
        lemmy_posts,
        held_out_ids,
        cutoff,
        float(config.get("adapter", {}).get("validation_fraction", 0.2)),
    )
    target_history.to_parquet(
        output / "lemmy_adapter_historical_split.parquet",
        index=False,
    )
    simulation_raw = json.loads(
        input_paths["simulation_config"].read_text(encoding="utf-8")
    )
    default_config = RevisionSimulationConfig.from_dict(
        simulation_raw["simulation"]
    )
    profile = fit_pooled_model(target_history, (), "lemmy_target")
    adapter = config.get("adapter", {})
    selected_config, grid, adapter_manifest = select_platform_adapter(
        target_history,
        profile,
        default_config,
        adapter.get(
            "grid",
            {
                "ranking": ["new", "top", "best"],
                "viewport_k": [10, 20, -1],
                "deep_drill_lambda": [0.0, 0.5, 1.0],
            },
        ),
        adapter.get(
            "selection_metrics",
            [
                "max_depth",
                "mean_leaf_depth",
                "mean_branching_factor",
                "root_reply_share",
                "time_to_90_minutes",
            ],
        ),
        int(adapter.get("max_validation_cascades", 250)),
        seed,
    )
    grid.to_csv(output / "adapter_validation_grid.csv", index=False)
    write_json(output / "adapter_selection_manifest.json", adapter_manifest)
    simulations = _run_matched_simulations(
        structural_pairs,
        lemmy_metrics,
        target_history,
        hn_metrics,
        selected_config,
        default_config,
        config.get("test_seeds", range(10)),
    )
    simulations.to_parquet(output / "matched_simulated_metrics.parquet", index=False)
    scored, model_summary, ranking = _evaluate_simulations(
        simulations,
        lemmy_metrics,
        target_history,
        bootstrap_samples,
        seed,
    )
    scored.to_csv(output / "matched_model_errors.csv", index=False)
    model_summary.to_csv(output / "model_metric_summary.csv", index=False)
    ranking.to_csv(output / "model_ranking.csv", index=False)
    target_fit_improvement = _paired_model_improvement(
        scored,
        "exact_url",
        "lemmy_target_fitted_bdmtf",
        "hackernews_zero_shot_bdmtf",
        bootstrap_samples,
        seed + 1,
    )
    structure_improvement = _paired_model_improvement(
        scored,
        "exact_url",
        "lemmy_structure_selected_bdmtf",
        "hackernews_zero_shot_bdmtf",
        bootstrap_samples,
        seed + 2,
    )
    exact_ranking = ranking[ranking["match_type"].eq("exact_url")]
    target_fitted_row = exact_ranking[
        exact_ranking["model"].eq("lemmy_target_fitted_bdmtf")
    ].iloc[0]
    structure_selected_row = exact_ranking[
        exact_ranking["model"].eq("lemmy_structure_selected_bdmtf")
    ].iloc[0]
    zero_shot_row = exact_ranking[
        exact_ranking["model"].eq("hackernews_zero_shot_bdmtf")
    ].iloc[0]
    result = {
        "status": "complete",
        "claim_allowed": bool(
            len(exact_post) >= int(
                config.get("minimums", {}).get("post_level_exact_pairs", 100)
            )
            and len(exact_structural)
            >= int(
                config.get("minimums", {}).get(
                    "structural_exact_pairs",
                    20,
                )
            )
        ),
        "protocol": config.get("protocol", {}),
        "matching": {
            "max_hours": max_hours,
            "semantic_threshold": float(
                config.get("matching", {}).get("semantic_threshold", 0.72)
            ),
            "exact_post_pairs": int(len(exact_post)),
            "exact_unique_urls": int(exact_post["canonical_url"].nunique()),
            "exact_structural_pairs": int(len(exact_structural)),
            "semantic_structural_pairs": int(len(semantic)),
            "one_to_one_no_post_reuse": True,
            "exact_and_semantic_reported_separately": True,
        },
        "post_level_result": post_summary,
        "adapter_training": {
            "cutoff": cutoff.isoformat(),
            "historical_cascades": int(len(target_history)),
            "train_cascades": int(
                target_history["split"].eq("train").sum()
            ),
            "validation_cascades": int(
                target_history["split"].eq("validation").sum()
            ),
            "held_out_matched_lemmy_cascades": int(len(held_out_ids)),
            "selected_config": asdict(selected_config),
            "test_seeds": int(len(list(config.get("test_seeds", range(10))))),
        },
        "exact_url_model_result": {
            "lemmy_target_fitted_mean_normalized_mae": float(
                target_fitted_row["mean_normalized_mae"]
            ),
            "lemmy_target_fitted_ci": [
                float(target_fitted_row["ci_low"]),
                float(target_fitted_row["ci_high"]),
            ],
            "lemmy_structure_selected_mean_normalized_mae": float(
                structure_selected_row["mean_normalized_mae"]
            ),
            "lemmy_structure_selected_ci": [
                float(structure_selected_row["ci_low"]),
                float(structure_selected_row["ci_high"]),
            ],
            "hackernews_zero_shot_mean_normalized_mae": float(
                zero_shot_row["mean_normalized_mae"]
            ),
            "hackernews_zero_shot_ci": [
                float(zero_shot_row["ci_low"]),
                float(zero_shot_row["ci_high"]),
            ],
            "target_fitted_improvement_over_zero_shot": (
                target_fit_improvement[0]
            ),
            "target_fitted_improvement_ci": [
                target_fit_improvement[1],
                target_fit_improvement[2],
            ],
            "structure_selected_improvement_over_zero_shot": (
                structure_improvement[0]
            ),
            "structure_selected_improvement_ci": [
                structure_improvement[1],
                structure_improvement[2],
            ],
        },
        "leakage_audit": {
            "matched_test_posts_excluded_from_adapter_fit": True,
            "adapter_fit_ends_before_first_exact_matched_test_post": True,
            "validation_only_selects_ranking_viewport_depth": True,
            "test_outcomes_used_for_selection": False,
        },
        "evidence_boundary": (
            "The exact-URL design controls observed story identity and tests "
            "portability between two threaded platforms. It supports bounded "
            "Lemmy adaptation only if the held-out improvement is positive; "
            "it does not establish zero-shot universality or transfer to "
            "non-threaded platforms."
        ),
        "inputs": {
            name: {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
            for name, path in input_paths.items()
        },
    }
    result["framework_support"] = (
        "bounded_lemmy_adaptation_supported"
        if structure_improvement[1] > 0
        else "adaptation_gain_not_statistically_resolved"
        if structure_improvement[0] > 0
        else "adaptation_not_supported_on_exact_pairs"
    )
    write_json(output / "content_matched_manifest.json", result)
    _write_report(
        output / "LEMMY_CONTENT_MATCHED_REPORT.md",
        result,
        gap_summary,
        ranking,
    )
    return result


def _write_report(
    path: Path,
    result: Mapping[str, Any],
    gap_summary: pd.DataFrame,
    ranking: pd.DataFrame,
) -> None:
    matching = result["matching"]
    post = result["post_level_result"]
    model = result["exact_url_model_result"]
    lines = [
        "# Lemmy Content-Matched Cross-Platform Validation",
        "",
        "## Purpose",
        "",
        (
            "This experiment compares Hacker News and Lemmy threads carrying "
            "the same URL, separating content identity from platform mechanism."
        ),
        "",
        "## Matched Samples",
        "",
        f"- Exact-URL post pairs: {matching['exact_post_pairs']}.",
        f"- Unique exact URLs: {matching['exact_unique_urls']}.",
        (
            "- Exact-URL pairs with complete reply trees on both platforms: "
            f"{matching['exact_structural_pairs']}."
        ),
        (
            "- Separate high-threshold semantic-event tree pairs: "
            f"{matching['semantic_structural_pairs']}."
        ),
        "",
        "## Post-Level Result",
        "",
        (
            f"- Paired comment-count Spearman correlation: "
            f"{post['spearman_comment_count']:.3f} "
            f"(p={post['spearman_p_value']:.3g})."
        ),
        (
            f"- Median comments: HN {post['median_hn_comments']:.1f}, "
            f"Lemmy {post['median_lemmy_comments']:.1f}."
        ),
        "",
        "## Matched Adapter Result",
        "",
        (
            "- Lemmy target-fitted BDMTF normalized MAE: "
            f"{model['lemmy_target_fitted_mean_normalized_mae']:.3f} "
            f"[{model['lemmy_target_fitted_ci'][0]:.3f}, "
            f"{model['lemmy_target_fitted_ci'][1]:.3f}]."
        ),
        (
            "- Validation-selected structure adapter normalized MAE: "
            f"{model['lemmy_structure_selected_mean_normalized_mae']:.3f} "
            f"[{model['lemmy_structure_selected_ci'][0]:.3f}, "
            f"{model['lemmy_structure_selected_ci'][1]:.3f}]."
        ),
        (
            "- HN zero-shot BDMTF normalized MAE: "
            f"{model['hackernews_zero_shot_mean_normalized_mae']:.3f} "
            f"[{model['hackernews_zero_shot_ci'][0]:.3f}, "
            f"{model['hackernews_zero_shot_ci'][1]:.3f}]."
        ),
        (
            "- Target-fit improvement over zero-shot: "
            f"{model['target_fitted_improvement_over_zero_shot']:+.3f} "
            f"[{model['target_fitted_improvement_ci'][0]:+.3f}, "
            f"{model['target_fitted_improvement_ci'][1]:+.3f}]."
        ),
        (
            "- Structure-selected improvement over zero-shot: "
            f"{model['structure_selected_improvement_over_zero_shot']:+.3f} "
            f"[{model['structure_selected_improvement_ci'][0]:+.3f}, "
            f"{model['structure_selected_improvement_ci'][1]:+.3f}]."
        ),
        "",
        "## Platform Gap Summary",
        "",
        "| Match | Metric | N | HN mean | Lemmy mean | Paired difference | 95% CI |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in gap_summary.itertuples(index=False):
        lines.append(
            f"| {row.match_type} | {row.metric} | {row.n_pairs} | "
            f"{row.hn_mean:.3f} | {row.lemmy_mean:.3f} | "
            f"{row.paired_mean_difference:+.3f} | "
            f"[{row.ci_low:+.3f}, {row.ci_high:+.3f}] |"
        )
    lines.extend(
        [
            "",
            "## Model Ranking",
            "",
            "| Match | Model | N | Normalized MAE | 95% CI |",
            "| --- | --- | ---: | ---: | --- |",
        ]
    )
    for row in ranking.itertuples(index=False):
        lines.append(
            f"| {row.match_type} | {row.model} | {row.n_pairs} | "
            f"{row.mean_normalized_mae:.3f} | "
            f"[{row.ci_low:.3f}, {row.ci_high:.3f}] |"
        )
    lines.extend(
        [
            "",
            "## Evidence Boundary",
            "",
            str(result["evidence_boundary"]),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
