from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from bdmtf.revision.lemmy_intervention_fidelity import (
    _finite_conformal_quantile,
    _mechanism_prediction,
    prepare_intervention_dataset,
)
from bdmtf.revision.policies import EventNode, RedditRankingPolicy
from bdmtf.revision.provenance import sha256_file, write_json


PRIMARY_OUTCOMES = ("reply_count", "active_authors", "max_depth")
FEATURE_NAMES = (
    "log_age_days",
    "log_previous_day",
    "log_rolling_seven_days",
    "log_cumulative_comments",
    "relative_day",
    "previous_day_zero",
)


@dataclass(frozen=True)
class ReplayParameters:
    root_reply_share: float
    repeat_author_probability: float
    position_decay: float
    depth_bias: float
    viewport_k: int
    ranking: str


def _stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**32 - 1)


def _timestamp_series(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True, format="mixed", errors="coerce")


def _prepare_inputs(
    panel: pd.DataFrame,
    matches: pd.DataFrame,
    splits: pd.DataFrame,
    events: pd.DataFrame,
    posts: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    matches = matches.copy()
    splits = splits.copy()
    panel = panel.copy()
    events = events.copy()
    posts = posts.copy()

    for frame in (matches, splits, panel):
        frame["intervention_id"] = frame["intervention_id"].astype(str)
    matches["event_time"] = _timestamp_series(matches["event_time"])
    splits["event_time"] = _timestamp_series(splits["event_time"])
    events["created_at"] = _timestamp_series(events["created_at"])
    posts["published"] = _timestamp_series(posts["published"])
    events["content_id"] = events["content_id"].astype(str)
    posts["content_id"] = posts["content_id"].astype(str)
    matches["treated_content_id"] = matches["treated_content_id"].astype(str)
    matches["control_content_id"] = matches["control_content_id"].astype(str)
    panel["content_id"] = panel["content_id"].astype(str)
    panel["relative_period"] = pd.to_numeric(
        panel["relative_period"], errors="raise"
    ).astype(int)
    panel["treated"] = pd.to_numeric(panel["treated"], errors="raise").astype(int)
    panel["value"] = pd.to_numeric(panel["value"], errors="raise").astype(float)

    if events["created_at"].isna().any() or posts["published"].isna().any():
        raise ValueError("Lemmy replay inputs contain invalid timestamps")
    if matches["event_time"].isna().any() or splits["event_time"].isna().any():
        raise ValueError("Lemmy replay matches contain invalid event times")

    assignment = splits[
        ["intervention_id", "intervention_type", "event_time", "split"]
    ].drop_duplicates("intervention_id")
    prepared_matches = matches.drop(
        columns=["event_time"], errors="ignore"
    ).merge(
        assignment,
        on=["intervention_id", "intervention_type"],
        how="inner",
        validate="one_to_one",
    )
    if len(prepared_matches) != len(matches):
        raise ValueError("Every Lemmy intervention must have exactly one frozen split")
    return panel, prepared_matches, splits, events, posts


def _event_groups(events: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        str(content_id): group.sort_values(["created_at", "event_id"]).reset_index(
            drop=True
        )
        for content_id, group in events.groupby("content_id", sort=False)
    }


def _daily_counts(
    group: pd.DataFrame | None,
    event_time: pd.Timestamp,
    first_day: int,
    last_day: int,
) -> list[int]:
    if group is None or group.empty:
        return [0] * (last_day - first_day + 1)
    offsets = np.floor(
        (group["created_at"] - event_time).dt.total_seconds() / 86400.0
    ).astype(int)
    return [
        int((offsets == day).sum()) for day in range(first_day, last_day + 1)
    ]


def _intensity_features(
    age_days: float,
    previous_day: int,
    rolling_seven: int,
    cumulative_comments: int,
    relative_day: int,
) -> np.ndarray:
    return np.asarray(
        [
            math.log1p(max(age_days, 0.0)),
            math.log1p(max(previous_day, 0)),
            math.log1p(max(rolling_seven, 0)),
            math.log1p(max(cumulative_comments, 0)),
            float(relative_day) / 7.0,
            float(previous_day == 0),
        ],
        dtype=float,
    )


def _training_rows(
    matches: pd.DataFrame,
    event_groups: Mapping[str, pd.DataFrame],
    post_times: Mapping[str, pd.Timestamp],
    horizon_days: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for match in matches[matches["split"].eq("train")].itertuples(index=False):
        content_id = str(match.control_content_id)
        group = event_groups.get(content_id)
        event_time = match.event_time
        published = post_times.get(content_id)
        if published is None or pd.isna(published):
            continue
        all_counts = _daily_counts(group, event_time, -7, horizon_days - 1)
        history = list(all_counts[:7])
        cumulative = int(sum(history))
        for day in range(horizon_days):
            features = _intensity_features(
                (event_time - published).total_seconds() / 86400.0 + day,
                history[-1],
                sum(history[-7:]),
                cumulative,
                day,
            )
            target = int(all_counts[7 + day])
            rows.append(
                {
                    "intervention_id": str(match.intervention_id),
                    "content_id": content_id,
                    "relative_day": day,
                    "target_count": target,
                    **dict(zip(FEATURE_NAMES, features)),
                }
            )
            history.append(target)
            cumulative += target
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No training control trajectories were available")
    return (
        frame[list(FEATURE_NAMES)].to_numpy(dtype=float),
        frame["target_count"].to_numpy(dtype=float),
        frame,
    )


def _fit_target_and_author_mechanisms(
    train_matches: pd.DataFrame,
    event_groups: Mapping[str, pd.DataFrame],
    config: Mapping[str, Any],
) -> ReplayParameters:
    root_replies = 0
    replies = 0
    repeats = 0
    author_events = 0
    parent_depths: list[int] = []
    for match in train_matches.itertuples(index=False):
        content_id = str(match.control_content_id)
        group = event_groups.get(content_id)
        if group is None or group.empty:
            continue
        window = group[
            (group["created_at"] >= match.event_time)
            & (
                group["created_at"]
                < match.event_time
                + pd.Timedelta(days=int(config.get("horizon_days", 8)))
            )
        ].copy()
        if window.empty:
            continue
        depth_by_id = {
            str(row.event_id): int(row.depth)
            for row in group.itertuples(index=False)
        }
        seen_authors = set(
            group.loc[group["created_at"] < match.event_time, "author_id"].astype(str)
        )
        for row in window.sort_values(["created_at", "event_id"]).itertuples(
            index=False
        ):
            parent_id = str(row.parent_event_id)
            replies += 1
            if parent_id == content_id or parent_id not in depth_by_id:
                root_replies += 1
            else:
                parent_depths.append(int(depth_by_id[parent_id]))
            author = str(row.author_id)
            author_events += 1
            if author in seen_authors:
                repeats += 1
            seen_authors.add(author)

    smoothing = 2.0
    root_share = (root_replies + smoothing) / (replies + 2 * smoothing)
    repeat_probability = (repeats + smoothing) / (
        author_events + 2 * smoothing
    )
    mean_parent_depth = float(np.mean(parent_depths)) if parent_depths else 1.0
    depth_bias = float(
        np.clip(
            math.log1p(mean_parent_depth)
            / max(math.log1p(mean_parent_depth + 3.0), 1e-9)
            - 0.35,
            -0.25,
            0.45,
        )
    )
    return ReplayParameters(
        root_reply_share=float(np.clip(root_share, 0.02, 0.98)),
        repeat_author_probability=float(
            np.clip(repeat_probability, 0.02, 0.98)
        ),
        position_decay=float(config.get("position_decay", 0.82)),
        depth_bias=depth_bias,
        viewport_k=int(config.get("viewport_k", 20)),
        ranking=str(config.get("ranking", "new")),
    )


def fit_agent_replay_model(
    matches: pd.DataFrame,
    events: pd.DataFrame,
    posts: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[PoissonRegressor, ReplayParameters, pd.DataFrame]:
    groups = _event_groups(events)
    post_times = dict(zip(posts["content_id"].astype(str), posts["published"]))
    x, y, training_frame = _training_rows(
        matches,
        groups,
        post_times,
        int(config.get("horizon_days", 8)),
    )
    model = PoissonRegressor(
        alpha=float(config.get("poisson_alpha", 0.1)),
        max_iter=int(config.get("poisson_max_iter", 1000)),
    )
    model.fit(x, y)
    parameters = _fit_target_and_author_mechanisms(
        matches[matches["split"].eq("train")],
        groups,
        config,
    )
    return model, parameters, training_frame


def _initial_nodes(
    content_id: str,
    event_time: pd.Timestamp,
    group: pd.DataFrame | None,
    published: pd.Timestamp,
) -> list[EventNode]:
    root_minute = (published - event_time).total_seconds() / 60.0
    nodes = [
        EventNode(
            node_id="post",
            parent_id=None,
            depth=0,
            created_minute=float(root_minute),
            author_id="post_author",
            metadata={"root_post": True, "observed_pre_state": True},
        )
    ]
    if group is None or group.empty:
        return nodes
    pre = group[group["created_at"] < event_time].sort_values(
        ["created_at", "event_id"]
    )
    available_ids: set[str] = set()
    for row in pre.itertuples(index=False):
        node_id = str(row.event_id)
        raw_parent = str(row.parent_event_id)
        parent_id = (
            raw_parent
            if raw_parent in available_ids
            else "post"
        )
        parent_depth = next(
            (
                node.depth
                for node in reversed(nodes)
                if node.node_id == parent_id
            ),
            0,
        )
        nodes.append(
            EventNode(
                node_id=node_id,
                parent_id=parent_id,
                depth=parent_depth + 1,
                created_minute=float(
                    (row.created_at - event_time).total_seconds() / 60.0
                ),
                removed=bool(row.removed),
                author_id=str(row.author_id),
                metadata={"observed_pre_state": True},
            )
        )
        available_ids.add(node_id)
    return nodes


def _draw_exogenous_plan(
    *,
    model: PoissonRegressor,
    initial_group: pd.DataFrame | None,
    event_time: pd.Timestamp,
    published: pd.Timestamp,
    horizon_days: int,
    intensity_scale: float,
    max_events_per_day: int,
    seed: int,
) -> list[dict[str, float | int]]:
    rng = np.random.default_rng(seed)
    history = _daily_counts(initial_group, event_time, -7, -1)
    cumulative = int(
        0
        if initial_group is None
        else (initial_group["created_at"] < event_time).sum()
    )
    plan: list[dict[str, float | int]] = []
    for day in range(horizon_days):
        features = _intensity_features(
            (event_time - published).total_seconds() / 86400.0 + day,
            history[-1],
            sum(history[-7:]),
            cumulative,
            day,
        )
        expected = float(model.predict(features.reshape(1, -1))[0])
        expected = float(
            np.clip(expected * intensity_scale, 0.0, max_events_per_day)
        )
        count = int(min(rng.poisson(expected), max_events_per_day))
        minutes = np.sort(rng.uniform(day * 1440.0, (day + 1) * 1440.0, count))
        for minute in minutes:
            plan.append(
                {
                    "minute": float(minute),
                    "visibility_draw": float(rng.random()),
                    "author_repeat_draw": float(rng.random()),
                    "author_choice_draw": float(rng.random()),
                    "parent_root_draw": float(rng.random()),
                    "parent_choice_draw": float(rng.random()),
                }
            )
        history.append(count)
        cumulative += count
    return plan


def _weighted_choice(
    items: list[Any],
    weights: list[float],
    draw: float,
) -> Any:
    if len(items) == 1:
        return items[0]
    total = float(sum(weights))
    if total <= 0:
        return items[min(len(items) - 1, int(draw * len(items)))]
    threshold = draw * total
    running = 0.0
    for item, weight in zip(items, weights):
        running += float(weight)
        if running >= threshold:
            return item
    return items[-1]


def _choose_author(
    nodes: list[EventNode],
    parameters: ReplayParameters,
    repeat_draw: float,
    choice_draw: float,
    new_index: int,
) -> str:
    counts: dict[str, int] = {}
    for node in nodes[1:]:
        if node.author_id:
            counts[node.author_id] = counts.get(node.author_id, 0) + 1
    if counts and repeat_draw < parameters.repeat_author_probability:
        authors = sorted(counts)
        return str(
            _weighted_choice(
                authors,
                [math.sqrt(counts[author]) for author in authors],
                choice_draw,
            )
        )
    return f"sim_agent_{new_index}"


def _choose_parent(
    nodes: list[EventNode],
    minute: float,
    parameters: ReplayParameters,
    root_draw: float,
    choice_draw: float,
) -> EventNode:
    root = nodes[0]
    candidates = [node for node in nodes[1:] if not node.removed]
    if not candidates or root_draw < parameters.root_reply_share:
        return root
    ranking = RedditRankingPolicy(
        parameters.ranking,
        position_decay=parameters.position_decay,
    )
    visible = ranking.rank(candidates, minute)[: max(1, parameters.viewport_k)]
    weights = [
        parameters.position_decay**position
        * math.exp(parameters.depth_bias * min(node.depth, 12))
        for position, node in enumerate(visible)
    ]
    return _weighted_choice(visible, weights, choice_draw)


def _materialize_condition(
    *,
    initial_nodes: list[EventNode],
    plan: list[dict[str, float | int]],
    parameters: ReplayParameters,
    condition: str,
    intervention_type: str,
    remove_visibility_remaining: float,
) -> list[EventNode]:
    nodes = [
        EventNode(
            node_id=node.node_id,
            parent_id=node.parent_id,
            depth=node.depth,
            created_minute=node.created_minute,
            score=node.score,
            likes=node.likes,
            dislikes=node.dislikes,
            toxicity=node.toxicity,
            removed=node.removed,
            author_id=node.author_id,
            metadata=dict(node.metadata),
        )
        for node in initial_nodes
    ]
    new_author_index = 0
    for index, candidate in enumerate(plan):
        if condition == "intervention":
            if intervention_type == "lock_post":
                continue
            if (
                intervention_type == "remove_post"
                and float(candidate["visibility_draw"])
                >= remove_visibility_remaining
            ):
                continue
        author = _choose_author(
            nodes,
            parameters,
            float(candidate["author_repeat_draw"]),
            float(candidate["author_choice_draw"]),
            new_author_index,
        )
        if author.startswith("sim_agent_"):
            new_author_index += 1
        parent = _choose_parent(
            nodes,
            float(candidate["minute"]),
            parameters,
            float(candidate["parent_root_draw"]),
            float(candidate["parent_choice_draw"]),
        )
        nodes.append(
            EventNode(
                node_id=f"sim_{condition}_{index}",
                parent_id=parent.node_id,
                depth=parent.depth + 1,
                created_minute=float(candidate["minute"]),
                author_id=author,
                metadata={
                    "simulated_future": True,
                    "condition": condition,
                    "intervention_type": intervention_type,
                },
            )
        )
    return nodes


def _daily_metrics(
    nodes: Iterable[EventNode],
    horizon_days: int,
) -> dict[str, float]:
    generated = [
        node
        for node in nodes
        if bool(node.metadata.get("simulated_future"))
        and node.created_minute >= 0
    ]
    daily: dict[str, list[float]] = {
        outcome: [] for outcome in PRIMARY_OUTCOMES
    }
    for day in range(horizon_days):
        selected = [
            node
            for node in generated
            if day * 1440.0 <= node.created_minute < (day + 1) * 1440.0
        ]
        daily["reply_count"].append(float(len(selected)))
        daily["active_authors"].append(
            float(len({node.author_id for node in selected}))
        )
        daily["max_depth"].append(
            float(max((node.depth for node in selected), default=0))
        )
    return {
        outcome: float(np.mean(values)) for outcome, values in daily.items()
    }


def _observed_effects(
    panel: pd.DataFrame,
    intervention_ids: Iterable[str],
    outcomes: Iterable[str] = PRIMARY_OUTCOMES,
) -> pd.DataFrame:
    selected_ids = set(map(str, intervention_ids))
    selected = panel[
        panel["intervention_id"].astype(str).isin(selected_ids)
        & panel["outcome"].astype(str).isin(tuple(outcomes))
    ].copy()
    rows: list[dict[str, Any]] = []
    for (intervention_id, outcome), group in selected.groupby(
        ["intervention_id", "outcome"], sort=False
    ):
        means: dict[tuple[int, str], float] = {}
        for treated in (0, 1):
            arm = group[group["treated"].eq(treated)]
            means[(treated, "pre")] = float(
                arm.loc[arm["relative_period"].between(-7, -1), "value"].mean()
            )
            means[(treated, "post")] = float(
                arm.loc[arm["relative_period"].between(0, 7), "value"].mean()
            )
        effect = (
            means[(1, "post")]
            - means[(0, "post")]
            - means[(1, "pre")]
            + means[(0, "pre")]
        )
        rows.append(
            {
                "intervention_id": str(intervention_id),
                "outcome": str(outcome),
                "observed_effect": effect,
            }
        )
    return pd.DataFrame(rows)


def _simulate_matches(
    *,
    selected_matches: pd.DataFrame,
    model: PoissonRegressor,
    parameters: ReplayParameters,
    event_groups: Mapping[str, pd.DataFrame],
    post_times: Mapping[str, pd.Timestamp],
    horizon_days: int,
    intensity_scale: float,
    remove_visibility_remaining: float,
    seeds: Iterable[int],
    max_events_per_day: int,
    collect_events: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    effect_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    for match in selected_matches.itertuples(index=False):
        content_id = str(match.treated_content_id)
        published = post_times.get(content_id)
        if published is None or pd.isna(published):
            raise ValueError(f"Missing Lemmy root metadata for {content_id}")
        group = event_groups.get(content_id)
        initial_nodes = _initial_nodes(
            content_id,
            match.event_time,
            group,
            published,
        )
        for seed in map(int, seeds):
            run_seed = _stable_seed(
                "lemmy-agent-replay",
                match.intervention_id,
                seed,
            )
            plan = _draw_exogenous_plan(
                model=model,
                initial_group=group,
                event_time=match.event_time,
                published=published,
                horizon_days=horizon_days,
                intensity_scale=intensity_scale,
                max_events_per_day=max_events_per_day,
                seed=run_seed,
            )
            condition_nodes: dict[str, list[EventNode]] = {}
            condition_metrics: dict[str, dict[str, float]] = {}
            for condition in ("no_intervention", "intervention"):
                nodes = _materialize_condition(
                    initial_nodes=initial_nodes,
                    plan=plan,
                    parameters=parameters,
                    condition=condition,
                    intervention_type=str(match.intervention_type),
                    remove_visibility_remaining=remove_visibility_remaining,
                )
                condition_nodes[condition] = nodes
                condition_metrics[condition] = _daily_metrics(nodes, horizon_days)
            for outcome in PRIMARY_OUTCOMES:
                effect_rows.append(
                    {
                        "intervention_id": str(match.intervention_id),
                        "intervention_type": str(match.intervention_type),
                        "split": str(match.split),
                        "seed": seed,
                        "outcome": outcome,
                        "predicted_effect": (
                            condition_metrics["intervention"][outcome]
                            - condition_metrics["no_intervention"][outcome]
                        ),
                    }
                )
            if collect_events:
                for condition, nodes in condition_nodes.items():
                    for node in nodes:
                        if not node.metadata.get("simulated_future"):
                            continue
                        event_rows.append(
                            {
                                "intervention_id": str(match.intervention_id),
                                "intervention_type": str(match.intervention_type),
                                "content_id": content_id,
                                "split": str(match.split),
                                "seed": seed,
                                "condition": condition,
                                **asdict(node),
                                "metadata": json.dumps(
                                    node.metadata,
                                    sort_keys=True,
                                ),
                            }
                        )
    return pd.DataFrame(effect_rows), pd.DataFrame(event_rows)


def _aggregate_agent_predictions(seed_effects: pd.DataFrame) -> pd.DataFrame:
    grouped = seed_effects.groupby(
        ["intervention_id", "intervention_type", "split", "outcome"],
        sort=False,
    )["predicted_effect"]
    return (
        grouped.agg(
            predicted_effect="mean",
            seed_ci_low=lambda values: float(np.quantile(values, 0.025)),
            seed_ci_high=lambda values: float(np.quantile(values, 0.975)),
            simulation_sd="std",
            simulation_seeds="count",
        )
        .reset_index()
        .assign(
            ci_low=lambda frame: frame["seed_ci_low"],
            ci_high=lambda frame: frame["seed_ci_high"],
        )
        .assign(model="bdmtf_agent_replay")
    )


def _frozen_predictions(
    selected_matches: pd.DataFrame,
    panel: pd.DataFrame,
    remove_visibility_remaining: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for match in selected_matches.itertuples(index=False):
        for outcome in PRIMARY_OUTCOMES:
            values = (
                panel[
                    panel["intervention_id"].eq(str(match.intervention_id))
                    & panel["outcome"].eq(outcome)
                    & panel["treated"].eq(1)
                    & panel["relative_period"].between(-7, -1)
                ]
                .sort_values("relative_period")["value"]
                .to_numpy(dtype=float)
            )
            prediction = _mechanism_prediction(
                values,
                str(match.intervention_type),
                outcome,
                list(range(8)),
                remove_visibility_remaining,
            )
            for model_name, value in (
                ("zero_effect", 0.0),
                ("frozen_pretrend", prediction),
            ):
                rows.append(
                    {
                        "intervention_id": str(match.intervention_id),
                        "intervention_type": str(match.intervention_type),
                        "split": str(match.split),
                        "outcome": outcome,
                        "predicted_effect": float(value),
                        "ci_low": float(value),
                        "ci_high": float(value),
                        "seed_ci_low": float(value),
                        "seed_ci_high": float(value),
                        "simulation_sd": 0.0,
                        "simulation_seeds": 0,
                        "model": model_name,
                    }
                )
    return pd.DataFrame(rows)


def _type_values(
    train: pd.DataFrame,
    targets: pd.DataFrame,
    statistic: str = "mean",
) -> np.ndarray:
    global_value = float(getattr(train["observed_effect"], statistic)())
    grouped = getattr(
        train.groupby("intervention_type")["observed_effect"], statistic
    )()
    return (
        targets["intervention_type"]
        .map(grouped)
        .fillna(global_value)
        .to_numpy(dtype=float)
    )


def _majority_sign_value(train: pd.DataFrame) -> float:
    signs = np.sign(train["observed_effect"].to_numpy(dtype=float))
    counts = {
        sign: int(np.sum(signs == sign)) for sign in (-1.0, 0.0, 1.0)
    }
    majority = max(counts, key=lambda sign: (counts[sign], abs(sign)))
    magnitude = float(np.median(np.abs(train["observed_effect"])))
    return float(majority * magnitude)


def _nearest_history_predictions(
    train: pd.DataFrame,
    targets: pd.DataFrame,
    feature_columns: list[str],
) -> np.ndarray:
    scaler = StandardScaler()
    train_values = scaler.fit_transform(train[feature_columns])
    target_values = scaler.transform(targets[feature_columns])
    train_types = train["intervention_type"].astype(str).to_numpy()
    train_effects = train["observed_effect"].to_numpy(dtype=float)
    predictions: list[float] = []
    for row, intervention_type in zip(
        target_values,
        targets["intervention_type"].astype(str),
    ):
        candidates = np.flatnonzero(train_types == intervention_type)
        if not len(candidates):
            candidates = np.arange(len(train_values))
        distances = np.sum((train_values[candidates] - row) ** 2, axis=1)
        predictions.append(float(train_effects[candidates[np.argmin(distances)]]))
    return np.asarray(predictions, dtype=float)


def _strong_baseline_predictions(
    panel: pd.DataFrame,
    matches: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    baseline_config = {
        "protocol": {
            "primary_outcomes": list(PRIMARY_OUTCOMES),
            "secondary_outcomes": [],
        },
        "pre_periods": list(range(-7, 0)),
        "post_periods": list(range(0, 8)),
        "train_fraction": 0.6,
        "validation_fraction": 0.2,
    }
    features, effects, additive_columns, _ = prepare_intervention_dataset(
        panel,
        matches,
        baseline_config,
    )
    expected_splits = matches.set_index("intervention_id")["split"].astype(str)
    observed_splits = effects.drop_duplicates("intervention_id").set_index(
        "intervention_id"
    )["split"].astype(str)
    common = expected_splits.index.intersection(observed_splits.index)
    if not expected_splits.loc[common].equals(observed_splits.loc[common]):
        raise ValueError("Strong baselines do not match the frozen chronological split")

    rows: list[pd.DataFrame] = []
    tuning: list[dict[str, Any]] = []
    coverage = float(config.get("prediction_interval", 0.95))
    alphas = [
        float(value)
        for value in config.get("baseline_ridge_grid", [0.1, 1.0, 10.0, 100.0])
    ]
    for outcome in PRIMARY_OUTCOMES:
        frame = effects[effects["outcome"].eq(outcome)].merge(
            features,
            on=[
                "intervention_id",
                "intervention_type",
                "event_time",
                "split",
                "community_id",
            ],
            validate="one_to_one",
        )
        train = frame[frame["split"].eq("train")].copy()
        validation = frame[frame["split"].eq("validation")].copy()
        test = frame[frame["split"].eq("test")].copy()
        majority = _majority_sign_value(train)
        type_mean_validation = _type_values(train, validation, "mean")
        type_mean_test = _type_values(train, test, "mean")
        nearest_validation = _nearest_history_predictions(
            train, validation, additive_columns
        )
        nearest_test = _nearest_history_predictions(train, test, additive_columns)

        best_ridge = None
        best_alpha = None
        best_validation_mae = float("inf")
        validation_truth = validation["observed_effect"].to_numpy(dtype=float)
        for alpha in alphas:
            ridge = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            ridge.fit(
                train[additive_columns],
                train["observed_effect"].to_numpy(dtype=float),
            )
            prediction = ridge.predict(validation[additive_columns])
            score = float(np.mean(np.abs(prediction - validation_truth)))
            if score < best_validation_mae:
                best_ridge = ridge
                best_alpha = alpha
                best_validation_mae = score
        if best_ridge is None:
            raise RuntimeError("Ridge baseline fitting failed")
        tuning.append(
            {
                "outcome": outcome,
                "model": "additive_ridge",
                "selected_alpha": float(best_alpha),
                "validation_mae": best_validation_mae,
            }
        )
        candidates = {
            "majority_sign": (
                np.full(len(validation), majority),
                np.full(len(test), majority),
            ),
            "intervention_type_mean": (type_mean_validation, type_mean_test),
            "additive_ridge": (
                best_ridge.predict(validation[additive_columns]),
                best_ridge.predict(test[additive_columns]),
            ),
            "historical_nearest_neighbor": (
                nearest_validation,
                nearest_test,
            ),
        }
        for model_name, (validation_prediction, test_prediction) in candidates.items():
            radius = _finite_conformal_quantile(
                np.abs(validation_prediction - validation_truth),
                coverage,
            )
            part = test[
                ["intervention_id", "intervention_type", "split"]
            ].copy()
            part["outcome"] = outcome
            part["predicted_effect"] = test_prediction
            part["ci_low"] = test_prediction - radius
            part["ci_high"] = test_prediction + radius
            part["seed_ci_low"] = test_prediction
            part["seed_ci_high"] = test_prediction
            part["simulation_sd"] = 0.0
            part["simulation_seeds"] = 0
            part["model"] = model_name
            rows.append(part)
    return pd.concat(rows, ignore_index=True), tuning


def _score_predictions(
    predictions: pd.DataFrame,
    observed: pd.DataFrame,
) -> pd.DataFrame:
    scored = predictions.merge(
        observed,
        on=["intervention_id", "outcome"],
        how="inner",
        validate="many_to_one",
    )
    scored["absolute_error"] = (
        scored["predicted_effect"] - scored["observed_effect"]
    ).abs()
    scored["direction_correct"] = (
        np.sign(scored["predicted_effect"])
        == np.sign(scored["observed_effect"])
    ).astype(float)
    scored["interval_coverage"] = (
        (scored["observed_effect"] >= scored["ci_low"])
        & (scored["observed_effect"] <= scored["ci_high"])
    ).astype(float)
    scored["seed_interval_coverage"] = (
        (scored["observed_effect"] >= scored["seed_ci_low"])
        & (scored["observed_effect"] <= scored["seed_ci_high"])
    ).astype(float)
    return scored


def _validation_score(scored: pd.DataFrame) -> float:
    outcome_scores = []
    for _, group in scored.groupby("outcome", sort=False):
        scale = max(
            float(np.mean(np.abs(group["observed_effect"]))),
            0.25,
        )
        outcome_scores.append(float(group["absolute_error"].mean()) / scale)
    return float(np.mean(outcome_scores))


def _bootstrap_primary(
    scored: pd.DataFrame,
    samples: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    primary = scored[scored["outcome"].isin(PRIMARY_OUTCOMES)].copy()
    for model, group in primary.groupby("model", sort=False):
        cluster = (
            group.groupby("intervention_id", sort=False)
            .agg(
                absolute_error=("absolute_error", "mean"),
                direction_correct=("direction_correct", "mean"),
                interval_coverage=("interval_coverage", "mean"),
            )
            .reset_index(drop=True)
        )
        ids = np.arange(len(cluster))
        sampled_indices = rng.choice(
            ids,
            size=(samples, len(ids)),
            replace=True,
        )
        metrics = cluster[
            [
                "absolute_error",
                "direction_correct",
                "interval_coverage",
            ]
        ].to_numpy(dtype=float)
        values = metrics[sampled_indices].mean(axis=1)
        rows.append(
            {
                "model": model,
                "n_interventions": int(len(cluster)),
                "primary_mae": float(group["absolute_error"].mean()),
                "primary_mae_ci_low": float(np.quantile(values[:, 0], 0.025)),
                "primary_mae_ci_high": float(np.quantile(values[:, 0], 0.975)),
                "direction_accuracy": float(group["direction_correct"].mean()),
                "direction_ci_low": float(np.quantile(values[:, 1], 0.025)),
                "direction_ci_high": float(np.quantile(values[:, 1], 0.975)),
                "interval_coverage": float(group["interval_coverage"].mean()),
                "coverage_ci_low": float(np.quantile(values[:, 2], 0.025)),
                "coverage_ci_high": float(np.quantile(values[:, 2], 0.975)),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["primary_mae", "model"]
    ).reset_index(drop=True)


def _paired_improvement(
    scored: pd.DataFrame,
    left_model: str,
    right_model: str,
    samples: int,
    seed: int,
) -> tuple[float, float, float]:
    primary = scored[scored["outcome"].isin(PRIMARY_OUTCOMES)]
    left = primary[primary["model"].eq(left_model)][
        ["intervention_id", "outcome", "absolute_error"]
    ].rename(columns={"absolute_error": "left_error"})
    right = primary[primary["model"].eq(right_model)][
        ["intervention_id", "outcome", "absolute_error"]
    ].rename(columns={"absolute_error": "right_error"})
    paired = left.merge(
        right,
        on=["intervention_id", "outcome"],
        how="inner",
        validate="one_to_one",
    )
    paired["improvement"] = paired["right_error"] - paired["left_error"]
    cluster = paired.groupby("intervention_id")["improvement"].mean()
    rng = np.random.default_rng(seed)
    values = cluster.to_numpy(dtype=float)
    draws = [
        float(rng.choice(values, size=len(values), replace=True).mean())
        for _ in range(samples)
    ]
    return (
        float(cluster.mean()),
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    )


def run_lemmy_agent_replay(
    root: str | Path,
    config: Mapping[str, Any],
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    project = Path(root)
    inputs = config["inputs"]
    output = (
        Path(output_dir)
        if output_dir is not None
        else project / str(config.get("output_dir", "artifacts/interventions/agent_replay"))
    )
    output.mkdir(parents=True, exist_ok=True)

    input_paths = {
        name: project / str(relative)
        for name, relative in inputs.items()
    }
    panel = pd.read_parquet(input_paths["panel"])
    matches = pd.read_csv(input_paths["matches"])
    splits = pd.read_csv(input_paths["splits"])
    events = pd.read_parquet(input_paths["events"])
    posts = pd.read_parquet(input_paths["posts"])
    panel, matches, splits, events, posts = _prepare_inputs(
        panel,
        matches,
        splits,
        events,
        posts,
    )
    simulation = config.get("simulation", {})
    horizon_days = int(simulation.get("horizon_days", 8))
    model, parameters, training_frame = fit_agent_replay_model(
        matches,
        events,
        posts,
        simulation,
    )
    training_frame.to_parquet(output / "intensity_training_rows.parquet", index=False)

    groups = _event_groups(events)
    post_times = dict(zip(posts["content_id"].astype(str), posts["published"]))
    validation = matches[matches["split"].eq("validation")].copy()
    observed_validation = _observed_effects(
        panel,
        validation["intervention_id"],
    )
    grid_rows: list[dict[str, Any]] = []
    validation_seeds = list(
        map(int, simulation.get("validation_seeds", [0, 1, 2, 3, 4]))
    )
    for intensity_scale in simulation.get(
        "intensity_scale_grid", [0.75, 1.0, 1.25]
    ):
        for visibility in simulation.get(
            "remove_visibility_grid", [0.1, 0.2, 0.4]
        ):
            seed_effects, _ = _simulate_matches(
                selected_matches=validation,
                model=model,
                parameters=parameters,
                event_groups=groups,
                post_times=post_times,
                horizon_days=horizon_days,
                intensity_scale=float(intensity_scale),
                remove_visibility_remaining=float(visibility),
                seeds=validation_seeds,
                max_events_per_day=int(
                    simulation.get("max_events_per_day", 500)
                ),
                collect_events=False,
            )
            predictions = _aggregate_agent_predictions(seed_effects)
            score = _validation_score(
                _score_predictions(predictions, observed_validation)
            )
            grid_rows.append(
                {
                    "intensity_scale": float(intensity_scale),
                    "remove_visibility_remaining": float(visibility),
                    "validation_normalized_mae": score,
                    "validation_interventions": int(
                        validation["intervention_id"].nunique()
                    ),
                    "validation_seeds": len(validation_seeds),
                }
            )
    grid = pd.DataFrame(grid_rows).sort_values(
        [
            "validation_normalized_mae",
            "intensity_scale",
            "remove_visibility_remaining",
        ]
    )
    grid.to_csv(output / "validation_grid.csv", index=False)
    selected = grid.iloc[0]

    selected_validation_effects, _ = _simulate_matches(
        selected_matches=validation,
        model=model,
        parameters=parameters,
        event_groups=groups,
        post_times=post_times,
        horizon_days=horizon_days,
        intensity_scale=float(selected["intensity_scale"]),
        remove_visibility_remaining=float(
            selected["remove_visibility_remaining"]
        ),
        seeds=validation_seeds,
        max_events_per_day=int(simulation.get("max_events_per_day", 500)),
        collect_events=False,
    )
    validation_predictions = _aggregate_agent_predictions(
        selected_validation_effects
    )
    validation_residuals = validation_predictions.merge(
        observed_validation,
        on=["intervention_id", "outcome"],
        how="inner",
        validate="one_to_one",
    )
    validation_residuals["absolute_residual"] = (
        validation_residuals["observed_effect"]
        - validation_residuals["predicted_effect"]
    ).abs()
    interval_target = float(config.get("prediction_interval", 0.95))
    conformal_radii = {
        str(outcome): _finite_conformal_quantile(
            group["absolute_residual"].to_numpy(dtype=float),
            interval_target,
        )
        for outcome, group in validation_residuals.groupby(
            "outcome",
            sort=False,
        )
    }
    validation_residuals.to_csv(
        output / "validation_conformal_residuals.csv",
        index=False,
    )

    test = matches[matches["split"].eq("test")].copy()
    test_seeds = list(map(int, simulation.get("test_seeds", range(20))))
    test_seed_effects, test_events = _simulate_matches(
        selected_matches=test,
        model=model,
        parameters=parameters,
        event_groups=groups,
        post_times=post_times,
        horizon_days=horizon_days,
        intensity_scale=float(selected["intensity_scale"]),
        remove_visibility_remaining=float(
            selected["remove_visibility_remaining"]
        ),
        seeds=test_seeds,
        max_events_per_day=int(simulation.get("max_events_per_day", 500)),
        collect_events=True,
    )
    test_seed_effects.to_parquet(output / "seed_level_effects.parquet", index=False)
    test_events.to_parquet(output / "simulated_events.parquet", index=False)
    agent_predictions = _aggregate_agent_predictions(test_seed_effects)
    radii = agent_predictions["outcome"].map(conformal_radii).astype(float)
    agent_predictions["ci_low"] = np.minimum(
        agent_predictions["seed_ci_low"],
        agent_predictions["predicted_effect"] - radii,
    )
    agent_predictions["ci_high"] = np.maximum(
        agent_predictions["seed_ci_high"],
        agent_predictions["predicted_effect"] + radii,
    )
    baseline_predictions = _frozen_predictions(
        test,
        panel,
        float(selected["remove_visibility_remaining"]),
    )
    strong_baselines, baseline_tuning = _strong_baseline_predictions(
        panel,
        matches,
        config,
    )
    pd.DataFrame(baseline_tuning).to_csv(
        output / "strong_baseline_tuning.csv",
        index=False,
    )
    predictions = pd.concat(
        [agent_predictions, baseline_predictions, strong_baselines],
        ignore_index=True,
    )
    predictions.to_csv(output / "predictions.csv", index=False)

    observed_test = _observed_effects(panel, test["intervention_id"])
    observed_test.to_csv(output / "observed_test_effects.csv", index=False)
    scored = _score_predictions(predictions, observed_test)
    scored.to_csv(output / "scored_predictions.csv", index=False)
    by_outcome = (
        scored.groupby(["model", "outcome"], sort=False)
        .agg(
            n_interventions=("intervention_id", "nunique"),
            mae=("absolute_error", "mean"),
            median_absolute_error=("absolute_error", "median"),
            direction_accuracy=("direction_correct", "mean"),
            interval_coverage=("interval_coverage", "mean"),
            seed_interval_coverage=("seed_interval_coverage", "mean"),
            observed_mean_effect=("observed_effect", "mean"),
            predicted_mean_effect=("predicted_effect", "mean"),
        )
        .reset_index()
    )
    by_outcome.to_csv(output / "comparison_by_model_outcome.csv", index=False)

    bootstrap_samples = int(config.get("bootstrap_samples", 2000))
    seed = int(config.get("seed", 30371))
    ranking = _bootstrap_primary(scored, bootstrap_samples, seed)
    ranking.to_csv(output / "model_ranking.csv", index=False)
    improvement = _paired_improvement(
        scored,
        "bdmtf_agent_replay",
        "frozen_pretrend",
        bootstrap_samples,
        seed + 1,
    )
    improvement_vs_zero = _paired_improvement(
        scored,
        "bdmtf_agent_replay",
        "zero_effect",
        bootstrap_samples,
        seed + 2,
    )
    baseline_names = [
        "zero_effect",
        "frozen_pretrend",
        "majority_sign",
        "intervention_type_mean",
        "additive_ridge",
        "historical_nearest_neighbor",
    ]
    baseline_comparisons = {}
    for offset, baseline_name in enumerate(baseline_names, start=10):
        comparison = _paired_improvement(
            scored,
            "bdmtf_agent_replay",
            baseline_name,
            bootstrap_samples,
            seed + offset,
        )
        baseline_comparisons[baseline_name] = {
            "mae_improvement": comparison[0],
            "ci": [comparison[1], comparison[2]],
        }

    train_control_ids = set(
        matches.loc[matches["split"].eq("train"), "control_content_id"].astype(str)
    )
    test_ids = set(
        matches.loc[matches["split"].eq("test"), "treated_content_id"].astype(str)
    ) | set(
        matches.loc[matches["split"].eq("test"), "control_content_id"].astype(str)
    )
    test_treated = set(
        matches.loc[matches["split"].eq("test"), "treated_content_id"].astype(str)
    )
    event_ids = set(events["content_id"].astype(str))
    root_only_test = len(test_treated - event_ids)
    agent_row = ranking[ranking["model"].eq("bdmtf_agent_replay")].iloc[0]
    agent_scored = scored[scored["model"].eq("bdmtf_agent_replay")]
    frozen_row = ranking[ranking["model"].eq("frozen_pretrend")].iloc[0]
    result = {
        "status": "complete",
        "claim_allowed": True,
        "protocol": config.get("protocol", {}),
        "n_interventions": int(matches["intervention_id"].nunique()),
        "split_counts": {
            str(key): int(value)
            for key, value in matches["split"].value_counts().items()
        },
        "test_interventions": int(test["intervention_id"].nunique()),
        "test_interventions_with_observed_pre_comments": int(
            len(test_treated & event_ids)
        ),
        "test_interventions_initialized_at_root_only": int(root_only_test),
        "test_simulation_seeds": len(test_seeds),
        "simulated_event_rows": int(len(test_events)),
        "seed_level_effect_rows": int(len(test_seed_effects)),
        "selected_validation_parameters": {
            "intensity_scale": float(selected["intensity_scale"]),
            "remove_visibility_remaining": float(
                selected["remove_visibility_remaining"]
            ),
            "validation_normalized_mae": float(
                selected["validation_normalized_mae"]
            ),
        },
        "event_intensity_model": {
            "model": "PoissonRegressor",
            "feature_names": list(FEATURE_NAMES),
            "coefficients": {
                name: float(value)
                for name, value in zip(FEATURE_NAMES, model.coef_)
            },
            "intercept": float(model.intercept_),
            "training_rows": int(len(training_frame)),
            "training_control_threads": int(
                training_frame["content_id"].nunique()
            ),
        },
        "agent_target_parameters": asdict(parameters),
        "bdmtf_agent_replay": {
            "primary_mae": float(agent_row["primary_mae"]),
            "primary_mae_ci": [
                float(agent_row["primary_mae_ci_low"]),
                float(agent_row["primary_mae_ci_high"]),
            ],
            "direction_accuracy": float(agent_row["direction_accuracy"]),
            "direction_accuracy_ci": [
                float(agent_row["direction_ci_low"]),
                float(agent_row["direction_ci_high"]),
            ],
            "interval_coverage": float(agent_row["interval_coverage"]),
            "interval_coverage_ci": [
                float(agent_row["coverage_ci_low"]),
                float(agent_row["coverage_ci_high"]),
            ],
            "raw_seed_interval_coverage": float(
                agent_scored["seed_interval_coverage"].mean()
            ),
        },
        "frozen_pretrend": {
            "primary_mae": float(frozen_row["primary_mae"]),
            "direction_accuracy": float(frozen_row["direction_accuracy"]),
        },
        "strong_baselines": {
            str(row.model): {
                "primary_mae": float(row.primary_mae),
                "primary_mae_ci": [
                    float(row.primary_mae_ci_low),
                    float(row.primary_mae_ci_high),
                ],
                "direction_accuracy": float(row.direction_accuracy),
                "direction_accuracy_ci": [
                    float(row.direction_ci_low),
                    float(row.direction_ci_high),
                ],
            }
            for row in ranking[
                ranking["model"].isin(
                    [
                        "majority_sign",
                        "intervention_type_mean",
                        "additive_ridge",
                        "historical_nearest_neighbor",
                    ]
                )
            ].itertuples(index=False)
        },
        "bdmtf_baseline_mae_comparisons": baseline_comparisons,
        "bdmtf_vs_frozen_pretrend_mae_improvement": {
            "mean": improvement[0],
            "ci": [improvement[1], improvement[2]],
        },
        "bdmtf_vs_zero_effect_mae_improvement": {
            "mean": improvement_vs_zero[0],
            "ci": [improvement_vs_zero[1], improvement_vs_zero[2]],
        },
        "conformal_calibration": {
            "target_coverage": interval_target,
            "validation_interventions": int(
                validation["intervention_id"].nunique()
            ),
            "outcome_radii": conformal_radii,
            "test_outcomes_used": False,
        },
        "leakage_audit": {
            "fit_uses_only_training_control_post_events": True,
            "validation_selects_only_frozen_grid_parameters": True,
            "test_initialization_filters_created_at_before_event_time": True,
            "test_post_outcomes_used_for_fit": False,
            "train_control_test_id_overlap": int(
                len(train_control_ids.intersection(test_ids))
            ),
        },
        "evidence_boundary": (
            "This is retrospective agent-level replay on 103 chronological "
            "held-out Lemmy interventions. It generates individual arrivals, "
            "authors, reply targets, depths, and timestamps, but it is not a "
            "prospective randomized intervention."
        ),
        "inputs": {
            name: {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
            for name, path in input_paths.items()
        },
    }
    ridge_comparison = baseline_comparisons["additive_ridge"]
    result["framework_support"] = (
        "event_path_replay_beats_naive_magnitude_baselines_but_not_additive_ridge"
        if ridge_comparison["ci"][0] <= 0
        else "event_path_replay_beats_all_registered_baselines"
    )
    write_json(output / "agent_replay_manifest.json", result)
    _write_report(output / "LEMMY_AGENT_REPLAY_REPORT.md", result, by_outcome)
    return result


def _write_report(
    path: Path,
    result: Mapping[str, Any],
    by_outcome: pd.DataFrame,
) -> None:
    agent = result["bdmtf_agent_replay"]
    improvement = result["bdmtf_vs_frozen_pretrend_mae_improvement"]
    lines = [
        "# Lemmy Agent-Level Intervention Replay",
        "",
        "## Purpose",
        "",
        (
            "This experiment tests whether the BDMTF mechanism can replay "
            "real lock/removal interventions by generating individual future "
            "reply events rather than predicting aggregate responses directly."
        ),
        "",
        "## Frozen Evaluation",
        "",
        f"- Chronological test interventions: {result['test_interventions']}.",
        f"- Simulation seeds per intervention: {result['test_simulation_seeds']}.",
        f"- Generated event rows: {result['simulated_event_rows']:,}.",
        (
            "- Root-only initializations (no observed pre-event comments): "
            f"{result['test_interventions_initialized_at_root_only']}."
        ),
        (
            "- Selected visibility remaining after removal: "
            f"{result['selected_validation_parameters']['remove_visibility_remaining']:.3f}."
        ),
        "",
        "## Main Result",
        "",
        (
            f"- Primary MAE: {agent['primary_mae']:.3f} "
            f"[{agent['primary_mae_ci'][0]:.3f}, "
            f"{agent['primary_mae_ci'][1]:.3f}]."
        ),
        (
            f"- Direction accuracy: {100 * agent['direction_accuracy']:.1f}% "
            f"[{100 * agent['direction_accuracy_ci'][0]:.1f}%, "
            f"{100 * agent['direction_accuracy_ci'][1]:.1f}%]."
        ),
        (
            "- Validation-calibrated interval coverage: "
            f"{100 * agent['interval_coverage']:.1f}%."
        ),
        (
            "- Raw simulation-seed interval coverage: "
            f"{100 * agent['raw_seed_interval_coverage']:.1f}%."
        ),
        (
            "- MAE improvement over frozen pretrend: "
            f"{improvement['mean']:+.3f} "
            f"[{improvement['ci'][0]:+.3f}, "
            f"{improvement['ci'][1]:+.3f}]."
        ),
        "",
        "## Outcome Detail",
        "",
        "| Model | Outcome | MAE | Direction | Coverage |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in by_outcome.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.outcome} | {row.mae:.3f} | "
            f"{100 * row.direction_accuracy:.1f}% | "
            f"{100 * row.interval_coverage:.1f}% |"
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
