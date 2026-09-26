from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from bdmtf.revision.fitted_models import CommunityModel
from bdmtf.revision.policies import EmpiricalCorrectionPolicy, EventNode, RedditRankingPolicy, TraitMode, sample_traits


MODEL_NAMES = ("empirical_bootstrap", "branching_process", "hawkes", "legacy_heuristic", "learned_bdmtf")


@dataclass(frozen=True)
class RevisionSimulationConfig:
    step_minutes: int = 10
    steps: int = 72
    num_agents: int = 50
    viewport_k: int = 5
    leader_fraction: float = 0.2
    ranking: str = "best"
    trait_mode: str = "independent"
    moderation_probability: float = 0.0
    counterspeech_probability: float = 0.0
    hostile_dropout_probability: float = 0.0
    deep_drill_lambda: float = 1.0
    activation_multiplier: float = 1.0
    external_traffic: bool = False
    conflict_intensity: float = 0.0
    max_comments: int = 5000

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "RevisionSimulationConfig":
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: raw[key] for key in allowed if key in raw})


def _draw_target_size(model_name: str, profile: CommunityModel, train_rows: pd.DataFrame, rng: np.random.Generator, config: RevisionSimulationConfig) -> int:
    if model_name == "empirical_bootstrap" and not train_rows.empty:
        target = float(train_rows.iloc[int(rng.integers(0, len(train_rows)))]["size"])
    elif model_name == "branching_process":
        reproduction = min(0.97, profile.mean_branching_factor / (profile.mean_branching_factor + 1.0))
        target = rng.negative_binomial(1, max(0.03, 1.0 - reproduction))
    elif model_name == "hawkes":
        target = rng.lognormal(profile.log_size_mean - 0.12, profile.log_size_std * 1.15)
    elif model_name == "legacy_heuristic":
        target = rng.lognormal(profile.log_size_mean, profile.log_size_std) * 1.15
    else:
        target = rng.lognormal(profile.log_size_mean, profile.log_size_std)
    target *= config.activation_multiplier
    if model_name == "legacy_heuristic":
        target *= 1.0 + 0.75 * max(0.0, config.conflict_intensity)
    else:
        target *= math.exp(profile.toxicity_log_size_effect * config.conflict_intensity)
    if config.external_traffic:
        target *= 1.1
    return int(np.clip(round(target), 1, config.max_comments))


def _arrival_minutes(model_name: str, count: int, profile: CommunityModel, config: RevisionSimulationConfig, rng: np.random.Generator) -> np.ndarray:
    horizon = config.steps * config.step_minutes
    if model_name == "hawkes":
        arrivals = []
        time = 0.0
        intensity = max(0.01, count / max(horizon, 1))
        while len(arrivals) < count and time < horizon:
            time += float(rng.exponential(1.0 / max(intensity, 1e-5)))
            if time >= horizon:
                break
            arrivals.append(time)
            intensity = intensity * math.exp(-config.step_minutes / max(profile.time_decay_minutes, 1.0)) + 0.08
        if len(arrivals) < count:
            extra = rng.uniform(0, horizon, size=count - len(arrivals))
            arrivals.extend(extra.tolist())
        return np.sort(np.asarray(arrivals[:count], dtype=float))
    scale = max(1.0, profile.time_decay_minutes)
    arrivals = rng.exponential(scale=scale, size=count)
    return np.sort(np.clip(arrivals, 0, horizon))


def _choose_parent(root: EventNode, visible: list[EventNode], ranking: RedditRankingPolicy, root_share: float, depth_bias: float, rng: random.Random, all_access: bool = False) -> EventNode:
    if not visible or rng.random() < root_share:
        return root
    if all_access and len(visible) > 64:
        sampled = []
        decay = min(0.999, max(0.01, ranking.position_decay))
        for _ in range(64):
            position = min(len(visible) - 1, int(math.log(max(1e-12, 1.0 - rng.random())) / math.log(decay)))
            sampled.append(visible[position])
        visible = list({node.node_id: node for node in sampled}.values())
    weights = []
    for position, node in enumerate(visible):
        position_weight = ranking.position_decay**position
        depth_weight = math.exp(min(6.0, depth_bias * node.depth / 8.0))
        weights.append(position_weight * depth_weight)
    return rng.choices(visible, weights=weights, k=1)[0]


def simulate_cascade(
    model_name: str,
    profile: CommunityModel,
    train_rows: pd.DataFrame,
    post_id: str,
    seed: int,
    config: RevisionSimulationConfig,
) -> list[EventNode]:
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unknown model: {model_name}")
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)
    traits = sample_traits(config.num_agents, TraitMode(config.trait_mode), seed)
    ranking = RedditRankingPolicy(config.ranking)
    correction = EmpiricalCorrectionPolicy(
        config.moderation_probability,
        config.counterspeech_probability,
        config.hostile_dropout_probability,
    )
    target_size = _draw_target_size(model_name, profile, train_rows, rng, config)
    arrivals = _arrival_minutes(model_name, target_size, profile, config, rng)
    nodes = [EventNode("post", None, 0, 0.0, author_id="post_author", metadata={"root_post": True})]
    ranked_cache: list[EventNode] = []
    dropped_agents: set[int] = set()
    leader_count = int(round(config.num_agents * config.leader_fraction))
    for index, minute in enumerate(arrivals):
        available = [agent for agent in range(config.num_agents) if agent not in dropped_agents]
        if not available:
            break
        if index < leader_count:
            agent_index = index % max(1, leader_count)
        elif py_rng.random() < profile.repeat_author_share and len(nodes) > 2:
            previous_authors = [int(node.author_id[1:]) for node in nodes[1:] if node.author_id.startswith("a")]
            agent_index = py_rng.choice(previous_authors) if previous_authors else py_rng.choice(available)
        else:
            agent_index = py_rng.choice(available)
        trait = traits[agent_index]
        root_share = profile.root_reply_share
        if model_name == "branching_process":
            root_share = min(0.95, root_share + 0.1)
        if model_name == "legacy_heuristic":
            root_share = max(0.02, root_share - 0.12 * trait.antagonism)
            root_share = min(0.99, root_share + 0.15 * max(0.0, config.conflict_intensity))
        else:
            predicted_leaf_shift = profile.toxicity_leaf_depth_effect * config.conflict_intensity
            root_share = float(np.clip(root_share - 0.05 * predicted_leaf_shift, 0.01, 0.99))
        refresh_interval = 25
        if not ranked_cache or index % refresh_interval == 0:
            candidates = [item for item in nodes[1:] if not item.removed]
            ranked = ranking.rank(candidates, float(minute))
            ranked_cache = ranked if config.viewport_k < 0 else ranked[: max(1, config.viewport_k)]
        recent = [item for item in nodes[max(1, len(nodes) - 5) :] if not item.removed]
        visible_by_id = {item.node_id: item for item in ranked_cache + recent}
        visible = list(visible_by_id.values())
        if config.viewport_k >= 0:
            visible = visible[: max(1, config.viewport_k)]
        parent = _choose_parent(nodes[0], visible, ranking, root_share, config.deep_drill_lambda * trait.antagonism, py_rng, all_access=config.viewport_k < 0)
        toxic_probability = np.clip(profile.toxicity_density * (0.5 + 1.5 * trait.antagonism) + 0.1 * config.conflict_intensity, 0.0, 1.0)
        toxicity = float(rng.uniform(0.30, 1.0)) if rng.random() < toxic_probability else float(rng.uniform(0.0, 0.12))
        likes = int(max(0, rng.poisson(1.5 + 2.0 * trait.attention)))
        dislikes = int(max(0, rng.poisson(0.3 + 1.5 * trait.antagonism)))
        node = EventNode(
            node_id=f"c{index}",
            parent_id=parent.node_id,
            depth=parent.depth + 1,
            created_minute=float(minute),
            score=float(likes - dislikes),
            likes=likes,
            dislikes=dislikes,
            toxicity=toxicity,
            author_id=f"a{agent_index}",
            metadata={"model": model_name, "post_id": post_id},
        )
        nodes.append(node)
        outcome = correction.apply(node, py_rng)
        node.metadata["correction"] = outcome
        if outcome == "dropout":
            dropped_agents.add(agent_index)
        elif outcome == "counterspeech" and len(nodes) < config.max_comments + 1:
            reply_index = len(nodes) - 1
            nodes.append(
                EventNode(
                    node_id=f"cs{reply_index}",
                    parent_id=node.node_id,
                    depth=node.depth + 1,
                    created_minute=min(float(minute) + 1.0, config.steps * config.step_minutes),
                    score=1.0,
                    likes=1,
                    toxicity=0.0,
                    author_id=f"a{py_rng.choice(available)}",
                    metadata={"model": model_name, "post_id": post_id, "counterspeech": True},
                )
            )
    return nodes


def event_metrics(nodes: Iterable[EventNode]) -> dict[str, float]:
    comments = [node for node in nodes if not node.metadata.get("root_post")]
    if not comments:
        return {key: 0.0 for key in ("size", "max_depth", "mean_leaf_depth", "mean_depth", "mean_branching_factor", "root_reply_share", "width_gini", "time_to_50_minutes", "time_to_90_minutes", "duration_minutes", "mean_interarrival_minutes", "repeat_author_share", "toxicity_density", "counterspeech_rate", "removed_rate")}
    child_counts: dict[str, int] = {}
    for node in comments:
        if node.parent_id:
            child_counts[node.parent_id] = child_counts.get(node.parent_id, 0) + 1
    comment_ids = {node.node_id for node in comments}
    non_root_child_counts = [count for parent_id, count in child_counts.items() if parent_id in comment_ids]
    leaves = [node for node in comments if node.node_id not in child_counts]
    depths = np.asarray([node.depth for node in comments], dtype=float)
    times = np.sort(np.asarray([node.created_minute for node in comments], dtype=float))
    width = pd.Series(depths).value_counts().to_numpy(dtype=float)
    authors = pd.Series([node.author_id for node in comments]).value_counts()
    return {
        "size": float(len(comments)),
        "max_depth": float(depths.max()),
        "mean_leaf_depth": float(np.mean([node.depth for node in leaves])) if leaves else 0.0,
        "mean_depth": float(depths.mean()),
        "mean_branching_factor": float(np.mean(non_root_child_counts)) if non_root_child_counts else 0.0,
        "root_reply_share": float(sum(node.parent_id == "post" for node in comments) / len(comments)),
        "width_gini": _gini(width),
        "time_to_50_minutes": float(np.quantile(times, 0.5)),
        "time_to_90_minutes": float(np.quantile(times, 0.9)),
        "duration_minutes": float(times.max() - times.min()) if len(times) > 1 else 0.0,
        "mean_interarrival_minutes": float(np.diff(times).mean()) if len(times) > 1 else 0.0,
        "repeat_author_share": float(authors[authors > 1].sum() / len(comments)),
        "toxicity_density": float(np.mean([node.toxicity >= 0.30 for node in comments])),
        "counterspeech_rate": float(np.mean([bool(node.metadata.get("counterspeech")) for node in comments])),
        "removed_rate": float(np.mean([node.removed for node in comments])),
    }


def _gini(values: np.ndarray) -> float:
    values = np.sort(np.maximum(np.asarray(values, dtype=float), 0.0))
    if len(values) == 0 or values.sum() == 0:
        return 0.0
    index = np.arange(1, len(values) + 1)
    return float(np.sum((2 * index - len(values) - 1) * values) / (len(values) * values.sum()))


def nodes_to_records(nodes: Iterable[EventNode], context: dict[str, Any]) -> list[dict[str, Any]]:
    return [{**context, **asdict(node), "metadata": str(node.metadata)} for node in nodes]


def metrics_from_event_records(events: pd.DataFrame) -> pd.DataFrame:
    """Recompute summary metrics from saved event Parquet without resimulation."""
    node_columns = {
        "node_id",
        "parent_id",
        "depth",
        "created_minute",
        "score",
        "likes",
        "dislikes",
        "toxicity",
        "removed",
        "author_id",
        "metadata",
    }
    context_columns = [name for name in events.columns if name not in node_columns]
    grouping = [name for name in ("platform", "community", "post_id", "model", "seed", "split", "fit_scope") if name in events]
    records: list[dict[str, Any]] = []
    for _, group in events.groupby(grouping, dropna=False, sort=False):
        nodes = []
        for row in group.itertuples(index=False):
            node_id = str(row.node_id)
            parent = None if pd.isna(row.parent_id) else str(row.parent_id)
            nodes.append(
                EventNode(
                    node_id=node_id,
                    parent_id=parent,
                    depth=int(row.depth),
                    created_minute=float(row.created_minute),
                    score=float(row.score),
                    likes=int(row.likes),
                    dislikes=int(row.dislikes),
                    toxicity=float(row.toxicity),
                    removed=bool(row.removed),
                    author_id=str(row.author_id),
                    metadata={"root_post": node_id == "post", "counterspeech": node_id.startswith("cs")},
                )
            )
        context = {name: group.iloc[0][name] for name in context_columns}
        records.append({**context, **event_metrics(nodes)})
    return pd.DataFrame(records)
