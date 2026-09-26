from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bdmtf.data.social_loader import load_posts, resolve_community_paths
from bdmtf.revision.policies import EventNode, RedditRankingPolicy
from bdmtf.revision.provenance import sha256_file, write_json
from bdmtf.revision.simulation import event_metrics


NUMERIC_CONTEXT_FEATURES = (
    "early_num_comments",
    "early_score_sum",
    "early_depth_max",
    "early_comment_length_avg",
    "author_link_karma",
    "author_comment_karma",
    "subreddit_subscribers",
)
BINARY_CONTEXT_FEATURES = ("is_self", "is_video")
PRIMARY_SCALAR_METRICS = (
    "mean_leaf_depth",
    "max_depth",
    "root_reply_share",
    "mean_branching_factor",
)
SECONDARY_METRICS = (
    "width_gini",
    "leaf_fraction",
    "depth_variance",
    "p90_depth",
    "repeat_author_share",
)
FACTORIAL_CELLS = {
    "B0_P0": (False, "best"),
    "B0_P1": (False, "controversial"),
    "B1_P0": (True, "best"),
    "B1_P1": (True, "controversial"),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**32 - 1)


def _as_bool(value: Any) -> float:
    return float(str(value).strip().lower() in {"1", "true", "yes", "t"})


def _context_vector(row: pd.Series | dict[str, Any], communities: list[str]) -> np.ndarray:
    values: list[float] = []
    for name in NUMERIC_CONTEXT_FEATURES:
        raw = pd.to_numeric(pd.Series([row.get(name, 0)]), errors="coerce").iloc[0]
        value = 0.0 if pd.isna(raw) else float(raw)
        values.append(math.log1p(max(0.0, value)))
    values.extend(_as_bool(row.get(name, False)) for name in BINARY_CONTEXT_FEATURES)
    community = str(row.get("community", ""))
    values.extend(float(community == name) for name in communities)
    return np.asarray(values, dtype=float)


def load_context_frame(
    social_root: Path,
    metrics: pd.DataFrame,
    communities: list[str],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for community in communities:
        paths = resolve_community_paths(social_root, community)
        posts = load_posts(paths, enriched=False).copy()
        posts["post_id"] = posts["post_id"].astype(str)
        posts["community"] = community
        frames.append(posts)
    contexts = pd.concat(frames, ignore_index=True)
    metric_columns = [
        "community",
        "post_id",
        "split",
        "size",
        "max_depth",
        "mean_leaf_depth",
        "mean_branching_factor",
        "root_reply_share",
        "width_gini",
        "repeat_author_share",
        "n_leaf_nodes",
    ]
    available = [name for name in metric_columns if name in metrics.columns]
    merged = contexts.merge(
        metrics[available], on=["community", "post_id"], how="inner", validate="one_to_one"
    )
    return merged


def load_cascade_shapes(
    social_root: Path,
    metrics: pd.DataFrame,
    communities: list[str],
) -> pd.DataFrame:
    lookup = metrics.set_index(["community", "post_id"], drop=False)
    records: list[dict[str, Any]] = []
    for community in communities:
        path = social_root / f"{community}_data" / "empirical_cascades_fixed.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            raw = json.loads(line)
            key = (community, str(raw["post_id"]))
            if key not in lookup.index:
                continue
            metric = lookup.loc[key]
            depths = [int(value) for value in raw.get("depths", []) if int(value) >= 1]
            if not depths:
                continue
            counts = np.bincount(np.asarray(depths, dtype=int))[1:]
            pmf = counts / counts.sum()
            records.append(
                {
                    "community": community,
                    "post_id": str(raw["post_id"]),
                    "split": str(metric["split"]),
                    "size": int(metric["size"]),
                    "max_depth": int(max(depths)),
                    "mean_branching_factor": float(metric["mean_branching_factor"]),
                    "root_reply_share": float(metric["root_reply_share"]),
                    "depths": depths,
                    "depth_pmf": pmf.tolist(),
                }
            )
    return pd.DataFrame(records)


def fit_frozen_topology_model(
    contexts: pd.DataFrame,
    shapes: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    communities = [str(value) for value in config["communities"]]
    train = contexts[contexts["split"].eq("train")].copy()
    if train.empty:
        raise ValueError("No training contexts are available")
    x = np.vstack([_context_vector(row, communities) for _, row in train.iterrows()])
    y = np.log1p(train["size"].to_numpy(dtype=float))
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    standardized = (x - mean) / scale
    design = np.column_stack([np.ones(len(standardized)), standardized])
    alpha = float(config["model"]["ridge_alpha"])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    fitted = design @ beta
    residual_std = float(max(np.std(y - fitted, ddof=1), 0.05))

    train_shapes = shapes[shapes["split"].eq("train")].copy()
    by_community: dict[str, Any] = {}
    for community in communities:
        group = train_shapes[train_shapes["community"].eq(community)]
        if group.empty:
            raise ValueError(f"No training cascade shapes for {community}")
        by_community[community] = {
            "training_post_ids": sorted(group["post_id"].astype(str).tolist()),
            "cascades": [
                {
                    "post_id": str(row.post_id),
                    "size": int(row.size),
                    "max_depth": int(row.max_depth),
                    "mean_branching_factor": float(row.mean_branching_factor),
                    "root_reply_share": float(row.root_reply_share),
                    "depth_pmf": [float(value) for value in row.depth_pmf],
                }
                for row in group.itertuples(index=False)
            ],
        }
    feature_names = [
        *(f"log1p_{name}" for name in NUMERIC_CONTEXT_FEATURES),
        *BINARY_CONTEXT_FEATURES,
        *(f"community={name}" for name in communities),
    ]
    return {
        "schema_version": 1,
        "fit_scope": "chronological training split only",
        "uses_test_outcomes": False,
        "size_regressor": {
            "feature_names": feature_names,
            "feature_mean": [float(value) for value in mean],
            "feature_scale": [float(value) for value in scale],
            "intercept": float(beta[0]),
            "coefficients": [float(value) for value in beta[1:]],
            "residual_std": residual_std,
            "ridge_alpha": alpha,
            "n_train": int(len(train)),
        },
        "communities": by_community,
    }


def predict_log_size(model: dict[str, Any], context: dict[str, Any] | pd.Series) -> float:
    communities = list(model["communities"])
    predictor = model["size_regressor"]
    vector = _context_vector(context, communities)
    mean = np.asarray(predictor["feature_mean"], dtype=float)
    scale = np.asarray(predictor["feature_scale"], dtype=float)
    coefficients = np.asarray(predictor["coefficients"], dtype=float)
    return float(predictor["intercept"] + ((vector - mean) / scale) @ coefficients)


def _local_topology_profile(
    model: dict[str, Any],
    community: str,
    target_size: int,
    neighbors: int,
) -> dict[str, Any]:
    cascades = list(model["communities"][community]["cascades"])
    distances = np.asarray(
        [abs(math.log1p(item["size"]) - math.log1p(target_size)) for item in cascades],
        dtype=float,
    )
    order = np.argsort(distances, kind="stable")[: min(neighbors, len(cascades))]
    selected = [cascades[int(index)] for index in order]
    selected_distances = distances[order]
    bandwidth = max(float(np.median(selected_distances)), 0.10)
    weights = np.exp(-selected_distances / bandwidth)
    weights /= weights.sum()
    max_depth = max(len(item["depth_pmf"]) for item in selected)
    pmf = np.zeros(max_depth, dtype=float)
    for weight, item in zip(weights, selected, strict=True):
        values = np.asarray(item["depth_pmf"], dtype=float)
        pmf[: len(values)] += weight * values
    pmf /= pmf.sum()
    return {
        "depth_pmf": pmf,
        "mean_branching_factor": float(
            np.average(
                [item["mean_branching_factor"] for item in selected], weights=weights
            )
        ),
        "root_reply_share": float(
            np.average([item["root_reply_share"] for item in selected], weights=weights)
        ),
        "source_post_ids": [item["post_id"] for item in selected],
    }


def _proposal_pool(seed: int, count: int) -> list[dict[str, float]]:
    rng = np.random.default_rng(seed)
    antagonism = rng.beta(2.0, 4.0, size=count)
    attention = rng.beta(2.2, 3.2, size=count)
    likes = rng.poisson(1.5 + 2.0 * attention)
    dislikes = rng.poisson(0.3 + 1.5 * antagonism)
    total = likes + dislikes
    balance = 1.0 - np.abs(likes - dislikes) / (total + 1.0)
    conflict_signal = 0.6 * antagonism + 0.4 * balance
    return [
        {
            "antagonism": float(antagonism[index]),
            "attention": float(attention[index]),
            "likes": int(likes[index]),
            "dislikes": int(dislikes[index]),
            "conflict_signal": float(conflict_signal[index]),
        }
        for index in range(count)
    ]


def _exposure_signal(
    proposals: list[dict[str, float]],
    ranking_mode: str,
    viewport: int,
) -> float:
    nodes = [
        EventNode(
            node_id=f"proposal_{index}",
            parent_id="post",
            depth=1,
            created_minute=float(index),
            likes=int(item["likes"]),
            dislikes=int(item["dislikes"]),
            score=0.0,
            metadata={"proposal_index": index},
        )
        for index, item in enumerate(proposals)
    ]
    ranked = RedditRankingPolicy(ranking_mode).rank(nodes, float(len(nodes)))
    visible = ranked[: max(1, min(viewport, len(ranked)))]
    return float(
        np.mean(
            [proposals[int(node.metadata["proposal_index"])]["conflict_signal"] for node in visible]
        )
    )


def _sample_depth_counts(
    size: int,
    base_pmf: np.ndarray,
    behavior: bool,
    exposure_signal: float,
    depth_fatigue_gain: float,
    rng: np.random.Generator,
) -> np.ndarray:
    pmf = np.asarray(base_pmf, dtype=float).copy()
    if behavior:
        depths = np.arange(1, len(pmf) + 1, dtype=float)
        pmf *= np.exp(-depth_fatigue_gain * exposure_signal * (depths - 1.0))
    pmf = np.maximum(pmf, 0.0)
    pmf /= pmf.sum()
    counts = rng.multinomial(size, pmf)
    if counts[0] == 0:
        donor = int(np.argmax(counts))
        counts[donor] -= 1
        counts[0] += 1
    deepest = max((index for index, count in enumerate(counts) if count > 0), default=0)
    for index in range(1, deepest + 1):
        if counts[index] > 0:
            continue
        donors = [position for position in range(deepest + 1) if counts[position] > 1]
        if donors:
            donor = max(donors, key=lambda position: counts[position])
            counts[donor] -= 1
            counts[index] += 1
            continue
        counts[index - 1] += int(counts[index + 1 :].sum())
        counts[index:] = 0
        deepest = index - 1
        break
    return counts[: deepest + 1]


def _build_tree(
    counts: np.ndarray,
    proposals: list[dict[str, float]],
    branch_factor: float,
    ranking_mode: str,
    seed: int,
    post_id: str,
) -> list[EventNode]:
    rng = np.random.default_rng(seed)
    root = EventNode(
        "post", None, 0, 0.0, author_id="post_author", metadata={"root_post": True}
    )
    nodes = [root]
    by_depth: dict[int, list[EventNode]] = {0: [root]}
    proposal_index = 0
    ranking = RedditRankingPolicy(ranking_mode)
    for depth, count in enumerate(counts, start=1):
        parents = by_depth[depth - 1]
        if depth == 1:
            active_parents = parents
        else:
            desired_internal = max(1, int(round(float(count) / max(branch_factor, 1.0))))
            desired_internal = min(len(parents), desired_internal)
            active_parents = ranking.rank(parents, float(depth * 60))[:desired_internal]
        current: list[EventNode] = []
        for offset in range(int(count)):
            parent = active_parents[offset % len(active_parents)]
            proposal = proposals[proposal_index]
            created = min(
                720.0,
                parent.created_minute + float(rng.exponential(18.0 + 7.0 * depth)),
            )
            toxic_probability = min(1.0, 0.02 + 0.18 * proposal["antagonism"])
            toxicity = (
                float(rng.uniform(0.30, 1.0))
                if rng.random() < toxic_probability
                else float(rng.uniform(0.0, 0.12))
            )
            node = EventNode(
                node_id=f"c{proposal_index}",
                parent_id=parent.node_id,
                depth=depth,
                created_minute=created,
                likes=int(proposal["likes"]),
                dislikes=int(proposal["dislikes"]),
                score=float(proposal["likes"] - proposal["dislikes"]),
                toxicity=toxicity,
                author_id=f"a{proposal_index % 50}",
                metadata={"model": "learned_topology_bdmtf", "post_id": post_id},
            )
            nodes.append(node)
            current.append(node)
            proposal_index += 1
        by_depth[depth] = current
    return nodes


def extended_topology_metrics(nodes: list[EventNode]) -> dict[str, Any]:
    metrics = event_metrics(nodes)
    comments = [node for node in nodes if not node.metadata.get("root_post")]
    depths = np.asarray([node.depth for node in comments], dtype=int)
    child_ids = {node.parent_id for node in comments if node.parent_id not in {None, "post"}}
    leaves = [node for node in comments if node.node_id not in child_ids]
    counts = np.bincount(depths)[1:] if len(depths) else np.zeros(1, dtype=int)
    metrics.update(
        {
            "leaf_fraction": float(len(leaves) / len(comments)) if comments else 0.0,
            "depth_variance": float(np.var(depths)) if len(depths) else 0.0,
            "p90_depth": float(np.quantile(depths, 0.90)) if len(depths) else 0.0,
            "depth_histogram": [int(value) for value in counts],
        }
    )
    return metrics


def simulate_learned_topology(
    model: dict[str, Any],
    context: dict[str, Any] | pd.Series,
    seed: int,
    config: dict[str, Any],
    *,
    behavior: bool,
    ranking_mode: str,
) -> tuple[list[EventNode], dict[str, Any]]:
    model_config = config["model"]
    operator = config["operators"]
    community = str(context["community"])
    post_id = str(context["post_id"])
    block_seed = _stable_seed(config["seed"], community, post_id, seed)
    size_rng = np.random.default_rng(_stable_seed(block_seed, "size"))
    predicted_log_size = predict_log_size(model, context)
    base_size = int(
        np.clip(
            round(
                math.expm1(
                    predicted_log_size
                    + size_rng.normal(0.0, float(model["size_regressor"]["residual_std"]))
                )
            ),
            1,
            int(model_config["max_comments"]),
        )
    )
    local = _local_topology_profile(
        model, community, base_size, int(model_config["local_neighbors"])
    )
    exposure_proposals = _proposal_pool(
        _stable_seed(block_seed, "proposals"), int(operator["exposure_pool_size"])
    )
    exposure = _exposure_signal(
        exposure_proposals, ranking_mode, int(operator["exposure_viewport"])
    )
    multiplier = (
        math.exp(float(operator["activation_log_gain"]) * exposure) if behavior else 1.0
    )
    realized_size = int(
        np.clip(
            round(base_size * multiplier),
            1,
            int(model_config["max_comments"]),
        )
    )
    counts = _sample_depth_counts(
        realized_size,
        np.asarray(local["depth_pmf"], dtype=float),
        behavior,
        exposure,
        float(operator["depth_fatigue_gain"]),
        np.random.default_rng(_stable_seed(block_seed, "depths")),
    )
    proposals = _proposal_pool(_stable_seed(block_seed, "proposals"), realized_size)
    nodes = _build_tree(
        counts,
        proposals,
        float(local["mean_branching_factor"]),
        ranking_mode,
        _stable_seed(block_seed, "tree"),
        post_id,
    )
    return nodes, {
        "predicted_log_size": predicted_log_size,
        "base_size": base_size,
        "realized_size": realized_size,
        "exposure_signal": exposure,
        "activation_multiplier": multiplier,
        "local_root_reply_share": float(local["root_reply_share"]),
        "local_branching_factor": float(local["mean_branching_factor"]),
        "local_source_posts": "|".join(local["source_post_ids"]),
    }


def _post_bootstrap_metric(
    real: np.ndarray,
    simulated: np.ndarray,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    scale = max(float(np.std(real, ddof=1)), abs(float(np.mean(real))) * 0.1, 1e-9)
    gap = simulated - real
    boot_gap = np.empty(bootstrap_samples, dtype=float)
    boot_nwd = np.empty(bootstrap_samples, dtype=float)
    for index in range(bootstrap_samples):
        sample = rng.integers(0, len(real), size=len(real))
        boot_gap[index] = float(np.mean(gap[sample]))
        boot_nwd[index] = float(
            wasserstein_distance(real[sample], simulated[sample]) / scale
        )
    return {
        "real_mean": float(np.mean(real)),
        "simulated_mean": float(np.mean(simulated)),
        "mean_gap": float(np.mean(gap)),
        "gap_ci_low": float(np.quantile(boot_gap, 0.025)),
        "gap_ci_high": float(np.quantile(boot_gap, 0.975)),
        "normalized_wasserstein": float(wasserstein_distance(real, simulated) / scale),
        "nwd_ci_low": float(np.quantile(boot_nwd, 0.025)),
        "nwd_ci_high": float(np.quantile(boot_nwd, 0.975)),
    }


def _pad_pmf(values: list[float], length: int) -> np.ndarray:
    result = np.zeros(length, dtype=float)
    result[: min(length, len(values))] = values[:length]
    total = result.sum()
    return result / total if total > 0 else result


def _depth_distribution_summary(
    real_profiles: dict[str, list[float]],
    simulated_profiles: dict[str, list[float]],
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    post_ids = sorted(set(real_profiles) & set(simulated_profiles))
    length = max(
        max(len(real_profiles[post_id]) for post_id in post_ids),
        max(len(simulated_profiles[post_id]) for post_id in post_ids),
    )
    real = np.vstack([_pad_pmf(real_profiles[post_id], length) for post_id in post_ids])
    simulated = np.vstack(
        [_pad_pmf(simulated_profiles[post_id], length) for post_id in post_ids]
    )
    depths = np.arange(1, length + 1, dtype=float)
    real_mean = real.mean(axis=0)
    simulated_mean = simulated.mean(axis=0)
    real_location = float(np.sum(depths * real_mean))
    scale = max(
        float(np.sqrt(np.sum(((depths - real_location) ** 2) * real_mean))),
        1e-9,
    )
    distance = float(
        wasserstein_distance(depths, depths, u_weights=real_mean, v_weights=simulated_mean)
        / scale
    )
    boot = np.empty(bootstrap_samples, dtype=float)
    for index in range(bootstrap_samples):
        sample = rng.integers(0, len(post_ids), size=len(post_ids))
        observed = real[sample].mean(axis=0)
        generated = simulated[sample].mean(axis=0)
        boot[index] = float(
            wasserstein_distance(depths, depths, u_weights=observed, v_weights=generated)
            / scale
        )
    return (
        {
            "metric": "node_depth_distribution",
            "normalized_wasserstein": distance,
            "nwd_ci_low": float(np.quantile(boot, 0.025)),
            "nwd_ci_high": float(np.quantile(boot, 0.975)),
        },
        real_mean,
        simulated_mean,
    )


def summarize_topology_fidelity(
    real: pd.DataFrame,
    simulations: pd.DataFrame,
    real_profiles: dict[str, list[float]],
    simulated_profiles: dict[str, list[float]],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    bootstrap_samples = int(config["analysis"]["bootstrap_samples"])
    rng = np.random.default_rng(int(config["seed"]))
    post_sim = simulations.groupby(["community", "post_id"], as_index=False)[
        list(PRIMARY_SCALAR_METRICS) + list(SECONDARY_METRICS)
    ].mean()
    paired = real.merge(
        post_sim,
        on=["community", "post_id"],
        suffixes=("_real", "_simulated"),
        validate="one_to_one",
    )
    rows: list[dict[str, Any]] = []
    for metric in PRIMARY_SCALAR_METRICS + SECONDARY_METRICS:
        result = _post_bootstrap_metric(
            paired[f"{metric}_real"].to_numpy(dtype=float),
            paired[f"{metric}_simulated"].to_numpy(dtype=float),
            bootstrap_samples,
            rng,
        )
        rows.append({"metric": metric, **result})
    depth_result, real_depth, simulated_depth = _depth_distribution_summary(
        real_profiles, simulated_profiles, bootstrap_samples, rng
    )
    rows.append(depth_result)
    results = pd.DataFrame(rows)

    paired["size_stratum"] = pd.qcut(
        paired["size"],
        q=5,
        labels=["Q1-small", "Q2", "Q3-medium", "Q4", "Q5-large"],
        duplicates="drop",
    ).astype(str)
    strata_rows: list[dict[str, Any]] = []
    for stratum, group in paired.groupby("size_stratum", sort=True):
        for metric in PRIMARY_SCALAR_METRICS:
            stats = _post_bootstrap_metric(
                group[f"{metric}_real"].to_numpy(dtype=float),
                group[f"{metric}_simulated"].to_numpy(dtype=float),
                min(1000, bootstrap_samples),
                rng,
            )
            strata_rows.append(
                {"size_stratum": stratum, "metric": metric, "posts": len(group), **stats}
            )
    community_rows: list[dict[str, Any]] = []
    for community, group in paired.groupby("community", sort=True):
        for metric in PRIMARY_SCALAR_METRICS:
            stats = _post_bootstrap_metric(
                group[f"{metric}_real"].to_numpy(dtype=float),
                group[f"{metric}_simulated"].to_numpy(dtype=float),
                min(1000, bootstrap_samples),
                rng,
            )
            community_rows.append(
                {"community": community, "metric": metric, "posts": len(group), **stats}
            )

    scalar_primary = results[results["metric"].isin(PRIMARY_SCALAR_METRICS)]
    scalar_supported = int(
        (
            scalar_primary["normalized_wasserstein"]
            <= float(config["analysis"]["scalar_nwd_support_threshold"])
        ).sum()
    )
    depth_nwd = float(depth_result["normalized_wasserstein"])
    if scalar_supported == 4 and depth_nwd <= float(
        config["analysis"]["depth_nwd_strong_threshold"]
    ):
        classification = "strong topology fidelity"
    elif scalar_supported >= 2 and depth_nwd <= float(
        config["analysis"]["depth_nwd_partial_threshold"]
    ):
        classification = "partial topology fidelity"
    else:
        classification = "weak topology fidelity"
    summary = {
        "posts": int(len(paired)),
        "seeds_per_post": int(simulations["seed"].nunique()),
        "classification": classification,
        "scalar_primary_metrics_with_nwd_at_most_threshold": scalar_supported,
        "scalar_primary_metric_count": 4,
        "node_depth_normalized_wasserstein": depth_nwd,
        "no_posthoc_scaling": True,
        "real_depth_profile": [float(value) for value in real_depth],
        "simulated_depth_profile": [float(value) for value in simulated_depth],
    }
    return results, summary, pd.DataFrame(strata_rows), pd.DataFrame(community_rows)


def _cluster_bootstrap_mean(
    values: np.ndarray, bootstrap_samples: int, rng: np.random.Generator
) -> tuple[float, float, float]:
    boot = np.empty(bootstrap_samples, dtype=float)
    for index in range(bootstrap_samples):
        sample = rng.integers(0, len(values), size=len(values))
        boot[index] = float(np.mean(values[sample]))
    return (
        float(np.mean(values)),
        float(np.quantile(boot, 0.025)),
        float(np.quantile(boot, 0.975)),
    )


def summarize_learned_interaction(
    runs: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    required_cells = set(FACTORIAL_CELLS)
    if set(runs["cell"].unique()) != required_cells:
        raise ValueError("Factorial runs do not contain exactly the four prespecified cells")
    effect_rows: list[dict[str, Any]] = []
    for (community, post_id, seed), group in runs.groupby(
        ["community", "post_id", "seed"], sort=True
    ):
        cells = group.set_index("cell")
        log_sizes = np.log1p(cells["size"])
        volume_interaction = float(
            log_sizes["B1_P1"]
            - log_sizes["B1_P0"]
            - log_sizes["B0_P1"]
            + log_sizes["B0_P0"]
        )
        depth_interaction = float(
            cells.loc["B1_P1", "mean_leaf_depth"]
            - cells.loc["B1_P0", "mean_leaf_depth"]
            - cells.loc["B0_P1", "mean_leaf_depth"]
            + cells.loc["B0_P0", "mean_leaf_depth"]
        )
        effect_rows.append(
            {
                "community": community,
                "post_id": post_id,
                "seed": int(seed),
                "log_volume_interaction": volume_interaction,
                "volume_ratio_of_ratios": math.exp(volume_interaction),
                "depth_interaction": depth_interaction,
                "controversy_exposure_gain": float(
                    cells.loc["B1_P1", "exposure_signal"]
                    - cells.loc["B1_P0", "exposure_signal"]
                ),
            }
        )
    effects = pd.DataFrame(effect_rows)
    post_effects = effects.groupby(["community", "post_id"], as_index=False).mean(
        numeric_only=True
    )
    bootstrap_samples = int(config["analysis"]["bootstrap_samples"])
    rng = np.random.default_rng(int(config["seed"]) + 1)
    log_estimate, log_low, log_high = _cluster_bootstrap_mean(
        post_effects["log_volume_interaction"].to_numpy(dtype=float),
        bootstrap_samples,
        rng,
    )
    depth_estimate, depth_low, depth_high = _cluster_bootstrap_mean(
        post_effects["depth_interaction"].to_numpy(dtype=float),
        bootstrap_samples,
        rng,
    )
    exposure_estimate, exposure_low, exposure_high = _cluster_bootstrap_mean(
        post_effects["controversy_exposure_gain"].to_numpy(dtype=float),
        bootstrap_samples,
        rng,
    )

    cell_post = runs.groupby(["community", "post_id", "cell"], as_index=False)[
        ["size", "mean_leaf_depth"]
    ].mean()
    cell_rows: list[dict[str, Any]] = []
    for cell, group in cell_post.groupby("cell", sort=True):
        log_volume = np.log1p(group["size"].to_numpy(dtype=float))
        depth = group["mean_leaf_depth"].to_numpy(dtype=float)
        log_cell = _cluster_bootstrap_mean(log_volume, bootstrap_samples, rng)
        depth_cell = _cluster_bootstrap_mean(depth, bootstrap_samples, rng)
        cell_rows.append(
            {
                "cell": cell,
                "posts": len(group),
                "mean_log1p_volume": log_cell[0],
                "log1p_volume_ci_low": log_cell[1],
                "log1p_volume_ci_high": log_cell[2],
                "mean_leaf_depth": depth_cell[0],
                "mean_leaf_depth_ci_low": depth_cell[1],
                "mean_leaf_depth_ci_high": depth_cell[2],
            }
        )

    ror = math.exp(log_estimate)
    ror_low = math.exp(log_low)
    ror_high = math.exp(log_high)
    if ror_low > 1.0 and depth_high < 0.0:
        classification = "strong replication"
    elif ror > 1.0 and depth_estimate < 0.0:
        classification = "directional replication"
    elif ror_low > 1.0 and depth_low <= 0.0 <= depth_high:
        classification = "partial replication"
    else:
        classification = "mechanism does not transport"
    summary = {
        "posts": int(post_effects["post_id"].nunique()),
        "seeds_per_post": int(runs["seed"].nunique()),
        "factorial_runs": int(len(runs)),
        "volume_ratio_of_ratios": ror,
        "volume_ratio_of_ratios_ci": [math.exp(log_low), math.exp(log_high)],
        "log_volume_interaction": log_estimate,
        "log_volume_interaction_ci": [log_low, log_high],
        "depth_interaction": depth_estimate,
        "depth_interaction_ci": [depth_low, depth_high],
        "controversy_exposure_gain": exposure_estimate,
        "controversy_exposure_gain_ci": [exposure_low, exposure_high],
        "classification": classification,
        "uses_raw_depth_without_scaling": True,
    }
    return effects, pd.DataFrame(cell_rows), summary


def _plot_topology_profiles(summary: dict[str, Any], path: Path) -> None:
    real = np.asarray(summary["real_depth_profile"], dtype=float)
    simulated = np.asarray(summary["simulated_depth_profile"], dtype=float)
    length = max(len(real), len(simulated))
    real = _pad_pmf(real.tolist(), length)
    simulated = _pad_pmf(simulated.tolist(), length)
    depths = np.arange(1, length + 1)
    figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.0))
    axes[0].step(depths, np.cumsum(real), where="post", label="Observed Reddit", color="#202020")
    axes[0].step(depths, np.cumsum(simulated), where="post", label="Learned BDMTF", color="#167D8D")
    axes[0].set_xlabel("Reply depth")
    axes[0].set_ylabel("Cumulative node share")
    axes[0].set_ylim(0.0, 1.02)
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(depths, real, marker="o", markersize=3, label="Observed Reddit", color="#202020")
    axes[1].plot(depths, simulated, marker="s", markersize=3, label="Learned BDMTF", color="#B43C3C")
    axes[1].set_xlabel("Reply depth")
    axes[1].set_ylabel("Mean node share")
    for axis in axes:
        axis.grid(color="#DDDDDD", linewidth=0.6)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def _write_latex_outputs(
    output_dir: Path,
    topology: pd.DataFrame,
    topology_summary: dict[str, Any],
    cells: pd.DataFrame,
    interaction: dict[str, Any],
) -> None:
    primary = topology[topology["metric"].isin(PRIMARY_SCALAR_METRICS)].set_index("metric")
    labels = {
        "mean_leaf_depth": "Mean leaf depth",
        "max_depth": "Maximum depth",
        "root_reply_share": "Root-reply share",
        "mean_branching_factor": "Branching factor",
    }
    lines = [
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Metric & Reddit & Learned & NWD \\\\",
        "\\midrule",
    ]
    for metric in PRIMARY_SCALAR_METRICS:
        row = primary.loc[metric]
        lines.append(
            f"{labels[metric]} & {row['real_mean']:.3f} & "
            f"{row['simulated_mean']:.3f} & {row['normalized_wasserstein']:.3f} \\\\"
        )
    lines.append(
        "Node-depth distribution & -- & -- & "
        f"{topology_summary['node_depth_normalized_wasserstein']:.3f} \\\\"
    )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (output_dir / "table_topology_fidelity.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    by_cell = cells.set_index("cell")
    lines = [
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Outcome & $B_0P_0$ & $B_1P_0$ & $B_0P_1$ & $B_1P_1$ & Interaction \\\\",
        "\\midrule",
        "Log volume & "
        f"{by_cell.loc['B0_P0', 'mean_log1p_volume']:.3f} & "
        f"{by_cell.loc['B1_P0', 'mean_log1p_volume']:.3f} & "
        f"{by_cell.loc['B0_P1', 'mean_log1p_volume']:.3f} & "
        f"{by_cell.loc['B1_P1', 'mean_log1p_volume']:.3f} & "
        f"{interaction['log_volume_interaction']:.3f} \\\\",
        "Mean leaf depth & "
        f"{by_cell.loc['B0_P0', 'mean_leaf_depth']:.3f} & "
        f"{by_cell.loc['B1_P0', 'mean_leaf_depth']:.3f} & "
        f"{by_cell.loc['B0_P1', 'mean_leaf_depth']:.3f} & "
        f"{by_cell.loc['B1_P1', 'mean_leaf_depth']:.3f} & "
        f"{interaction['depth_interaction']:.3f} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
    ]
    (output_dir / "table_learned_topology_interaction.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    macros = (
        "% Auto-generated by run_topology_fidelity.py.\n"
        f"\\providecommand{{\\TopologyAuditPosts}}{{{topology_summary['posts']}}}\n"
        f"\\providecommand{{\\TopologyAuditSeeds}}{{{topology_summary['seeds_per_post']}}}\n"
        f"\\providecommand{{\\TopologyDepthNWD}}{{{topology_summary['node_depth_normalized_wasserstein']:.3f}}}\n"
        f"\\providecommand{{\\LearnedInteractionRuns}}{{{interaction['factorial_runs']}}}\n"
        f"\\providecommand{{\\LearnedVolumeROR}}{{{interaction['volume_ratio_of_ratios']:.3f}}}\n"
        f"\\providecommand{{\\LearnedVolumeRORLow}}{{{interaction['volume_ratio_of_ratios_ci'][0]:.3f}}}\n"
        f"\\providecommand{{\\LearnedVolumeRORHigh}}{{{interaction['volume_ratio_of_ratios_ci'][1]:.3f}}}\n"
        f"\\providecommand{{\\LearnedDepthInteraction}}{{{interaction['depth_interaction']:.3f}}}\n"
        f"\\providecommand{{\\LearnedDepthInteractionLow}}{{{interaction['depth_interaction_ci'][0]:.3f}}}\n"
        f"\\providecommand{{\\LearnedDepthInteractionHigh}}{{{interaction['depth_interaction_ci'][1]:.3f}}}\n"
    )
    (output_dir / "topology_fidelity_macros.tex").write_text(macros, encoding="utf-8")


def freeze_topology_design(
    repo_root: Path,
    config_path: Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir = repo_root / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "freeze_manifest.json"
    if manifest_path.exists() and not force:
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    metrics_path = repo_root / config["metrics_path"]
    splits_path = repo_root / config["splits_path"]
    social_root = repo_root / config["social_root"]
    metrics = pd.read_csv(metrics_path, dtype={"post_id": str})
    contexts = load_context_frame(social_root, metrics, config["communities"])
    shapes = load_cascade_shapes(social_root, metrics, config["communities"])
    model = fit_frozen_topology_model(contexts, shapes, config)
    model_path = output_dir / "frozen_topology_model.json"
    write_json(model_path, model)

    test = contexts[contexts["split"].eq("test")].copy()
    test["predicted_log_size"] = [predict_log_size(model, row) for _, row in test.iterrows()]
    selected_columns = [
        "community",
        "post_id",
        *NUMERIC_CONTEXT_FEATURES,
        *BINARY_CONTEXT_FEATURES,
        "predicted_log_size",
    ]
    test_path = output_dir / "frozen_test_posts.csv"
    test[selected_columns].sort_values(["community", "post_id"]).to_csv(test_path, index=False)
    cascade_hashes = {
        community: sha256_file(
            social_root / f"{community}_data" / "empirical_cascades_fixed.jsonl"
        )
        for community in config["communities"]
    }
    manifest = {
        "schema_version": 1,
        "created_at": _utc_now(),
        "status": "retrospective_protocol_frozen",
        "analyst_blind": False,
        "reason_not_blind": "The source corpus and earlier aggregate analyses predate this protocol.",
        "fit_scope": "training split only",
        "test_posts": int(len(test)),
        "test_outcomes_used_for_fit_or_selection": False,
        "seeds": [int(value) for value in config["seeds"]],
        "operators": config["operators"],
        "primary_topology_metrics": list(PRIMARY_SCALAR_METRICS)
        + ["node_depth_distribution"],
        "factorial_cells": FACTORIAL_CELLS,
        "hashes": {
            "config": sha256_file(config_path),
            "source_code": sha256_file(Path(__file__)),
            "splits": sha256_file(splits_path),
            "metrics": sha256_file(metrics_path),
            "frozen_model": sha256_file(model_path),
            "frozen_test_posts": sha256_file(test_path),
            "cascade_sources": cascade_hashes,
        },
    }
    write_json(manifest_path, manifest)
    return manifest


def _real_topology_frame(
    metrics: pd.DataFrame,
    shapes: pd.DataFrame,
    test_ids: set[tuple[str, str]],
) -> tuple[pd.DataFrame, dict[str, list[float]]]:
    test = metrics[
        metrics.apply(lambda row: (str(row["community"]), str(row["post_id"])) in test_ids, axis=1)
    ].copy()
    shape_lookup = shapes.set_index(["community", "post_id"])
    extra_rows: list[dict[str, Any]] = []
    profiles: dict[str, list[float]] = {}
    for row in test.itertuples(index=False):
        shape = shape_lookup.loc[(str(row.community), str(row.post_id))]
        depths = np.asarray(shape["depths"], dtype=float)
        profiles[f"{row.community}|{row.post_id}"] = [float(value) for value in shape["depth_pmf"]]
        extra_rows.append(
            {
                "community": str(row.community),
                "post_id": str(row.post_id),
                "leaf_fraction": float(row.n_leaf_nodes / row.size),
                "depth_variance": float(np.var(depths)),
                "p90_depth": float(np.quantile(depths, 0.90)),
            }
        )
    return test.merge(pd.DataFrame(extra_rows), on=["community", "post_id"]), profiles


def run_topology_experiment(repo_root: Path, config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir = repo_root / config["output_dir"]
    manifest_path = output_dir / "freeze_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError("Freeze the topology design before running outcomes")
    freeze_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model_path = output_dir / "frozen_topology_model.json"
    test_path = output_dir / "frozen_test_posts.csv"
    if sha256_file(model_path) != freeze_manifest["hashes"]["frozen_model"]:
        raise RuntimeError("Frozen topology model hash changed")
    if sha256_file(test_path) != freeze_manifest["hashes"]["frozen_test_posts"]:
        raise RuntimeError("Frozen test-post list hash changed")
    model = json.loads(model_path.read_text(encoding="utf-8"))
    test = pd.read_csv(test_path, dtype={"post_id": str})

    baseline_rows: list[dict[str, Any]] = []
    baseline_profiles: dict[tuple[str, str, int], list[float]] = {}
    for context in test.to_dict(orient="records"):
        for seed in config["seeds"]:
            nodes, metadata = simulate_learned_topology(
                model, context, int(seed), config, behavior=False, ranking_mode="best"
            )
            metrics = extended_topology_metrics(nodes)
            baseline_profiles[(context["community"], context["post_id"], int(seed))] = [
                float(value) / max(1.0, float(sum(metrics["depth_histogram"])))
                for value in metrics["depth_histogram"]
            ]
            baseline_rows.append(
                {
                    "community": context["community"],
                    "post_id": context["post_id"],
                    "seed": int(seed),
                    **metadata,
                    **{key: value for key, value in metrics.items() if key != "depth_histogram"},
                }
            )
    baseline = pd.DataFrame(baseline_rows)
    baseline.to_csv(output_dir / "stage_a_baseline_runs.csv", index=False)
    baseline.to_parquet(output_dir / "stage_a_baseline_runs.parquet", index=False)

    metrics_path = repo_root / config["metrics_path"]
    social_root = repo_root / config["social_root"]
    empirical = pd.read_csv(metrics_path, dtype={"post_id": str})
    shapes = load_cascade_shapes(social_root, empirical, config["communities"])
    test_ids = {(str(row.community), str(row.post_id)) for row in test.itertuples(index=False)}
    real, real_profiles = _real_topology_frame(empirical, shapes, test_ids)
    sim_profiles: dict[str, list[float]] = {}
    for community, post_id in sorted(test_ids):
        profiles = [
            baseline_profiles[(community, post_id, int(seed))] for seed in config["seeds"]
        ]
        length = max(len(values) for values in profiles)
        sim_profiles[f"{community}|{post_id}"] = np.vstack(
            [_pad_pmf(values, length) for values in profiles]
        ).mean(axis=0).tolist()
    topology, topology_summary, strata, communities = summarize_topology_fidelity(
        real,
        baseline,
        real_profiles,
        sim_profiles,
        config,
    )
    topology.to_csv(output_dir / "topology_fidelity_primary.csv", index=False)
    strata.to_csv(output_dir / "topology_fidelity_by_size_stratum.csv", index=False)
    communities.to_csv(output_dir / "topology_fidelity_by_community.csv", index=False)
    write_json(output_dir / "topology_fidelity_summary.json", topology_summary)
    _plot_topology_profiles(topology_summary, output_dir / "topology_profiles.png")

    factorial_rows: list[dict[str, Any]] = []
    baseline_lookup = baseline.set_index(["community", "post_id", "seed"])
    for context in test.to_dict(orient="records"):
        for seed in config["seeds"]:
            for cell, (behavior, ranking_mode) in FACTORIAL_CELLS.items():
                if cell == "B0_P0":
                    row = baseline_lookup.loc[
                        (context["community"], context["post_id"], int(seed))
                    ]
                    factorial_rows.append(
                        {
                            "community": context["community"],
                            "post_id": context["post_id"],
                            "seed": int(seed),
                            "cell": cell,
                            "behavior": "baseline",
                            "ranking": ranking_mode,
                            **row.to_dict(),
                        }
                    )
                    continue
                nodes, metadata = simulate_learned_topology(
                    model,
                    context,
                    int(seed),
                    config,
                    behavior=behavior,
                    ranking_mode=ranking_mode,
                )
                metrics = extended_topology_metrics(nodes)
                factorial_rows.append(
                    {
                        "community": context["community"],
                        "post_id": context["post_id"],
                        "seed": int(seed),
                        "cell": cell,
                        "behavior": "conflict_responsive" if behavior else "baseline",
                        "ranking": ranking_mode,
                        **metadata,
                        **{key: value for key, value in metrics.items() if key != "depth_histogram"},
                    }
                )
    factorial = pd.DataFrame(factorial_rows)
    factorial.to_csv(output_dir / "stage_b_factorial_runs.csv", index=False)
    factorial.to_parquet(output_dir / "stage_b_factorial_runs.parquet", index=False)
    effects, cells, interaction_summary = summarize_learned_interaction(factorial, config)
    effects.to_csv(output_dir / "stage_b_block_effects.csv", index=False)
    cells.to_csv(output_dir / "stage_b_cell_summary.csv", index=False)
    write_json(output_dir / "learned_interaction_summary.json", interaction_summary)
    _write_latex_outputs(output_dir, topology, topology_summary, cells, interaction_summary)

    result = {
        "schema_version": 1,
        "created_at": _utc_now(),
        "status": "complete",
        "freeze_manifest_sha256": sha256_file(manifest_path),
        "stage_a": topology_summary,
        "stage_b": interaction_summary,
        "evidence_scope": {
            "allowed": "Held-out topology fidelity and learned-topology directional replication under frozen operators.",
            "forbidden": "A total human causal effect or a preregistered analyst-blind test.",
        },
    }
    write_json(output_dir / "summary.json", result)
    return result
