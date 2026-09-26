from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from bdmtf.schema import AgentProfile, SimulationConfig


REDDIT_PREFIX_RE = re.compile(r"^t[13]_")


@dataclass(frozen=True)
class CommunityPaths:
    community: str
    base_dir: Path
    posts_csv: Path
    comments_csv: Path
    authors_csv: Path
    enriched_csv: Optional[Path]
    calibration_json: Optional[Path]
    population_json: Optional[Path]


def resolve_community_paths(social_root: str | Path, community: str) -> CommunityPaths:
    root = Path(social_root)
    base = root / f"{community}_data"
    enriched = base / f"posts_features_final_enriched_safe_{community}.csv"
    if not enriched.exists():
        enriched = None
    calibration = base / "calibration_summary.json"
    population = base / "population.json"
    return CommunityPaths(
        community=community,
        base_dir=base,
        posts_csv=base / f"posts_features_{community}.csv",
        comments_csv=base / f"comments_data_{community}.csv",
        authors_csv=base / f"authors_history_{community}.csv",
        enriched_csv=enriched,
        calibration_json=calibration if calibration.exists() else None,
        population_json=population if population.exists() else None,
    )


def strip_reddit_prefix(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return REDDIT_PREFIX_RE.sub("", str(value))


def load_posts(paths: CommunityPaths, enriched: bool = True) -> pd.DataFrame:
    target = paths.enriched_csv if enriched and paths.enriched_csv else paths.posts_csv
    df = pd.read_csv(target, encoding="utf-8", encoding_errors="replace")
    df["post_id"] = df["post_id"].astype(str)
    return df


def load_comments(paths: CommunityPaths) -> pd.DataFrame:
    df = pd.read_csv(paths.comments_csv, encoding="utf-8", encoding_errors="replace")
    df["post_id"] = df["post_id"].astype(str)
    df["comment_id"] = df["comment_id"].map(strip_reddit_prefix)
    df["parent_id_raw"] = df["parent_id"].astype(str)
    df["parent_id"] = df["parent_id"].map(strip_reddit_prefix)
    if "created_utc" in df.columns:
        df["created_utc"] = pd.to_datetime(df["created_utc"], utc=True, errors="coerce")
    return df


def load_population(paths: CommunityPaths, config: SimulationConfig, seed: int = 0) -> List[AgentProfile]:
    # The paper initializes each run by sampling behavioral drives from the
    # empirical community distribution. population.json contains semantic
    # personas for OASIS/frozen intents and must not override that calibration.
    if paths.calibration_json and paths.calibration_json.exists():
        with paths.calibration_json.open("r", encoding="utf-8") as handle:
            calibration = json.load(handle)
        return sample_agents_from_calibration(calibration, config, seed=seed)

    if paths.population_json and paths.population_json.exists():
        with paths.population_json.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        agents = []
        for idx, item in enumerate(raw[: config.num_agents]):
            scores = item.get("dark_tetrad_scores", {})
            agents.append(
                AgentProfile.from_dark_tetrad(
                    agent_id=idx,
                    scores=scores,
                    threshold_base=config.threshold_base,
                    is_leader=idx < int(config.num_agents * config.leader_fraction),
                )
            )
        if len(agents) >= config.num_agents:
            return agents

    return default_agents(config)


def sample_agents_from_calibration(
    calibration: Dict[str, Any],
    config: SimulationConfig,
    seed: int = 0,
) -> List[AgentProfile]:
    rng = np.random.default_rng(seed)
    dists = calibration.get("personality_distribution", {})
    traits = ["machiavellianism", "narcissism", "psychopathy", "sadism"]
    agents: List[AgentProfile] = []
    for idx in range(config.num_agents):
        scores = {}
        for trait in traits:
            dist = dists.get(trait) or {"1": 0.4, "2": 0.4, "3": 0.15, "4": 0.05, "5": 0.0}
            values = np.array([int(k) for k in sorted(dist, key=lambda x: int(x))])
            probs = np.array([float(dist[str(v)]) for v in values], dtype=float)
            probs = probs / probs.sum()
            scores[trait] = int(rng.choice(values, p=probs))
        agents.append(
            AgentProfile.from_dark_tetrad(
                agent_id=idx,
                scores=scores,
                threshold_base=config.threshold_base,
                is_leader=idx < int(config.num_agents * config.leader_fraction),
            )
        )
    return agents


def default_agents(config: SimulationConfig) -> List[AgentProfile]:
    agents = []
    for idx in range(config.num_agents):
        score = 2 + (idx % 3 == 0)
        agents.append(
            AgentProfile.from_dark_tetrad(
                idx,
                {
                    "machiavellianism": score,
                    "narcissism": 2,
                    "psychopathy": score,
                    "sadism": 2,
                },
                threshold_base=config.threshold_base,
                is_leader=idx < int(config.num_agents * config.leader_fraction),
            )
        )
    return agents


def extract_cascade_features(comments_df: pd.DataFrame, post_id: str) -> Optional[Dict[str, Any]]:
    post_comments = comments_df[comments_df["post_id"].astype(str) == str(post_id)].copy()
    if post_comments.empty:
        return None
    if "created_utc" in post_comments.columns:
        post_comments = post_comments.sort_values("created_utc")

    ids = set(post_comments["comment_id"].astype(str))
    depth: Dict[str, int] = {}
    children: Dict[str, List[str]] = {cid: [] for cid in ids}
    scores: Dict[str, float] = {}

    for row in post_comments.itertuples(index=False):
        cid = str(getattr(row, "comment_id"))
        parent = strip_reddit_prefix(getattr(row, "parent_id", ""))
        scores[cid] = float(getattr(row, "score", 0) or 0)
        if parent in ids and parent in depth:
            depth[cid] = depth[parent] + 1
            children[parent].append(cid)
        else:
            depth[cid] = int(getattr(row, "depth", 0) or 0) + 1 if parent in ids else 1

    depths = list(depth.values())
    degrees = [len(children[cid]) for cid in ids]
    leaves = [cid for cid in ids if len(children[cid]) == 0]
    leaf_depths = [depth[cid] for cid in leaves]
    return {
        "post_id": str(post_id),
        "size": len(ids),
        "max_depth": int(max(depths)),
        "mean_depth": float(np.mean(depths)),
        "mean_leaf_depth": float(np.mean(leaf_depths)) if leaf_depths else 0.0,
        "depth_variance": float(np.var(depths)),
        "degrees": degrees,
        "depths": depths,
        "n_leaf_nodes": len(leaves),
        "score_variance": float(np.var(list(scores.values()))) if scores else 0.0,
    }


def select_posts(posts_df: pd.DataFrame, limit: int, seed: int = 0) -> pd.DataFrame:
    if limit <= 0 or len(posts_df) <= limit:
        return posts_df.copy()
    if "is_viral" in posts_df.columns:
        viral = posts_df[posts_df["is_viral"] == 1]
        nonviral = posts_df[posts_df["is_viral"] != 1]
        half = limit // 2
        parts = []
        if not viral.empty:
            parts.append(viral.sample(n=min(half, len(viral)), random_state=seed))
        remaining = limit - sum(len(p) for p in parts)
        if not nonviral.empty and remaining > 0:
            parts.append(nonviral.sample(n=min(remaining, len(nonviral)), random_state=seed + 1))
        if parts:
            return pd.concat(parts).head(limit)
    return posts_df.sample(n=limit, random_state=seed)
