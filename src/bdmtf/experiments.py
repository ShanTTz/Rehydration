from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from .data.social_loader import load_comments, load_population, load_posts, resolve_community_paths, select_posts
from .intent_pool import FrozenIntentPool
from .metrics import compute_metrics, summarize_records
from .schema import Intervention, RankingPolicy, SimulationConfig
from .simulator import BDMTFSimulator


def config_from_dict(raw: Dict[str, Any]) -> SimulationConfig:
    sim = raw.get("simulation", raw)
    return SimulationConfig(
        num_agents=int(sim.get("num_agents", 50)),
        steps=int(sim.get("steps", 72)),
        viewport_k=int(sim.get("viewport_k", 5)),
        leader_fraction=float(sim.get("leader_fraction", 0.2)),
        base_impulse=float(sim.get("base_impulse", 0.2)),
        alpha_conflict=float(sim.get("alpha_conflict", 1.8)),
        beta_heat=float(sim.get("beta_heat", 1.0)),
        gamma_consensus=float(sim.get("gamma_consensus", 0.6)),
        polarity_kappa=float(sim.get("polarity_kappa", 5.0)),
        threshold_base=float(sim.get("threshold_base", 0.52)),
        noise_width=float(sim.get("noise_width", 0.1)),
        action_probability=float(sim.get("action_probability", 0.32)),
        action_probability_decay=float(sim.get("action_probability_decay", 0.998)),
        baseline_action_multiplier=float(sim.get("baseline_action_multiplier", 1.0)),
        toxic_action_multiplier=float(sim.get("toxic_action_multiplier", 3.0)),
        toxic_activation_boost=float(sim.get("toxic_activation_boost", 0.25)),
        toxic_reply_bonus=float(sim.get("toxic_reply_bonus", 0.04)),
        early_engagement_median=float(sim.get("early_engagement_median", 100.0)),
        initial_engagement_signal=float(sim.get("initial_engagement_signal", 0.0)),
        external_lambda=float(sim.get("external_lambda", 80.0)),
        external_decay=float(sim.get("external_decay", 0.998)),
        max_external_reactions_per_step=int(sim.get("max_external_reactions_per_step", 8000)),
        enable_external_traffic=bool(sim.get("enable_external_traffic", True)),
        enable_conflict_channel=bool(sim.get("enable_conflict_channel", True)),
        enable_heat_channel=bool(sim.get("enable_heat_channel", True)),
        enable_depth_targeting=bool(sim.get("enable_depth_targeting", True)),
        constructive_depth_lambda=float(sim.get("constructive_depth_lambda", -1.8)),
        antagonistic_depth_lambda=float(sim.get("antagonistic_depth_lambda", 3.5)),
        baseline_depth_lambda_floor=float(sim.get("baseline_depth_lambda_floor", -100.0)),
        toxic_depth_lambda_floor=float(sim.get("toxic_depth_lambda_floor", -100.0)),
        baseline_depth_fatigue_scale=float(sim.get("baseline_depth_fatigue_scale", 6.0)),
        toxic_depth_fatigue_scale=float(sim.get("toxic_depth_fatigue_scale", 6.0)),
        baseline_depth_collapse_start=float(sim.get("baseline_depth_collapse_start", 13.0)),
        toxic_depth_collapse_start=float(sim.get("toxic_depth_collapse_start", 13.0)),
        baseline_depth_collapse_relative_to_max=bool(
            sim.get("baseline_depth_collapse_relative_to_max", False)
        ),
        toxic_depth_collapse_relative_to_max=bool(
            sim.get("toxic_depth_collapse_relative_to_max", False)
        ),
        baseline_depth_collapse_scale=float(sim.get("baseline_depth_collapse_scale", 0.5)),
        toxic_depth_collapse_scale=float(sim.get("toxic_depth_collapse_scale", 0.5)),
        root_target_penalty=float(sim.get("root_target_penalty", 0.005)),
        baseline_root_target_penalty=float(
            sim.get("baseline_root_target_penalty", -1.0)
        ),
        toxic_root_target_penalty=float(sim.get("toxic_root_target_penalty", -1.0)),
        baseline_direct_root_reply_probability=float(
            sim.get("baseline_direct_root_reply_probability", 0.0)
        ),
        toxic_direct_root_reply_probability=float(
            sim.get("toxic_direct_root_reply_probability", 0.0)
        ),
        refresh_visibility_per_action=bool(sim.get("refresh_visibility_per_action", False)),
        visibility_refresh_interval_actions=int(
            sim.get("visibility_refresh_interval_actions", 1)
        ),
        baseline_max_comment_depth=int(sim.get("baseline_max_comment_depth", 0)),
        toxic_max_comment_depth=int(sim.get("toxic_max_comment_depth", 0)),
        baseline_max_comment_depth_jitter=int(
            sim.get("baseline_max_comment_depth_jitter", 0)
        ),
        toxic_max_comment_depth_jitter=int(
            sim.get("toxic_max_comment_depth_jitter", 0)
        ),
        intent_pool_capacity_multiplier=int(
            sim.get("intent_pool_capacity_multiplier", 1)
        ),
        intent_pool_exhaustion_policy=str(
            sim.get("intent_pool_exhaustion_policy", "no_op")
        ),
        semantic_payload_mode=str(sim.get("semantic_payload_mode", "verbatim")),
        community_calibration=dict(raw.get("community_calibration", sim.get("community_calibration", {}))),
    )


def interventions_from_config(raw: Dict[str, Any]) -> List[Intervention]:
    conditions = raw.get("conditions")
    if not conditions:
        conditions = [
            {"name": "baseline_best", "core": "baseline", "ranking": "best", "context": "neutral"},
            {
                "name": "core_toxic_controversial",
                "core": "toxic",
                "ranking": "controversial",
                "context": "neutral",
            },
        ]
    return [Intervention.from_dict(item) for item in conditions]


def run_batch(
    social_root: str | Path,
    output_dir: str | Path,
    communities: Iterable[str],
    posts_per_community: int,
    seeds: Iterable[int],
    config: SimulationConfig,
    interventions: Iterable[Intervention],
    seed_mode: str = "condition_specific",
    resume: bool = False,
) -> List[Dict[str, Any]]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    runs_path = output / "runs.jsonl"
    records: List[Dict[str, Any]] = []
    completed: set[tuple[str, str, int, str]] = set()
    if resume and runs_path.exists():
        with runs_path.open("r", encoding="utf-8") as existing:
            for line in existing:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = (
                    str(record["community"]),
                    str(record["post_id"]),
                    int(record["seed"]),
                    str(record["condition"]),
                )
                if key in completed:
                    continue
                completed.add(key)
                records.append(record)

    mode = "a" if resume and runs_path.exists() else "w"
    with runs_path.open(mode, encoding="utf-8") as handle:
        for community in communities:
            paths = resolve_community_paths(social_root, community)
            posts_df = load_posts(paths, enriched=True)
            comments_df = load_comments(paths)
            selected_posts = select_posts(posts_df, posts_per_community, seed=0)
            community_config = _with_community_calibration(config, community)
            frozen_path = paths.base_dir / "frozen_intents.jsonl"
            intent_pool = (
                FrozenIntentPool.from_jsonl(frozen_path)
                if frozen_path.exists()
                else FrozenIntentPool.from_comments(comments_df, seed=0)
            )

            for seed in seeds:
                agents = load_population(paths, community_config, seed=int(seed))
                for post in selected_posts.itertuples(index=False):
                    title = str(getattr(post, "title", ""))
                    post_id = str(getattr(post, "post_id"))
                    for intervention in interventions:
                        key = (community, post_id, int(seed), intervention.name)
                        if key in completed:
                            continue
                        sim_seed = simulation_seed(
                            community,
                            post_id,
                            intervention.name,
                            int(seed),
                            seed_mode=seed_mode,
                        )
                        run_config = replace(
                            community_config,
                            initial_engagement_signal=_post_early_engagement_signal(post),
                        )
                        sim = BDMTFSimulator(agents, intent_pool, config=run_config, seed=sim_seed)
                        state, traces = sim.run(
                            post_id=post_id,
                            title=title,
                            initial_text=str(getattr(post, "full_text", "")),
                            intervention=intervention,
                        )
                        metrics = compute_metrics(state, traces)
                        record: Dict[str, Any] = {
                            "community": community,
                            "post_id": post_id,
                            "seed": int(seed),
                            "condition": intervention.name,
                            "core": intervention.core,
                            "ranking": intervention.ranking.value,
                            "context": intervention.context,
                            "intent_pool_capacity_multiplier": int(
                                run_config.intent_pool_capacity_multiplier
                            ),
                            "intent_pool_exhaustion_policy": str(
                                run_config.intent_pool_exhaustion_policy
                            ),
                            "semantic_payload_mode": str(
                                run_config.semantic_payload_mode
                            ),
                        }
                        record.update(metrics)
                        records.append(record)
                        completed.add(key)
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    write_summary(records, output)
    return records


def _stable_seed(*parts: object) -> int:
    raw = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]
    return int(digest, 16)


def simulation_seed(
    community: str,
    post_id: str,
    condition: str,
    seed: int,
    seed_mode: str = "condition_specific",
) -> int:
    if seed_mode == "condition_specific":
        return _stable_seed(community, post_id, condition, seed)
    if seed_mode == "paired_by_post_seed":
        return _stable_seed(community, post_id, seed)
    raise ValueError(
        f"Unknown seed_mode {seed_mode!r}; expected 'condition_specific' "
        "or 'paired_by_post_seed'"
    )


def _with_community_calibration(config: SimulationConfig, community: str) -> SimulationConfig:
    raw = config.community_calibration.get(community) or config.community_calibration.get(community.lower())
    if not raw:
        return config
    updates: Dict[str, Any] = {}
    for key in (
        "threshold_base",
        "early_engagement_median",
        "external_lambda",
        "action_probability",
        "toxic_action_multiplier",
        "toxic_activation_boost",
        "toxic_reply_bonus",
    ):
        if key in raw:
            updates[key] = float(raw[key])
    return replace(config, **updates) if updates else config


def _post_early_engagement_signal(post: object) -> float:
    total = 0.0
    found = False
    for name in ("early_score_sum", "early_num_comments"):
        value = getattr(post, name, None)
        try:
            if value is not None and float(value) > 0:
                total += float(value)
                found = True
        except (TypeError, ValueError):
            pass
    if found:
        return max(0.0, total)
    value = getattr(post, "final_num_comments", None)
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return 0.0


def write_summary(records: List[Dict[str, Any]], output: Path) -> None:
    summary = summarize_records(records)
    summary_path = output / "summary.csv"
    if summary:
        fieldnames = ["condition"] + sorted(next(iter(summary.values())).keys())
        with summary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for condition, values in sorted(summary.items()):
                writer.writerow({"condition": condition, **values})

    volume_depth_path = output / "volume_depth.csv"
    with volume_depth_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["community", "post_id", "seed", "condition", "comment_volume", "mean_leaf_depth", "max_depth"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow({name: row.get(name) for name in fieldnames})
