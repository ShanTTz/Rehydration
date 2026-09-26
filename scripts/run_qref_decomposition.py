"""Decompose frozen-catalog and semantic-to-policy mapping sensitivity.

The experiment uses the existing three-family cache and makes no API calls.
It crosses a pooled versus family-specific frozen catalog with a pooled versus
family-specific, outcome-blind response-coordinate mapping.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bdmtf.data.social_loader import load_population, load_posts, resolve_community_paths
from bdmtf.experiments import (
    _post_early_engagement_signal,
    _stable_seed,
    _with_community_calibration,
    config_from_dict,
    interventions_from_config,
)
from bdmtf.features import estimate_toxicity
from bdmtf.intent_pool import FrozenIntentPool
from bdmtf.metrics import compute_metrics
from bdmtf.schema import Intent
from bdmtf.simulator import BDMTFSimulator


FACTORIAL_CELLS = {
    "B0_P0": {"name": "BASELINE", "core": "baseline", "ranking": "best", "context": "neutral"},
    "B0_P1": {"name": "BASELINE_CONTROVERSIAL", "core": "baseline", "ranking": "controversial", "context": "neutral"},
    "B1_P0": {"name": "CORE_TOXIC_BEST", "core": "toxic", "ranking": "best", "context": "neutral"},
    "B1_P1": {"name": "CORE_TOXIC_CONTROVERSIAL", "core": "toxic", "ranking": "controversial", "context": "neutral"},
}
CONDITION_TO_CELL = {value["name"]: key for key, value in FACTORIAL_CELLS.items()}
VARIANTS = (
    "pooled_pool_pooled_map",
    "family_pool_pooled_map",
    "pooled_pool_family_map",
    "family_pool_family_map",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_catalog(path: Path):
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows: list[dict[str, Any]] = []
    task_families: defaultdict[str, set[str]] = defaultdict(set)
    for record in records:
        response = json.loads(str(record["response"]))
        family = str(record["family"]).lower()
        community = str(record["community"]).lower()
        task_id = str(record["task_id"])
        task_families[task_id].add(family)
        rows.append(
            {
                "task_id": task_id,
                "family": family,
                "model": str(record["model"]),
                "community": community,
                "post_id": str(record["post_id"]),
                "leader": int(record["leader"]),
                "action": str(response.get("action", "abstain")).lower(),
                "polarity": str(response.get("polarity", "neutral")).lower(),
                "content": str(response.get("content", "")).strip(),
            }
        )
    frame = pd.DataFrame(rows)
    families = sorted(frame["family"].unique())
    if any(value != set(families) for value in task_families.values()):
        raise ValueError("Every task must contain all declared model families")
    if frame.duplicated(["task_id", "family"]).any():
        raise ValueError("Duplicate task-family response")

    pools: dict[tuple[str, str], FrozenIntentPool] = {}
    profiles: list[dict[str, Any]] = []
    for (family, community), group in frame.groupby(["family", "community"], sort=True):
        replies = group[(group["action"] == "reply") & (group["content"] != "")]
        intents = [
            Intent(
                intent_id=f"qref:{family}:{row.task_id}",
                agent_id=None,
                polarity="antagonistic" if row.polarity == "antagonistic" else "supportive",
                content=row.content,
                metadata={
                    "source": "cached_multimodel_qref",
                    "family": family,
                    "model": row.model,
                    "toxicity": estimate_toxicity(row.content),
                },
            )
            for row in replies.itertuples(index=False)
        ]
        if not intents:
            raise ValueError(f"No usable replies for {family}/{community}")
        pools[(family, community)] = FrozenIntentPool(intents)
        antagonistic = sum(intent.polarity == "antagonistic" for intent in intents)
        profiles.append(
            {
                "family": family,
                "community": community,
                "tasks": int(len(group)),
                "replies": int(len(replies)),
                "reply_rate": float(len(replies) / len(group)),
                "antagonistic_share": float(antagonistic / len(intents)),
                "templates": int(len(intents)),
            }
        )
    profile = pd.DataFrame(profiles)
    for community, group in profile.groupby("community", sort=True):
        pooled_tasks = float(group["tasks"].sum())
        pooled_replies = float(group["replies"].sum())
        pooled_antagonistic = sum(
            pools[(row.family, community)].unique_intents()[index].polarity
            == "antagonistic"
            for row in group.itertuples(index=False)
            for index in range(len(pools[(row.family, community)].unique_intents()))
        )
        mask = profile["community"] == community
        profile.loc[mask, "pooled_reply_rate"] = pooled_replies / pooled_tasks
        profile.loc[mask, "pooled_antagonistic_share"] = (
            pooled_antagonistic / pooled_replies
        )
        pooled_intents = [
            intent
            for family in families
            for intent in pools[(family, community)].unique_intents()
        ]
        pools[("pooled", community)] = FrozenIntentPool(pooled_intents)
    profile["reply_scale"] = (
        profile["reply_rate"] / profile["pooled_reply_rate"]
    ).clip(0.5, 1.5)
    profile["conflict_scale"] = (
        profile["antagonistic_share"] / profile["pooled_antagonistic_share"]
    ).clip(0.5, 1.5)
    posts = {
        community: sorted(group["post_id"].unique())
        for community, group in frame.groupby("community", sort=True)
    }
    return pools, profile, posts, families


def variant_specs(families: list[str]):
    yield "pooled_pool_pooled_map", "pooled", "pooled"
    for family in families:
        yield "family_pool_pooled_map", family, "pooled"
    for family in families:
        yield "pooled_pool_family_map", "pooled", family
    for family in families:
        yield "family_pool_family_map", family, family


def run_community(
    community_key: str,
    config_path: Path,
    intents_path: Path,
    output: Path,
    posts_per_community: int,
    seeds: list[int],
    capacity: int,
) -> dict[str, Any]:
    pools, profile, post_ids, families = load_catalog(intents_path)
    source = json.loads(config_path.read_text(encoding="utf-8"))
    source["conditions"] = list(FACTORIAL_CELLS.values())
    canonical = {str(value).lower(): str(value) for value in source["communities"]}
    community = canonical[community_key]
    paths = resolve_community_paths(ROOT / "data" / "social_paper", community)
    posts = load_posts(paths, enriched=True)
    selected_ids = post_ids[community_key][:posts_per_community]
    posts = posts[posts["post_id"].astype(str).isin(selected_ids)].copy()
    posts["post_id"] = posts["post_id"].astype(str)
    posts = posts.set_index("post_id").loc[selected_ids].reset_index()
    base = _with_community_calibration(config_from_dict(source), community)
    interventions = interventions_from_config(source)
    profile_lookup = {
        str(row.family): row
        for row in profile[profile["community"] == community_key].itertuples(index=False)
    }
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{community_key}.jsonl"
    done: set[tuple[str, str, str, int, str]] = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                done.add(
                    (
                        record["variant"],
                        record["catalog_family"],
                        record["mapping_family"],
                        int(record["seed"]),
                        record["post_id"] + "|" + record["condition"],
                    )
                )
    new_runs = 0
    with path.open("a", encoding="utf-8") as stream:
        for seed in seeds:
            for variant, catalog_family, mapping_family in variant_specs(families):
                mapping_row = None if mapping_family == "pooled" else profile_lookup[mapping_family]
                run_config = replace(
                    base,
                    action_probability=(
                        base.action_probability
                        if mapping_row is None
                        else min(1.0, base.action_probability * float(mapping_row.reply_scale))
                    ),
                    alpha_conflict=(
                        base.alpha_conflict
                        if mapping_row is None
                        else base.alpha_conflict * float(mapping_row.conflict_scale)
                    ),
                    polarity_kappa=(
                        base.polarity_kappa
                        if mapping_row is None
                        else base.polarity_kappa * float(mapping_row.conflict_scale)
                    ),
                    intent_pool_capacity_multiplier=capacity,
                    intent_pool_exhaustion_policy="no_op",
                )
                agents = load_population(paths, run_config, seed=seed)
                for post in posts.itertuples(index=False):
                    post_id = str(post.post_id)
                    sim_seed = _stable_seed("qref_decomposition", community, post_id, seed)
                    post_config = replace(
                        run_config,
                        initial_engagement_signal=_post_early_engagement_signal(post),
                    )
                    for intervention in interventions:
                        key = (
                            variant,
                            catalog_family,
                            mapping_family,
                            seed,
                            post_id + "|" + intervention.name,
                        )
                        if key in done:
                            continue
                        simulator = BDMTFSimulator(
                            agents,
                            pools[(catalog_family, community_key)],
                            config=post_config,
                            seed=sim_seed,
                        )
                        state, traces = simulator.run(
                            post_id=post_id,
                            title=str(getattr(post, "title", "")),
                            initial_text=str(getattr(post, "full_text", "")),
                            intervention=intervention,
                        )
                        record = {
                            "variant": variant,
                            "catalog_family": catalog_family,
                            "mapping_family": mapping_family,
                            "community": community,
                            "post_id": post_id,
                            "seed": seed,
                            "condition": intervention.name,
                            "cell": CONDITION_TO_CELL[intervention.name],
                            "reply_scale": 1.0 if mapping_row is None else float(mapping_row.reply_scale),
                            "conflict_scale": 1.0 if mapping_row is None else float(mapping_row.conflict_scale),
                            "capacity_multiplier": capacity,
                            **compute_metrics(state, traces),
                        }
                        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                        stream.flush()
                        done.add(key)
                        new_runs += 1
    return {"community": community, "new_runs": new_runs, "total_runs": len(done), "path": str(path)}


def factorial_blocks(runs: pd.DataFrame) -> pd.DataFrame:
    keys = ["variant", "catalog_family", "mapping_family", "community", "post_id", "seed"]
    rows: list[dict[str, Any]] = []
    for key, group in runs.groupby(keys, sort=True):
        cells = group.set_index("cell")
        if not set(FACTORIAL_CELLS).issubset(cells.index):
            continue
        logv = {cell: math.log1p(float(cells.loc[cell, "comment_volume"])) for cell in FACTORIAL_CELLS}
        depth = {cell: float(cells.loc[cell, "mean_leaf_depth"]) for cell in FACTORIAL_CELLS}
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "volume_log_interaction": logv["B1_P1"] - logv["B1_P0"] - logv["B0_P1"] + logv["B0_P0"],
                "depth_interaction": depth["B1_P1"] - depth["B1_P0"] - depth["B0_P1"] + depth["B0_P0"],
            }
        )
    return pd.DataFrame(rows)


def bootstrap_summary(blocks: pd.DataFrame, draws: int) -> pd.DataFrame:
    rows = []
    group_keys = ["variant", "catalog_family", "mapping_family"]
    for index, (key, group) in enumerate(blocks.groupby(group_keys, sort=True)):
        post = group.groupby(["community", "post_id"], as_index=False)[["volume_log_interaction", "depth_interaction"]].mean()
        rng = np.random.default_rng(20270908 + index)
        samples = np.empty((draws, 2))
        strata = list(post.groupby("community", sort=True))
        for draw in range(draws):
            sampled = pd.concat([g.iloc[rng.integers(0, len(g), len(g))] for _, g in strata], ignore_index=True)
            samples[draw] = sampled[["volume_log_interaction", "depth_interaction"]].mean().to_numpy()
        point = post[["volume_log_interaction", "depth_interaction"]].mean()
        low = np.quantile(samples, 0.025, axis=0)
        high = np.quantile(samples, 0.975, axis=0)
        rows.append(
            {
                **dict(zip(group_keys, key, strict=True)),
                "posts": len(post),
                "seeds": int(group["seed"].nunique()),
                "volume_ror": math.exp(float(point.iloc[0])),
                "volume_ror_ci_low": math.exp(float(low[0])),
                "volume_ror_ci_high": math.exp(float(high[0])),
                "depth_did": float(point.iloc[1]),
                "depth_did_ci_low": float(low[1]),
                "depth_did_ci_high": float(high[1]),
            }
        )
    return pd.DataFrame(rows)


def decomposition(blocks: pd.DataFrame, draws: int) -> pd.DataFrame:
    post = blocks.groupby(["variant", "catalog_family", "mapping_family", "community", "post_id"], as_index=False)[["volume_log_interaction", "depth_interaction"]].mean()
    base = post[post["variant"] == "pooled_pool_pooled_map"].drop(columns=["variant", "catalog_family", "mapping_family"])
    rows = []
    for family in sorted(set(post["catalog_family"]) - {"pooled"}):
        content = post[(post["variant"] == "family_pool_pooled_map") & (post["catalog_family"] == family)]
        mapping = post[(post["variant"] == "pooled_pool_family_map") & (post["mapping_family"] == family)]
        joint = post[(post["variant"] == "family_pool_family_map") & (post["catalog_family"] == family)]
        merged = base.merge(content, on=["community", "post_id"], suffixes=("_a", "_b"), validate="one_to_one")
        merged = merged.merge(mapping[["community", "post_id", "volume_log_interaction", "depth_interaction"]], on=["community", "post_id"], validate="one_to_one").rename(columns={"volume_log_interaction": "volume_log_interaction_c", "depth_interaction": "depth_interaction_c"})
        merged = merged.merge(joint[["community", "post_id", "volume_log_interaction", "depth_interaction"]], on=["community", "post_id"], validate="one_to_one").rename(columns={"volume_log_interaction": "volume_log_interaction_d", "depth_interaction": "depth_interaction_d"})
        metrics = {
            "content_only": (merged.volume_log_interaction_b - merged.volume_log_interaction_a, merged.depth_interaction_b - merged.depth_interaction_a),
            "mapping_only": (merged.volume_log_interaction_c - merged.volume_log_interaction_a, merged.depth_interaction_c - merged.depth_interaction_a),
            "nonadditive": (merged.volume_log_interaction_d - merged.volume_log_interaction_b - merged.volume_log_interaction_c + merged.volume_log_interaction_a, merged.depth_interaction_d - merged.depth_interaction_b - merged.depth_interaction_c + merged.depth_interaction_a),
        }
        rng = np.random.default_rng(int(hashlib.sha256(family.encode()).hexdigest()[:8], 16))
        strata = list(merged.groupby("community", sort=True))
        for effect, (volume_values, depth_values) in metrics.items():
            temp = merged[["community", "post_id"]].copy()
            temp["volume"] = volume_values
            temp["depth"] = depth_values
            temp_strata = list(temp.groupby("community", sort=True))
            samples = np.empty((draws, 2))
            for draw in range(draws):
                sampled = pd.concat([g.iloc[rng.integers(0, len(g), len(g))] for _, g in temp_strata], ignore_index=True)
                samples[draw] = sampled[["volume", "depth"]].mean().to_numpy()
            point = temp[["volume", "depth"]].mean()
            low, high = np.quantile(samples, [0.025, 0.975], axis=0)
            rows.append({"family": family, "effect": effect, "log_volume_difference": float(point.volume), "log_volume_ci_low": float(low[0]), "log_volume_ci_high": float(high[0]), "depth_difference": float(point.depth), "depth_ci_low": float(low[1]), "depth_ci_high": float(high[1])})
    return pd.DataFrame(rows)


def write_latex(summary: pd.DataFrame, path: Path) -> None:
    labels = {
        "pooled_pool_pooled_map": "Pooled pool / pooled map",
        "family_pool_pooled_map": "Family pool / pooled map",
        "pooled_pool_family_map": "Pooled pool / family map",
        "family_pool_family_map": "Family pool / family map",
    }
    lines = [r"\begin{tabular}{lllrr}", r"\toprule", r"Pool / map & Family & $N$ & Volume RoR [95\% CI] & Depth DiD [95\% CI] \\", r"\midrule"]
    for row in summary.itertuples(index=False):
        family = "--" if row.catalog_family == "pooled" and row.mapping_family == "pooled" else (row.catalog_family if row.catalog_family != "pooled" else row.mapping_family).title()
        lines.append(
            f"{labels[row.variant]} & {family} & {row.posts} & "
            f"{row.volume_ror:.3f} [{row.volume_ror_ci_low:.3f}, "
            f"{row.volume_ror_ci_high:.3f}] & {row.depth_did:.3f} "
            f"[{row.depth_did_ci_low:.3f}, {row.depth_did_ci_high:.3f}] \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "paper_exact_reproduction.json")
    parser.add_argument("--intents", type=Path, default=ROOT / "artifacts" / "api" / "frozen_intents_multimodel.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "reviewer_validation" / "qref_decomposition_20260908")
    parser.add_argument("--paper-generated", type=Path, default=ROOT / "manuscript" / "iclr2027_overleaf_package_review_fixed_20260908" / "generated")
    parser.add_argument("--posts-per-community", type=int, default=10)
    parser.add_argument("--seeds", default="0,1")
    parser.add_argument("--capacity", type=int, default=512)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=5000)
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    _, profile, posts, families = load_catalog(args.intents)
    communities = sorted(posts)
    expected = sum(min(args.posts_per_community, len(posts[c])) for c in communities) * len(seeds) * 4 * (1 + 3 * len(families))
    args.output.mkdir(parents=True, exist_ok=True)
    status_path = args.output / "run_status.json"
    status_path.write_text(json.dumps({"status": "running", "expected_runs": expected, "started_at": time.time()}, indent=2), encoding="utf-8")
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_community, community, args.config, args.intents, args.output / "workers", args.posts_per_community, seeds, args.capacity) for community in communities]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result), flush=True)
    records = [json.loads(line) for path in sorted((args.output / "workers").glob("*.jsonl")) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    runs = pd.DataFrame(records)
    if len(runs) != expected or runs.duplicated(["variant", "catalog_family", "mapping_family", "community", "post_id", "seed", "condition"]).any():
        raise ValueError(f"Expected {expected} unique runs, found {len(runs)}")
    blocks = factorial_blocks(runs)
    summary = bootstrap_summary(blocks, args.bootstrap)
    decomposition_frame = decomposition(blocks, args.bootstrap)
    runs.to_parquet(args.output / "runs.parquet", index=False)
    blocks.to_csv(args.output / "block_interactions.csv", index=False)
    summary.to_csv(args.output / "effect_summary.csv", index=False)
    decomposition_frame.to_csv(args.output / "decomposition.csv", index=False)
    table_path = args.paper_generated / "table_qref_decomposition.tex"
    write_latex(summary, table_path)
    manifest = {
        "status": "complete",
        "expected_runs": expected,
        "observed_runs": len(runs),
        "posts": int(runs[["community", "post_id"]].drop_duplicates().shape[0]),
        "seeds": seeds,
        "variants": list(VARIANTS),
        "families": families,
        "capacity": args.capacity,
        "api_calls": 0,
        "inputs": {str(args.config): sha256(args.config), str(args.intents): sha256(args.intents)},
        "outputs": {"runs.parquet": sha256(args.output / "runs.parquet"), "effect_summary.csv": sha256(args.output / "effect_summary.csv"), "decomposition.csv": sha256(args.output / "decomposition.csv")},
        "interpretation": "Pool and semantic-to-policy mapping axes are crossed independently; all policy outcomes remain unopened during mapping construction.",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    status_path.write_text(json.dumps({"status": "complete", "expected_runs": expected, "observed_runs": len(runs)}, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(decomposition_frame.to_string(index=False))


if __name__ == "__main__":
    main()
