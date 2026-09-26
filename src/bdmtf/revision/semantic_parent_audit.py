from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from bdmtf.data.social_loader import (
    load_comments,
    load_population,
    load_posts,
    resolve_community_paths,
    select_posts,
)
from bdmtf.experiments import (
    _post_early_engagement_signal,
    _with_community_calibration,
    config_from_dict,
    interventions_from_config,
    simulation_seed,
)
from bdmtf.intent_pool import FrozenIntentPool
from bdmtf.revision.provenance import sha256_file, write_json
from bdmtf.schema import Intent
from bdmtf.semantic_intent import render_frozen_intent
from bdmtf.simulator import BDMTFSimulator


def _depth_bin(depth: int) -> str:
    if depth <= 1:
        return "root_reply"
    if depth <= 3:
        return "shallow_reply"
    return "deep_reply"


def _stable_id(*parts: object) -> str:
    payload = "::".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def _clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)


def _collect_candidates(
    project: Path,
    raw: Mapping[str, Any],
    social_root: Path,
) -> pd.DataFrame:
    config = replace(
        config_from_dict(dict(raw)),
        semantic_payload_mode="verbatim",
        intent_pool_capacity_multiplier=int(raw.get("audit_capacity_multiplier", 256)),
        intent_pool_exhaustion_policy="no_op",
    )
    interventions = interventions_from_config(dict(raw))
    rows: list[dict[str, Any]] = []
    for community in raw["communities"]:
        paths = resolve_community_paths(social_root, str(community))
        posts = select_posts(
            load_posts(paths, enriched=True),
            int(raw["posts_per_community"]),
            seed=0,
        )
        comments = load_comments(paths)
        frozen_path = paths.base_dir / "frozen_intents.jsonl"
        pool = (
            FrozenIntentPool.from_jsonl(frozen_path)
            if frozen_path.exists()
            else FrozenIntentPool.from_comments(comments, seed=0)
        )
        community_config = _with_community_calibration(config, str(community))
        for seed in raw["seeds"]:
            agents = load_population(paths, community_config, seed=int(seed))
            for post in posts.itertuples(index=False):
                post_id = str(getattr(post, "post_id"))
                title = _clean_text(getattr(post, "title", ""))
                for intervention in interventions:
                    run_config = replace(
                        community_config,
                        initial_engagement_signal=_post_early_engagement_signal(post),
                    )
                    simulator = BDMTFSimulator(
                        agents,
                        pool,
                        config=run_config,
                        seed=simulation_seed(
                            str(community),
                            post_id,
                            intervention.name,
                            int(seed),
                            seed_mode=str(
                                raw.get("randomization", {}).get(
                                    "seed_mode", "paired_by_post_seed"
                                )
                            ),
                        ),
                    )
                    state, _ = simulator.run(
                        post_id,
                        title,
                        intervention,
                        initial_text=_clean_text(getattr(post, "full_text", "")),
                    )
                    for node in state.comments.values():
                        if "frozen_payload" not in node.metadata or not node.parent_id:
                            continue
                        parent = state.comments.get(node.parent_id)
                        if parent is None:
                            continue
                        frozen_payload = str(node.metadata["frozen_payload"])
                        frozen_intent = Intent(
                            intent_id=str(node.metadata.get("intent_id", "")),
                            agent_id=node.author_id,
                            polarity=str(node.metadata.get("polarity", "supportive")),
                            content=frozen_payload,
                            metadata={"toxicity": node.toxicity},
                        )
                        rendered = render_frozen_intent(
                            frozen_intent,
                            parent,
                            title,
                        )
                        rows.append(
                            {
                                "community": str(community),
                                "post_id": post_id,
                                "seed": int(seed),
                                "condition": intervention.name,
                                "core": intervention.core,
                                "ranking": intervention.ranking.value,
                                "node_id": node.node_id,
                                "parent_id": parent.node_id,
                                "reply_depth": int(node.depth),
                                "depth_bin": _depth_bin(int(node.depth)),
                                "parent_text": parent.content[:1200],
                                "verbatim_reply": frozen_payload[:1200],
                                "rendered_reply": rendered.text[:1200],
                                "polarity": str(node.metadata.get("polarity", "")),
                                "intent_id": str(node.metadata.get("intent_id", "")),
                                "source_topic_overlap": rendered.source_topic_overlap,
                            }
                        )
    return pd.DataFrame(rows)


def _stratified_sample(
    candidates: pd.DataFrame,
    sample_size: int,
    seed: int,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    strata = ["community", "condition", "depth_bin"]
    rng = np.random.default_rng(seed)
    ranked = candidates.copy()
    ranked["_priority"] = rng.random(len(ranked))
    groups = list(ranked.groupby(strata, sort=True))
    per_group = max(1, int(np.ceil(sample_size / max(len(groups), 1))))
    selected_indices: list[int] = []
    for _, group in groups:
        selected_indices.extend(
            group.nsmallest(per_group, "_priority").index.tolist()
        )
    selected_indices = selected_indices[:sample_size]
    if len(selected_indices) < min(sample_size, len(ranked)):
        remaining = ranked.drop(index=selected_indices).nsmallest(
            min(sample_size, len(ranked)) - len(selected_indices),
            "_priority",
        )
        selected_indices.extend(remaining.index.tolist())
    return ranked.loc[selected_indices].drop(columns=["_priority"]).reset_index(drop=True)


def _blind_items(
    sample: pd.DataFrame,
    seed: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rng = np.random.default_rng(seed ^ 0x53454D41)
    public_rows: list[dict[str, Any]] = []
    key: list[dict[str, Any]] = []
    for row in sample.itertuples(index=False):
        item_id = _stable_id(
            row.community,
            row.post_id,
            row.seed,
            row.condition,
            row.node_id,
        )
        rendered_is_a = bool(rng.integers(0, 2))
        public_rows.append(
            {
                "item_id": item_id,
                "community": row.community,
                "condition": row.condition,
                "core": row.core,
                "ranking": row.ranking,
                "reply_depth": row.reply_depth,
                "depth_bin": row.depth_bin,
                "parent_text": row.parent_text,
                "candidate_a": (
                    row.rendered_reply if rendered_is_a else row.verbatim_reply
                ),
                "candidate_b": (
                    row.verbatim_reply if rendered_is_a else row.rendered_reply
                ),
                "questions": {
                    "relevance_a": "How relevant is reply A to the parent comment? (1-5)",
                    "relevance_b": "How relevant is reply B to the parent comment? (1-5)",
                    "coherence_a": "How coherent is reply A as a direct response? (1-5)",
                    "coherence_b": "How coherent is reply B as a direct response? (1-5)",
                    "preference": "Which reply is the better direct response? (A/B/tie)",
                },
            }
        )
        key.append(
            {
                "item_id": item_id,
                "candidate_a_source": (
                    "frame_rendered" if rendered_is_a else "verbatim_payload"
                ),
                "candidate_b_source": (
                    "verbatim_payload" if rendered_is_a else "frame_rendered"
                ),
                "intent_id": row.intent_id,
                "polarity": row.polarity,
                "source_topic_overlap": row.source_topic_overlap,
            }
        )
    return pd.DataFrame(public_rows), key


def build_semantic_parent_audit(
    root: str | Path,
    config: Mapping[str, Any],
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    project = Path(root)
    base_path = project / str(config["base_config"])
    raw = json.loads(base_path.read_text(encoding="utf-8"))
    raw["posts_per_community"] = int(config.get("posts_per_community", 3))
    raw["seeds"] = [int(value) for value in config.get("seeds", [0])]
    raw["audit_capacity_multiplier"] = int(
        config.get("audit_capacity_multiplier", 256)
    )
    output = project / str(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    candidates = _collect_candidates(
        project,
        raw,
        project / str(config.get("social_root", "data/social_paper")),
    )
    sample = _stratified_sample(
        candidates,
        int(config.get("sample_size", 600)),
        int(config.get("seed", 30371)),
    )
    public, key = _blind_items(sample, int(config.get("seed", 30371)))
    public.to_csv(output / "semantic_audit_items.csv", index=False)
    with (output / "semantic_audit_items.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for record in public.to_dict(orient="records"):
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    write_json(output / "semantic_audit_blinding_key.json", key)
    manifest = {
        "status": "materials_ready",
        "candidate_pairs": int(len(candidates)),
        "audit_items": int(len(public)),
        "communities": sorted(public["community"].unique().tolist()),
        "conditions": sorted(public["condition"].unique().tolist()),
        "depth_bins": sorted(public["depth_bin"].unique().tolist()),
        "blinded": True,
        "ratings_status": "not_collected",
        "base_config": {"path": str(base_path.resolve()), "sha256": sha256_file(base_path)},
    }
    if config_path is not None:
        source = Path(config_path)
        manifest["config"] = {
            "path": str(source.resolve()),
            "sha256": sha256_file(source),
        }
    write_json(output / "manifest.json", manifest)
    return manifest
