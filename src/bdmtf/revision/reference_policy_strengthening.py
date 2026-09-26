"""Frozen-reference factorial and training-matched activation-form comparisons."""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bdmtf.data.social_loader import load_population, load_posts, resolve_community_paths
from bdmtf.experiments import config_from_dict, interventions_from_config, _with_community_calibration
from bdmtf.features import estimate_toxicity
from bdmtf.intent_pool import FrozenIntentPool
from bdmtf.metrics import compute_metrics
from bdmtf.schema import Intent, RankingPolicy
from bdmtf.simulator import BDMTFSimulator
from bdmtf.revision.api_intents import _parse_json_object
from bdmtf.revision.live_generation_identification import adapt_config_to_semantics, response_rates
from bdmtf.revision.provenance import sha256_file, write_json


def stable_seed(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "big")


def activation_signal(drives, exposure, weights, form, anchors):
    d, e, w = np.asarray(drives), np.asarray(exposure), np.asarray(weights)
    if form == "original":
        return np.sum(w * d * e, axis=-1)
    if form == "additive":
        d0, e0 = np.asarray(anchors["drive_mean"]), np.asarray(anchors["exposure_mean"])
        return np.sum(w * (d0 * e + e0 * d - d0 * e0), axis=-1)
    if form == "saturating":
        scale = anchors["saturation_scale"]
        return scale * np.tanh(np.sum(w * d * e, axis=-1) / scale)
    raise ValueError(f"Unknown activation form: {form}")


class ComparisonSimulator(BDMTFSimulator):
    """New panels only: operation-keyed streams; legacy simulator stays unchanged."""

    def __init__(self, *args, form="original", anchors=None, rank_null=False,
                 collect_activation=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.comparison_seed = kwargs.get("seed", 0)
        self.form = form
        self.anchors = anchors or {}
        self.rank_null = rank_null
        self.collect_activation = collect_activation
        self.activation_rows = []
        self._impulse_calls = 0
        self._step = -1
        self._agent = -1

    def _stream(self, operation):
        self.rng = random.Random(stable_seed(self.comparison_seed, self._step, self._agent, operation))

    def run(self, *args, **kwargs):
        if self.rank_null:
            kwargs["intervention"] = replace(kwargs["intervention"], ranking=RankingPolicy.BEST)
        return super().run(*args, **kwargs)

    def _compute_impulse(self, agent, visible):
        self._step = self._impulse_calls // len(self.base_agents)
        self._agent = agent.agent_id
        self._impulse_calls += 1
        self._stream("impulse_and_action")
        impulse, signals = super()._compute_impulse(agent, visible)
        drives = [agent.antagonism, agent.prosocial, agent.attention]
        exposure = [signals["controversy"], signals["consensus"], signals["heat"]]
        weights = [self.config.alpha_conflict, self.config.gamma_consensus, self.config.beta_heat]
        original = float(activation_signal(drives, exposure, weights, "original", {}))
        residual = impulse - original
        if self.collect_activation and not (self._step == 0 and agent.is_leader):
            self.activation_rows.append([*drives, *exposure, residual, agent.threshold, impulse])
        if self.form != "original":
            impulse = (residual + float(activation_signal(drives, exposure, weights, self.form, self.anchors))
                       + self.anchors["offsets"][self.form])
        return impulse, signals

    def _select_target(self, *args, **kwargs):
        self._stream("target")
        return super()._select_target(*args, **kwargs)

    def _sample_polarity(self, *args, **kwargs):
        self._stream("polarity")
        return super()._sample_polarity(*args, **kwargs)

    def _should_reply(self, *args, **kwargs):
        self._stream("reply")
        return super()._should_reply(*args, **kwargs)

    def _inject_external_traffic(self, state, step, **kwargs):
        self._step, self._agent = step, -1
        self._stream("external")
        return super()._inject_external_traffic(state, step, **kwargs)


def fit_activation_anchors(rows, weights):
    a = np.asarray(rows, dtype=float)
    if len(a) == 0 or not np.isfinite(a).all():
        raise ValueError("Finite training activation observations required")
    anchors = {"drive_mean": a[:, :3].mean(axis=0).tolist(),
               "exposure_mean": a[:, 3:6].mean(axis=0).tolist(),
               "saturation_scale": max(.1, float(np.median(a[:, 8] - a[:, 6]))),
               "offsets": {"original": 0.0}, "training_decisions": len(a)}
    target_rate = float(np.mean(a[:, 8] > a[:, 7]))
    anchors["target_gate_rate"] = target_rate
    anchors["matched_gate_rates"] = {"original": target_rate}
    for form in ("additive", "saturating"):
        score = a[:, 6] + activation_signal(a[:, :3], a[:, 3:6], weights, form, anchors)
        # Match one baseline gate-rate, not volume, depth, or a treatment effect.
        margin = a[:, 7] - score
        lo, hi = float(margin.min() - 1), float(margin.max() + 1)
        for _ in range(60):
            midpoint = (lo + hi) / 2
            if np.mean(margin < midpoint) < target_rate:
                lo = midpoint
            else:
                hi = midpoint
        offset = (lo + hi) / 2
        anchors["offsets"][form] = offset
        anchors["matched_gate_rates"][form] = float(np.mean(score + offset > a[:, 7]))
    return anchors


def cached_items(path):
    records = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]
    rows = []
    seen = set()
    for record in records:
        key = (record["community"], str(record["post_id"]), record["family"], record["leader"])
        if key in seen:
            raise ValueError(f"Duplicate cache item: {key}")
        seen.add(key)
        if hashlib.sha256(record["response"].encode()).hexdigest() != record["response_sha256"]:
            raise ValueError("Cache response hash mismatch")
        response = _parse_json_object(record["response"])
        if response["action"] not in {"reply", "abstain"}:
            raise ValueError("Unexpected cached action")
        rows.append({**{k: record[k] for k in ("community", "post_id", "family", "model", "leader", "task_id", "prompt_sha256")},
                     "agent_id": int(record["leader"]), **response})
    return pd.DataFrame(rows)


def build_reference_pool(background, items):
    # Common empirical background handles unmatched agent/polarity requests.
    # Its presence and realized use are audited, never called model-generated.
    catalog = list(background.unique_intents())
    for item in items:
        if item["action"] == "reply":
            catalog.append(Intent(
                intent_id="api_" + item["family"] + "_" + item["task_id"],
                agent_id=item["agent_id"],
                polarity="antagonistic" if item["polarity"] == "antagonistic" else "supportive",
                content=item["content"],
                metadata={"source": "cached_model", "family": item["family"],
                          "toxicity": estimate_toxicity(item["content"])}))
    return FrozenIntentPool(catalog)


def freeze_protocol(root, config_path):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = root / config["output"]
    output.mkdir(parents=True, exist_ok=True)
    items = cached_items(root / config["cache"])
    counts = items.groupby(["community", "post_id", "family"]).size()
    if not counts.eq(10).all() or set(items.family) != set(config["families"]):
        raise ValueError("Expected ten cached participants per post and each declared family")
    posts = items[["community", "post_id"]].drop_duplicates().sort_values(["community", "post_id"])
    if not items.groupby(["community", "post_id"]).family.nunique().eq(3).all():
        raise ValueError("Every post must have all families")
    # All family prompts must represent the same participant and context.
    if not items.groupby(["community", "post_id", "leader"]).prompt_sha256.nunique().eq(1).all():
        raise ValueError("Unmatched cross-family prompt")
    splits = pd.read_csv(root / config["splits"], dtype={"post_id": str})
    selected = posts.merge(splits, on=["community", "post_id"], validate="one_to_one")
    if not selected.split.eq("test").all():
        raise ValueError("Cache comparison posts must be outside activation training")
    train = splits[splits.split.eq("train")].sort_values(["community", "created_at", "post_id"])
    train = train.groupby("community").head(config["training_posts_per_community"])
    files = [config_path, root / config["cache"], root / config["splits"], root / config["base_config"],
             root / config["semantic_mapping_config"], Path(__file__), root / "src/bdmtf/simulator.py",
             root / "src/bdmtf/intent_pool.py", root / "src/bdmtf/metrics.py", root / "src/bdmtf/schema.py",
             root / "src/bdmtf/revision/live_generation_identification.py"]
    for community in sorted(posts.community.unique()):
        folder = root / "data/social_paper" / f"{community}_data"
        files.extend(folder.glob("posts_features*.csv"))
        files.extend([folder / "population.json", folder / "frozen_intents.jsonl"])
    manifest = {"config": config, "inputs": {str(p.relative_to(root)): sha256_file(p) for p in files},
                "post_count": len(posts), "cached_responses": len(items),
                "evaluation_ids": posts.to_dict("records"),
                "training_ids": train[["community", "post_id"]].to_dict("records"),
                "expected_runs": len(posts) * (len(config["seeds"]) * 4 * (3 + len(config["activation_forms"])) + 4),
                "new_api_calls": 0, "scope": config["scope"]}
    path = output / "frozen_protocol.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("Frozen protocol changed; use a new output directory")
    write_json(path, manifest)
    return config, manifest


def _post_config(base, post):
    early = [float(post.get(name, 0) or 0) for name in ("early_score_sum", "early_num_comments")]
    return replace(base, initial_engagement_signal=sum(v for v in early if math.isfinite(v) and v > 0))


def event_records(run_id, nodes):
    return [{"run_id": run_id, "node_id": n.node_id, "parent_id": n.parent_id,
             "agent_id": n.author_id, "step": n.created_step, "depth": n.depth,
             "intent_id": n.metadata.get("intent_id"), "toxicity": n.toxicity}
            for n in nodes]


def run_community(root, config_path, community):
    root, config_path = Path(root), Path(config_path)
    config, protocol = freeze_protocol(root, config_path)
    output = root / config["output"] / "workers" / community
    output.mkdir(parents=True, exist_ok=True)
    paths = resolve_community_paths(root / "data/social_paper", community)
    posts = load_posts(paths, enriched=True).set_index("post_id")
    raw = json.loads((root / config["base_config"]).read_text(encoding="utf-8"))
    base = replace(_with_community_calibration(config_from_dict(raw), community),
                   intent_pool_capacity_multiplier=config["capacity_multiplier"], semantic_payload_mode="frame_rendered")
    conditions = interventions_from_config(raw)
    mapping = json.loads((root / config["semantic_mapping_config"]).read_text(encoding="utf-8"))
    background = FrozenIntentPool.from_jsonl(paths.base_dir / "frozen_intents.jsonl")
    agents_by_seed = {s: load_population(paths, base, seed=s) for s in config["seeds"]}
    train_ids = [r["post_id"] for r in protocol["training_ids"] if r["community"] == community]
    eval_ids = [r["post_id"] for r in protocol["evaluation_ids"] if r["community"] == community]
    if set(train_ids) & set(eval_ids):
        raise ValueError("Activation calibration leaks evaluation posts")
    activation_rows = []
    for post_id in train_ids:
        post = posts.loc[post_id].to_dict()
        sim = ComparisonSimulator(agents_by_seed[config["seeds"][0]], background, config=_post_config(base, post),
                                  seed=stable_seed(community, post_id, "train"), collect_activation=True)
        sim.run(post_id=post_id, title=str(post["title"]), initial_text=str(post.get("full_text", "")), intervention=conditions[0])
        activation_rows.extend(sim.activation_rows)
    anchors = fit_activation_anchors(activation_rows, [base.alpha_conflict, base.gamma_consensus, base.beta_heat])
    anchors["training_ids"] = train_ids
    write_json(output / "activation_training_fit.json", anchors)
    items = cached_items(root / config["cache"])
    rows = []
    event_dir = output / "events"
    event_dir.mkdir(exist_ok=True)
    start = time.monotonic()
    runs_path = output / "runs.jsonl"
    done = set()
    if runs_path.exists():
        rows = [json.loads(line) for line in runs_path.read_text(encoding="utf-8").splitlines() if line]
        done = {row["run_id"] for row in rows}
    for post_id in eval_ids:
        post = posts.loc[post_id].to_dict()
        for seed in config["seeds"]:
            specs = [("qref", family, "original", False) for family in config["families"]]
            specs += [("activation", "empirical", form, False) for form in config["activation_forms"]]
            if seed == config["seeds"][0]:
                specs.append(("rank_null", "empirical", "original", True))
            for panel, family, form, null in specs:
                family_items = items[(items.community == community) & (items.post_id == post_id) & (items.family == family)].to_dict("records")
                pool = build_reference_pool(background, family_items) if panel == "qref" else background
                rates = response_rates(family_items) if family_items else None
                for condition in conditions:
                    run_id = f"{community}:{post_id}:{seed}:{panel}:{family}:{form}:{condition.name}"
                    if run_id in done:
                        continue
                    sim_config = _post_config(base, post)
                    factors = {}
                    if rates is not None:
                        sim_config, factors = adapt_config_to_semantics(sim_config, condition, rates, config["reference_rates"], mapping)
                    sim = ComparisonSimulator(agents_by_seed[seed], pool, config=sim_config,
                                              seed=stable_seed(community, post_id, seed), form=form, anchors=anchors, rank_null=null)
                    state, traces = sim.run(post_id=post_id, title=str(post["title"]),
                                            initial_text=str(post.get("full_text", "")), intervention=condition)
                    nodes = [n for n in state.comments.values() if not n.metadata.get("root_post")]
                    row = {"run_id": run_id, "community": community, "post_id": post_id, "seed": seed,
                           "panel": panel, "family": family, "form": form, "condition": condition.name,
                           "api_realized_replies": sum(str(n.metadata.get("intent_id", "")).startswith("api_") for n in nodes),
                           "surface_toxicity_mean": float(np.mean([n.toxicity for n in nodes])) if nodes else 0.0,
                           **factors, **compute_metrics(state, traces)}
                    event_rows = event_records(run_id, nodes)
                    columns = ["run_id", "node_id", "parent_id", "agent_id", "step", "depth", "intent_id", "toxicity"]
                    event_path = event_dir / (hashlib.sha256(run_id.encode()).hexdigest()[:24] + ".parquet")
                    pd.DataFrame(event_rows, columns=columns).to_parquet(event_path, index=False)
                    row["event_path"] = str(event_path.relative_to(root))
                    row["event_sha256"] = sha256_file(event_path)
                    with runs_path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    rows.append(row)
    write_json(output / "complete.json", {"runs": len(rows), "wall_seconds": time.monotonic() - start})
    return len(rows)


def factorial_contrasts(runs):
    keys = ["community", "post_id", "seed", "panel", "family", "form"]
    metrics = ["comment_volume", "mean_leaf_depth", "surface_toxicity_mean"]
    p = runs.pivot(index=keys, columns="condition", values=metrics)
    a, b, c, d = "BASELINE", "BASELINE_CONTROVERSIAL", "CORE_TOXIC_BEST", "CORE_TOXIC_CONTROVERSIAL"
    if p.isna().any().any():
        raise ValueError("Incomplete factorial blocks")
    out = p.index.to_frame(index=False)
    v = {cell: np.log1p(p["comment_volume"][cell].to_numpy()) for cell in (a, b, c, d)}
    depth = {cell: p["mean_leaf_depth"][cell].to_numpy() for cell in (a, b, c, d)}
    out["joint_volume_log"] = v[d] - v[a]
    out["interaction_volume_log"] = v[d] - v[c] - v[b] + v[a]
    out["joint_depth"] = depth[d] - depth[a]
    out["interaction_depth"] = depth[d] - depth[c] - depth[b] + depth[a]
    out["baseline_ranking_log"] = v[b] - v[a]
    out["toxic_ranking_log"] = v[d] - v[c]
    out["joint_surface_toxicity"] = (p["surface_toxicity_mean"][d] - p["surface_toxicity_mean"][a]).to_numpy()
    return out


def summarize(root, config_path):
    config, protocol = freeze_protocol(root, config_path)
    output = root / config["output"]
    paths = sorted((output / "workers").glob("*/runs.jsonl"))
    runs = pd.DataFrame([json.loads(line) for p in paths for line in p.read_text(encoding="utf-8").splitlines() if line])
    if len(runs) != protocol["expected_runs"] or runs.run_id.duplicated().any():
        raise ValueError(f"Expected {protocol['expected_runs']} unique runs, got {len(runs)}")
    for row in runs.itertuples():
        if sha256_file(root / row.event_path) != row.event_sha256:
            raise ValueError(f"Missing or changed event path: {row.run_id}")
    contrast = factorial_contrasts(runs)
    contrast.to_csv(output / "paired_contrasts.csv", index=False)
    runs.to_csv(output / "run_metrics.csv", index=False)
    metrics = ["joint_volume_log", "interaction_volume_log", "joint_depth", "interaction_depth", "joint_surface_toxicity"]
    rng = np.random.default_rng(config["analysis_seed"])
    rows = []
    for (panel, family, form), group in contrast.groupby(["panel", "family", "form"], sort=True):
        for metric in metrics:
            cluster = group.groupby(["community", "post_id"])[metric].mean()
            estimates = []
            for _ in range(config["bootstrap_samples"]):
                sampled = [rng.choice(g.to_numpy(), len(g), replace=True) for _, g in cluster.groupby(level=0)]
                estimates.append(float(np.concatenate(sampled).mean()))
            mean = float(cluster.mean())
            low, high = np.quantile(estimates, [.025, .975])
            if metric.endswith("_log"):
                mean, low, high = np.exp([mean, low, high])
            rows.append({"panel": panel, "family": family, "form": form, "metric": metric,
                         "estimate": mean, "ci_low": low, "ci_high": high, "posts": len(cluster)})
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "effect_summary.csv", index=False)
    # Paired cross-family and cross-function differences, preserving common posts/seeds.
    differences = []
    for panel, axis in [("qref", "family"), ("activation", "form")]:
        sub = contrast[contrast.panel == panel]
        labels = sorted(sub[axis].unique())
        for j, left in enumerate(labels):
            for right in labels[j + 1:]:
                matched = sub[sub[axis] == left].merge(sub[sub[axis] == right], on=["community", "post_id", "seed"], suffixes=("_l", "_r"), validate="one_to_one")
                for metric in metrics:
                    values = matched[["community", "post_id"]].copy()
                    values["gap"] = matched[metric + "_l"] - matched[metric + "_r"]
                    values = values.groupby(["community", "post_id"]).gap.mean()
                    draws = [np.concatenate([rng.choice(g, len(g)) for _, g in values.groupby(level=0)]).mean() for _ in range(config["bootstrap_samples"])]
                    low, high = np.quantile(draws, [.025, .975])
                    differences.append({"panel": panel, "left": left, "right": right, "metric": metric,
                                        "mean_difference": float(values.mean()), "ci_low": low, "ci_high": high})
    pd.DataFrame(differences).to_csv(output / "paired_between_variant_differences.csv", index=False)
    null = contrast[contrast.panel == "rank_null"]
    null_error = float(null[["interaction_volume_log", "interaction_depth", "baseline_ranking_log", "toxic_ranking_log"]].abs().max().max())
    if null_error != 0:
        raise AssertionError("Rank-disconnected contract control failed")
    usage = runs.groupby(["panel", "family", "form"])[["intent_requests", "pool_exhausted", "api_realized_replies", "comment_volume"]].sum()
    usage["exhaustion_rate"] = usage.pool_exhausted / usage.intent_requests
    usage["api_realized_share"] = usage.api_realized_replies / usage.comment_volume
    usage.to_csv(output / "pool_usage.csv")
    result = {"status": "complete", "runs": len(runs), "posts": protocol["post_count"],
              "rank_null_max_error": null_error, "new_api_calls": 0,
              "scope": config["scope"], "independent_generation_replicates": 1,
              "protocol_sha256": sha256_file(output / "frozen_protocol.json"),
              "source_hashes_unchanged": True,
              "artifacts": {p.name: sha256_file(p) for p in output.glob("*.csv")}}
    write_json(output / "result_manifest.json", result)
    return result
