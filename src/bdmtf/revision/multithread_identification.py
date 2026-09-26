from __future__ import annotations

import json
import math
import os
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bdmtf.data.social_loader import (
    load_population,
    load_posts,
    resolve_community_paths,
)
from bdmtf.experiments import (
    _with_community_calibration,
    config_from_dict,
    interventions_from_config,
    simulation_seed,
)
from bdmtf.intent_pool import FrozenIntentPool
from bdmtf.revision.live_generation_identification import (
    PROTOCOL_FROZEN,
    PROTOCOL_LIVE,
    _clean_text,
    _load_cache,
    _request_generation,
    _run_simulation,
    _sha256_file,
    adapt_config_to_semantics,
    build_generation_prompt,
    dispersion_summary,
    intent_pool_from_items,
    load_leader_personas,
    response_rates,
    summarize_thread_state,
    validate_generation_response,
)
from bdmtf.revision.provenance import write_json


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def select_stratified_test_posts(
    social_root: Path,
    splits_path: Path,
    metrics_path: Path,
    communities: list[str],
    quantiles: list[float],
) -> list[dict[str, Any]]:
    """Select one unique held-out thread nearest each prespecified size quantile."""
    splits = pd.read_csv(splits_path, dtype={"post_id": str})
    metrics = pd.read_csv(metrics_path, dtype={"post_id": str})
    selected: list[dict[str, Any]] = []
    for community in communities:
        test_ids = set(
            splits[
                splits["community"].astype(str).str.lower().eq(community.lower())
                & splits["split"].astype(str).eq("test")
            ]["post_id"].astype(str)
        )
        eligible = metrics[
            metrics["community"].astype(str).str.lower().eq(community.lower())
            & metrics["split"].astype(str).eq("test")
            & metrics["post_id"].astype(str).isin(test_ids)
        ].copy()
        eligible["size"] = pd.to_numeric(eligible["size"], errors="coerce")
        eligible = eligible.dropna(subset=["size"])
        if len(eligible) < len(quantiles):
            raise ValueError(
                f"{community} has {len(eligible)} eligible test cascades, "
                f"fewer than {len(quantiles)} requested threads"
            )

        paths = resolve_community_paths(social_root, community)
        posts = load_posts(paths, enriched=True)
        posts["post_id"] = posts["post_id"].astype(str)
        target_sizes = [
            float(eligible["size"].quantile(float(q), interpolation="linear"))
            for q in quantiles
        ]
        remaining = eligible.copy()
        for slot, (quantile, target_size) in enumerate(
            zip(quantiles, target_sizes, strict=True)
        ):
            candidates = remaining.assign(
                distance_to_target=(remaining["size"] - target_size).abs()
            ).sort_values(["distance_to_target", "post_id"], kind="stable")
            metric = candidates.iloc[0]
            post_id = str(metric["post_id"])
            post_match = posts[posts["post_id"].eq(post_id)]
            if post_match.empty:
                raise ValueError(
                    f"Selected post {community}/{post_id} is absent from the post table"
                )
            post = post_match.iloc[0]
            metric_record = {
                key: (value.item() if hasattr(value, "item") else value)
                for key, value in metric.drop(labels=["distance_to_target"]).to_dict().items()
            }
            selected.append(
                {
                    "thread_index": len(selected),
                    "community_slot": slot,
                    "selection_quantile": float(quantile),
                    "selection_target_size": target_size,
                    "selection_rule": "nearest_unique_test_cascade_to_prespecified_size_quantile",
                    "post": {
                        "community": community,
                        "post_id": post_id,
                        "title": _clean_text(post.get("title", ""), 1800),
                        "full_text": _clean_text(post.get("full_text", ""), 3500),
                        "url": _clean_text(post.get("url", ""), 1000),
                        "created_utc": _clean_text(post.get("created_utc", ""), 100),
                        "is_viral": int(float(post.get("is_viral", 0) or 0)),
                        "final_num_comments": float(
                            post.get("final_num_comments", 0) or 0
                        ),
                        "early_num_comments": float(
                            post.get("early_num_comments", 0) or 0
                        ),
                        "early_score_sum": float(post.get("early_score_sum", 0) or 0),
                    },
                    "empirical": metric_record,
                }
            )
            remaining = remaining[~remaining["post_id"].astype(str).eq(post_id)]
    return selected


def _one_sided_sign_test(improved: int, non_tied: int) -> float:
    if non_tied <= 0:
        return 1.0
    return float(
        sum(math.comb(non_tied, value) for value in range(improved, non_tied + 1))
        / (2**non_tied)
    )


def _format_p_plain(value: float) -> str:
    return f"{value:.2e}" if value < 1e-4 else f"{value:.4f}"


def _format_p_tex(value: float) -> str:
    if value >= 1e-4:
        return f"{value:.4f}"
    exponent = int(math.floor(math.log10(value)))
    coefficient = value / (10**exponent)
    return f"{coefficient:.2f}\\times 10^{{{exponent}}}"


def summarize_multithread_dispersion(
    effects: pd.DataFrame,
    *,
    bootstrap_samples: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {"community", "post_id", "protocol", "repeat", "log_volume_effect"}
    missing = required - set(effects.columns)
    if missing:
        raise ValueError(f"Effects are missing columns: {sorted(missing)}")

    thread_rows: list[dict[str, Any]] = []
    for thread_index, ((community, post_id), group) in enumerate(
        effects.groupby(["community", "post_id"], sort=True)
    ):
        summary = dispersion_summary(
            group,
            "log_volume_effect",
            bootstrap_samples=max(200, min(bootstrap_samples, 2000)),
            permutation_samples=2000,
            seed=seed + thread_index,
        )
        live_var = float(summary["live_variance"])
        frozen_var = float(summary["frozen_variance"])
        if frozen_var == 0.0:
            variance_ratio = math.inf if live_var > 0 else 1.0
        else:
            variance_ratio = live_var / frozen_var
        thread_rows.append(
            {
                "community": str(community),
                "post_id": str(post_id),
                "paired_seeds": int(summary["paired_repeats"]),
                "live_sd": float(summary["live_sd"]),
                "frozen_sd": float(summary["frozen_sd"]),
                "live_variance": live_var,
                "frozen_variance": frozen_var,
                "variance_ratio_live_over_frozen": variance_ratio,
                "rehydration_reduced_dispersion": bool(live_var > frozen_var),
                "tie": bool(live_var == frozen_var),
                "live_mean_effect": float(summary["live_mean"]),
                "frozen_mean_effect": float(summary["frozen_mean"]),
            }
        )
    thread_df = pd.DataFrame(thread_rows)
    if thread_df.empty:
        raise ValueError("No thread-level effects were available")

    ratios = thread_df["variance_ratio_live_over_frozen"].to_numpy(dtype=float)
    improved = int(thread_df["rehydration_reduced_dispersion"].sum())
    tied = int(thread_df["tie"].sum())
    non_tied = int(len(thread_df) - tied)
    rng = np.random.default_rng(seed)
    median_bootstrap = np.empty(bootstrap_samples, dtype=float)
    share_bootstrap = np.empty(bootstrap_samples, dtype=float)
    improved_flags = thread_df["rehydration_reduced_dispersion"].to_numpy(dtype=float)
    for index in range(bootstrap_samples):
        sample = rng.integers(0, len(thread_df), size=len(thread_df))
        median_bootstrap[index] = float(np.median(ratios[sample]))
        share_bootstrap[index] = float(np.mean(improved_flags[sample]))

    community_rows = []
    for community, group in thread_df.groupby("community", sort=True):
        community_rows.append(
            {
                "community": str(community),
                "threads": int(len(group)),
                "threads_improved": int(
                    group["rehydration_reduced_dispersion"].sum()
                ),
                "median_variance_ratio": float(
                    np.median(
                        group["variance_ratio_live_over_frozen"].to_numpy(dtype=float)
                    )
                ),
            }
        )
    aggregate = {
        "threads": int(len(thread_df)),
        "paired_seeds_per_thread": int(thread_df["paired_seeds"].min()),
        "threads_with_reduced_dispersion": improved,
        "threads_tied": tied,
        "fraction_with_reduced_dispersion": float(improved / len(thread_df)),
        "fraction_95pct_thread_bootstrap_ci": [
            float(np.quantile(share_bootstrap, 0.025)),
            float(np.quantile(share_bootstrap, 0.975)),
        ],
        "median_variance_ratio_live_over_frozen": float(np.median(ratios)),
        "median_variance_ratio_95pct_thread_bootstrap_ci": [
            float(np.quantile(median_bootstrap, 0.025)),
            float(np.quantile(median_bootstrap, 0.975)),
        ],
        "one_sided_sign_test_p": _one_sided_sign_test(improved, non_tied),
        "community_summary": community_rows,
    }
    return thread_df, aggregate


def _write_multithread_report(
    path: Path,
    manifest: dict[str, Any],
    aggregate: dict[str, Any],
) -> None:
    rows = [
        "# Multi-Thread Rehydration Identification Diagnostic",
        "",
        "## Design",
        "",
        f"- {aggregate['threads']} held-out Reddit threads, with "
        f"{aggregate['paired_seeds_per_thread']} paired simulator seeds per thread.",
        "- Five threads are selected in each community at prespecified cascade-size "
        "quantiles (0.10, 0.30, 0.50, 0.70, 0.90).",
        "- Selection uses test eligibility and observed cascade size only; no "
        "Rehydration effect is computed before the sample is locked.",
        "- The primary outcomes are the number of threads with lower frozen-replay "
        "variance and the median live-to-frozen variance ratio.",
        "",
        "## Results",
        "",
        f"Rehydration reduces log-volume-effect dispersion on "
        f"{aggregate['threads_with_reduced_dispersion']}/{aggregate['threads']} "
        f"threads ({100.0 * aggregate['fraction_with_reduced_dispersion']:.1f}%). "
        f"The median live-to-frozen variance ratio is "
        f"{aggregate['median_variance_ratio_live_over_frozen']:.2f} "
        f"(thread-bootstrap 95% interval "
        f"{aggregate['median_variance_ratio_95pct_thread_bootstrap_ci'][0]:.2f}--"
        f"{aggregate['median_variance_ratio_95pct_thread_bootstrap_ci'][1]:.2f}); "
        f"one-sided sign-test p={_format_p_plain(aggregate['one_sided_sign_test_p'])}.",
        "",
        "| Community | Threads improved | Threads | Median variance ratio |",
        "|---|---:|---:|---:|",
    ]
    for item in aggregate["community_summary"]:
        rows.append(
            f"| {item['community']} | {item['threads_improved']} | "
            f"{item['threads']} | {item['median_variance_ratio']:.2f} |"
        )
    rows.extend(
        [
            "",
            "## Evidence boundary",
            "",
            "With five seeds per thread, individual variance ratios are noisy. The "
            "cross-thread sign count and median are therefore primary; per-thread "
            "ratios are retained for audit rather than interpreted as precise effects. "
            "The experiment tests whether the motivating instability generalizes "
            "across held-out threads, not whether every structural outcome benefits.",
            "",
            f"Model requested: {manifest['model_requested']}. No API key is stored.",
        ]
    )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _write_multithread_macros(path: Path, aggregate: dict[str, Any]) -> None:
    content = (
        "% Auto-generated by run_multithread_identification.py.\n"
        f"\\providecommand{{\\MultiThreadRQThreads}}{{{aggregate['threads']}}}\n"
        f"\\providecommand{{\\MultiThreadRQSeeds}}{{{aggregate['paired_seeds_per_thread']}}}\n"
        f"\\providecommand{{\\MultiThreadRQImproved}}{{{aggregate['threads_with_reduced_dispersion']}}}\n"
        f"\\providecommand{{\\MultiThreadRQImprovedPct}}{{{100.0 * aggregate['fraction_with_reduced_dispersion']:.1f}\\%}}\n"
        f"\\providecommand{{\\MultiThreadRQMedianVarianceRatio}}{{{aggregate['median_variance_ratio_live_over_frozen']:.2f}}}\n"
        f"\\providecommand{{\\MultiThreadRQMedianVarianceRatioLow}}{{{aggregate['median_variance_ratio_95pct_thread_bootstrap_ci'][0]:.2f}}}\n"
        f"\\providecommand{{\\MultiThreadRQMedianVarianceRatioHigh}}{{{aggregate['median_variance_ratio_95pct_thread_bootstrap_ci'][1]:.2f}}}\n"
        f"\\providecommand{{\\MultiThreadRQSignP}}{{{_format_p_tex(aggregate['one_sided_sign_test_p'])}}}\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_multithread_table(path: Path, aggregate: dict[str, Any]) -> None:
    lines = [
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Community & Improved & Threads & Median ratio \\\\",
        "\\midrule",
    ]
    for item in aggregate["community_summary"]:
        lines.append(
            f"{item['community']} & {item['threads_improved']} & "
            f"{item['threads']} & {item['median_variance_ratio']:.2f} \\\\"
        )
    lines.extend(
        [
            "\\midrule",
            f"All & {aggregate['threads_with_reduced_dispersion']} & "
            f"{aggregate['threads']} & "
            f"{aggregate['median_variance_ratio_live_over_frozen']:.2f} \\\\",
            "\\bottomrule",
            "\\end{tabular}",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_multithread_identification_experiment(
    repo_root: Path,
    config_path: Path,
    *,
    execute: bool = False,
    plan_only: bool = False,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    social_root = repo_root / config["social_root"]
    splits_path = repo_root / config["splits_path"]
    metrics_path = repo_root / config["empirical_metrics_path"]
    simulator_config_path = repo_root / config["simulator_config"]
    output_dir = repo_root / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = repo_root / config.get(
        "generation_cache_path",
        str(Path(config["output_dir"]) / "generation_cache.jsonl"),
    )

    communities = [str(value) for value in config["communities"]]
    quantiles = [float(value) for value in config["selection_quantiles"]]
    selected = select_stratified_test_posts(
        social_root, splits_path, metrics_path, communities, quantiles
    )
    selection_rows = [
        {
            "thread_index": item["thread_index"],
            "community": item["post"]["community"],
            "post_id": item["post"]["post_id"],
            "selection_quantile": item["selection_quantile"],
            "selection_target_size": item["selection_target_size"],
            "observed_size": item["empirical"]["size"],
            "is_viral": item["empirical"].get("is_viral"),
            "selection_rule": item["selection_rule"],
        }
        for item in selected
    ]
    selection_path = output_dir / "selected_threads.csv"
    pd.DataFrame(selection_rows).to_csv(selection_path, index=False)
    design_lock = {
        "created_at": _utc_now(),
        "status": "planned" if plan_only else "running",
        "communities": communities,
        "selection_quantiles": quantiles,
        "threads": len(selected),
        "repeats": int(config["repeats"]),
        "expected_api_calls": len(selected) * (1 + 2 * int(config["repeats"])),
        "selection_sha256": _sha256_file(selection_path),
        "selection_uses_intervention_effects": False,
        "model_requested": os.environ.get("OPENAI_MODEL", str(config["model"])),
    }
    write_json(output_dir / "design_lock.json", design_lock)
    if plan_only:
        return design_lock

    simulator_raw = json.loads(simulator_config_path.read_text(encoding="utf-8"))
    conditions = {
        item.name: item for item in interventions_from_config(simulator_raw)
    }
    baseline = conditions[config["baseline_condition"]]
    intervention = conditions[config["intervention_condition"]]
    selected_conditions = [baseline, intervention]
    cache = _load_cache(cache_path)
    run_rows: list[dict[str, Any]] = []
    generation_rows: list[dict[str, Any]] = []

    for selected_thread in selected:
        post = selected_thread["post"]
        community = str(post["community"])
        post_id = str(post["post_id"])
        task_prefix = f"{community}_{post_id}"
        paths = resolve_community_paths(social_root, community)
        base_config = _with_community_calibration(
            config_from_dict(simulator_raw), community
        )
        run_config = replace(
            base_config,
            initial_engagement_signal=(
                float(post["early_score_sum"]) + float(post["early_num_comments"])
            ),
        )
        personas = load_leader_personas(
            social_root, community, int(config["leaders_in_prompt"])
        )
        expected_agent_ids = {int(item["agent_id"]) for item in personas}
        frozen_prompt = build_generation_prompt(post, personas, None)
        frozen_task_id = f"frozen_{task_prefix}"
        frozen_record = _request_generation(
            task_id=frozen_task_id,
            protocol=PROTOCOL_FROZEN,
            repeat=None,
            condition=None,
            prompt=frozen_prompt,
            expected_agent_ids=expected_agent_ids,
            config=config,
            cache_path=cache_path,
            cache=cache,
            execute=execute,
        )
        frozen_items = validate_generation_response(
            frozen_record["content"], expected_agent_ids
        )
        frozen_rates = response_rates(frozen_items)
        generation_rows.append(
            {
                "community": community,
                "post_id": post_id,
                "task_id": frozen_task_id,
                "protocol": PROTOCOL_FROZEN,
                "repeat": None,
                "condition": None,
                "prompt_sha256": frozen_record["prompt_sha256"],
                "response_sha256": frozen_record["response_sha256"],
                "model_returned": frozen_record.get("model_returned"),
                **frozen_rates,
            }
        )

        for repeat in range(int(config["repeats"])):
            agents = load_population(paths, run_config, seed=int(config["seed"]))
            sim_seed = simulation_seed(
                community,
                post_id,
                baseline.name,
                repeat,
                seed_mode="paired_by_post_seed",
            )
            for condition in selected_conditions:
                pilot_config = replace(
                    run_config, steps=int(config["pilot_steps"])
                )
                pilot_pool = FrozenIntentPool.from_jsonl(
                    paths.base_dir / "frozen_intents.jsonl"
                )
                _, pilot_state, pilot_traces, pilot_simulator = _run_simulation(
                    post=post,
                    intervention=condition,
                    agents=agents,
                    intent_pool=pilot_pool,
                    config=pilot_config,
                    sim_seed=sim_seed,
                )
                snapshot = summarize_thread_state(
                    pilot_state, pilot_simulator, condition, pilot_traces
                )
                prompt = build_generation_prompt(post, personas, snapshot)
                task_id = f"live_{task_prefix}_{repeat:02d}_{condition.name}"
                record = _request_generation(
                    task_id=task_id,
                    protocol=PROTOCOL_LIVE,
                    repeat=repeat,
                    condition=condition.name,
                    prompt=prompt,
                    expected_agent_ids=expected_agent_ids,
                    config=config,
                    cache_path=cache_path,
                    cache=cache,
                    execute=execute,
                )
                items = validate_generation_response(
                    record["content"], expected_agent_ids
                )
                rates = response_rates(items)
                live_pool = intent_pool_from_items(items, task_id, PROTOCOL_LIVE)
                live_config, factors = adapt_config_to_semantics(
                    run_config, condition, rates, frozen_rates, config
                )
                live_metrics, _, _, _ = _run_simulation(
                    post=post,
                    intervention=condition,
                    agents=agents,
                    intent_pool=live_pool,
                    config=live_config,
                    sim_seed=sim_seed,
                )
                run_rows.append(
                    {
                        "community": community,
                        "post_id": post_id,
                        "protocol": PROTOCOL_LIVE,
                        "repeat": repeat,
                        "sim_seed": sim_seed,
                        "condition": condition.name,
                        **rates,
                        **factors,
                        **live_metrics,
                    }
                )
                generation_rows.append(
                    {
                        "community": community,
                        "post_id": post_id,
                        "task_id": task_id,
                        "protocol": PROTOCOL_LIVE,
                        "repeat": repeat,
                        "condition": condition.name,
                        "prompt_sha256": record["prompt_sha256"],
                        "response_sha256": record["response_sha256"],
                        "model_returned": record.get("model_returned"),
                        **rates,
                    }
                )

                frozen_pool = intent_pool_from_items(
                    frozen_items, frozen_task_id, PROTOCOL_FROZEN
                )
                frozen_metrics, _, _, _ = _run_simulation(
                    post=post,
                    intervention=condition,
                    agents=agents,
                    intent_pool=frozen_pool,
                    config=run_config,
                    sim_seed=sim_seed,
                )
                run_rows.append(
                    {
                        "community": community,
                        "post_id": post_id,
                        "protocol": PROTOCOL_FROZEN,
                        "repeat": repeat,
                        "sim_seed": sim_seed,
                        "condition": condition.name,
                        **frozen_rates,
                        "reply_factor": 1.0,
                        "antagonism_factor": 1.0,
                        **frozen_metrics,
                    }
                )

    runs = pd.DataFrame(run_rows)
    generations = pd.DataFrame(generation_rows)
    effect_rows: list[dict[str, Any]] = []
    for (community, post_id, protocol, repeat), group in runs.groupby(
        ["community", "post_id", "protocol", "repeat"], sort=True
    ):
        by_condition = group.set_index("condition")
        base = by_condition.loc[baseline.name]
        treated = by_condition.loc[intervention.name]
        effect_rows.append(
            {
                "community": community,
                "post_id": post_id,
                "protocol": protocol,
                "repeat": int(repeat),
                "sim_seed": int(base["sim_seed"]),
                "log_volume_effect": float(
                    np.log1p(treated["comment_volume"])
                    - np.log1p(base["comment_volume"])
                ),
                "volume_ratio": float(
                    (treated["comment_volume"] + 1.0)
                    / (base["comment_volume"] + 1.0)
                ),
                "mean_leaf_depth_effect": float(
                    treated["mean_leaf_depth"] - base["mean_leaf_depth"]
                ),
                "max_depth_effect": float(
                    treated["max_depth"] - base["max_depth"]
                ),
            }
        )
    effects = pd.DataFrame(effect_rows)
    thread_summary, aggregate = summarize_multithread_dispersion(
        effects,
        bootstrap_samples=int(config["bootstrap_samples"]),
        seed=int(config["seed"]),
    )

    runs.to_csv(output_dir / "runs.csv", index=False)
    effects.to_csv(output_dir / "paired_effects.csv", index=False)
    generations.to_csv(output_dir / "generation_index.csv", index=False)
    thread_summary.to_csv(output_dir / "thread_dispersion.csv", index=False)
    write_json(output_dir / "summary.json", aggregate)

    cache_records = list(cache.values())
    manifest = {
        **design_lock,
        "created_at": _utc_now(),
        "status": "complete",
        "model_returned": sorted(
            str(value) for value in generations["model_returned"].dropna().unique()
        ),
        "base_host": next(
            (
                str(item["base_host"])
                for item in cache_records
                if item.get("base_host")
            ),
            str(config.get("api_base_host", "unknown")),
        ),
        "key_stored": False,
        "api_usage": {
            "successful_tasks": len(cache_records),
            "prompt_tokens": int(
                sum(
                    float(item.get("usage", {}).get("prompt_tokens", 0) or 0)
                    for item in cache_records
                )
            ),
            "completion_tokens": int(
                sum(
                    float(item.get("usage", {}).get("completion_tokens", 0) or 0)
                    for item in cache_records
                )
            ),
        },
        "files": {
            "config": {
                "path": str(config_path.relative_to(repo_root)),
                "sha256": _sha256_file(config_path),
            },
            "simulator_config": {
                "path": str(simulator_config_path.relative_to(repo_root)),
                "sha256": _sha256_file(simulator_config_path),
            },
            "splits": {
                "path": str(splits_path.relative_to(repo_root)),
                "sha256": _sha256_file(splits_path),
            },
            "metrics": {
                "path": str(metrics_path.relative_to(repo_root)),
                "sha256": _sha256_file(metrics_path),
            },
            "selection": {
                "path": str(selection_path.relative_to(repo_root)),
                "sha256": _sha256_file(selection_path),
            },
        },
        "summary": aggregate,
    }
    write_json(output_dir / "manifest.json", manifest)
    _write_multithread_report(
        output_dir / "MULTITHREAD_IDENTIFICATION_REPORT.md",
        manifest,
        aggregate,
    )
    for relative_path in config.get("paper_macro_paths", []):
        _write_multithread_macros(repo_root / relative_path, aggregate)
    for relative_path in config.get("paper_table_paths", []):
        _write_multithread_table(repo_root / relative_path, aggregate)
    return manifest
