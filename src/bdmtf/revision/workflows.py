from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from bdmtf.revision.api_intents import (
    generate_intents,
    load_intent_response_frame,
)
from bdmtf.revision.causal_validation import (
    analyze_human_rct,
    evaluate_intervention_fidelity,
    fit_pre_intervention_forecast,
    run_natural_experiments,
)
from bdmtf.revision.data_pipeline import COMMUNITIES, audit_dataset, build_splits
from bdmtf.revision.evaluation import evaluate_fidelity, run_model_suite, summarize_model_ranking
from bdmtf.revision.external_data import audit_external_datasets, read_table
from bdmtf.revision.external_sources import (
    build_story_matches,
    collect_lemmy,
    fetch_tbbt,
    import_reddit_expansion,
    import_tbbt,
    make_prospective_split,
)
from bdmtf.revision.external_validation import (
    run_cross_platform_transfer,
    run_leave_one_community_out,
    summarize_cross_community_fidelity,
    summarize_cross_platform_fidelity,
)
from bdmtf.revision.fitted_models import fit_models, load_models
from bdmtf.revision.interventions import analyze_intervention_file
from bdmtf.revision.lemmy_outcomes import collect_lemmy_outcomes
from bdmtf.revision.lemmy_intervention_fidelity import (
    run_lemmy_intervention_fidelity,
)
from bdmtf.revision.provenance import sha256_file, tree_manifest, write_json
from bdmtf.revision.reddit_archives import import_reddit_archive_bundles
from bdmtf.revision.reddit_expansion_validation import run_reddit_expansion_validation
from bdmtf.revision.simulation import RevisionSimulationConfig, event_metrics, metrics_from_event_records, simulate_cascade


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def paths_for(root: Path) -> dict[str, Path]:
    artifacts = root / "artifacts"
    return {
        "root": root,
        "social": root / "data" / "social_paper",
        "splits": artifacts / "splits",
        "audit": artifacts / "data_audit",
        "models": artifacts / "models" / "community_models.json",
        "baselines": artifacts / "runs" / "baselines",
        "main": artifacts / "runs" / "main",
        "ablations": artifacts / "runs" / "ablations",
        "api": artifacts / "api",
        "evaluation": artifacts / "evaluation",
        "external": artifacts / "external_validation",
        "interventions": artifacts / "interventions",
        "stories": artifacts / "story_matching",
        "prospective": artifacts / "prospective",
        "human_rct": artifacts / "human_rct",
        "provenance": artifacts / "provenance",
    }


def make_splits_workflow(root: Path, config: dict[str, Any]) -> pd.DataFrame:
    paths = paths_for(root)
    data_config = config.get("data", {})
    return build_splits(
        paths["social"],
        paths["splits"],
        config.get("communities", COMMUNITIES),
        float(data_config.get("train_fraction", 0.6)),
        float(data_config.get("validation_fraction", 0.2)),
    )


def audit_workflow(root: Path, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    paths = paths_for(root)
    split_file = paths["splits"] / "post_splits.csv"
    splits = pd.read_csv(split_file) if split_file.exists() else make_splits_workflow(root, config)
    return audit_dataset(paths["social"], paths["audit"], splits)


def fit_workflow(root: Path) -> dict[str, Any]:
    paths = paths_for(root)
    metrics = paths["audit"] / "empirical_cascade_metrics.parquet"
    if not metrics.exists():
        raise FileNotFoundError("Run audit-data before fit-models")
    return fit_models(metrics, paths["models"])


def run_suite_workflow(root: Path, config: dict[str, Any], names: Iterable[str], output_key: str, seeds: Iterable[int], max_posts: int = 0):
    paths = paths_for(root)
    empirical = pd.read_parquet(paths["audit"] / "empirical_cascade_metrics.parquet")
    models = load_models(paths["models"])
    simulation = RevisionSimulationConfig.from_dict(config.get("simulation", {}))
    return run_model_suite(empirical, models, paths[output_key], simulation, names, seeds, max_posts)


def evaluate_workflow(root: Path, bootstrap_samples: int = 400) -> pd.DataFrame:
    paths = paths_for(root)
    empirical = pd.read_parquet(paths["audit"] / "empirical_cascade_metrics.parquet")
    frames = []
    for key in ("baselines", "main"):
        candidate = paths[key] / "simulated_metrics.parquet"
        if candidate.exists():
            frames.append(pd.read_parquet(candidate))
    if not frames:
        raise FileNotFoundError("Run run-baselines and run-main first")
    simulated = pd.concat(frames, ignore_index=True)
    results = evaluate_fidelity(empirical, simulated, paths["evaluation"], bootstrap_samples=bootstrap_samples)
    ranking = summarize_model_ranking(results)
    ranking.to_csv(paths["evaluation"] / "model_ranking.csv", index=False)
    return results


def audit_external_workflow(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    paths = paths_for(root)
    _, platform_audit = audit_external_datasets(root, config, paths["external"] / "cross_platform")
    intervention_audit = analyze_intervention_file(root, config, paths["external"] / "interventions")
    payload = {
        "cross_platform": platform_audit,
        "intervention": intervention_audit,
        "external_claims_ready": bool(platform_audit.get("claim_allowed")) and bool(intervention_audit.get("claim_allowed")),
    }
    write_json(paths["external"] / "readiness.json", payload)
    return payload


def cross_community_workflow(
    root: Path,
    config: dict[str, Any],
    seeds: Iterable[int],
    max_posts: int = 0,
    bootstrap_samples: int = 400,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = paths_for(root)
    empirical_path = paths["audit"] / "empirical_cascade_metrics.parquet"
    if not empirical_path.exists():
        raise FileNotFoundError("Run audit-data before run-cross-community")
    empirical = pd.read_parquet(empirical_path)
    external = config.get("cross_community", {})
    model_names = external.get(
        "models",
        ["empirical_bootstrap", "branching_process", "hawkes", "legacy_heuristic", "learned_bdmtf"],
    )
    simulation = RevisionSimulationConfig.from_dict(config.get("simulation", {}))
    return run_leave_one_community_out(
        empirical,
        paths["external"] / "cross_community",
        simulation,
        model_names,
        seeds,
        max_posts,
        bootstrap_samples,
    )


def cross_platform_workflow(
    root: Path,
    config: dict[str, Any],
    seeds: Iterable[int],
    max_cascades: int = 0,
    bootstrap_samples: int = 400,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = paths_for(root)
    reddit_path = paths["audit"] / "empirical_cascade_metrics.parquet"
    if not reddit_path.exists():
        raise FileNotFoundError("Run audit-data before run-cross-platform")
    external_metrics, audit = audit_external_datasets(root, config, paths["external"] / "cross_platform")
    if not audit.get("claim_allowed") or external_metrics.empty:
        raise ValueError("No real external platform dataset passed audit; run collect-hackernews and audit-external-data first")
    external = config.get("cross_community", {})
    model_names = external.get(
        "models",
        ["empirical_bootstrap", "branching_process", "hawkes", "legacy_heuristic", "learned_bdmtf"],
    )
    return run_cross_platform_transfer(
        pd.read_parquet(reddit_path),
        external_metrics,
        paths["external"] / "cross_platform_transfer",
        RevisionSimulationConfig.from_dict(config.get("simulation", {})),
        model_names,
        seeds,
        max_cascades,
        bootstrap_samples,
    )


def recompute_saved_metrics_workflow(root: Path, bootstrap_samples: int = 400) -> dict[str, int]:
    """Repair summary artifacts from event-level records after metric-definition fixes."""
    paths = paths_for(root)
    run_dirs = {
        "baselines": paths["baselines"],
        "main": paths["main"],
        "cross_community": paths["external"] / "cross_community",
        "cross_platform": paths["external"] / "cross_platform_transfer",
    }
    counts: dict[str, int] = {}
    for name, directory in run_dirs.items():
        events_path = directory / "events.parquet"
        if not events_path.exists():
            continue
        metrics = metrics_from_event_records(pd.read_parquet(events_path))
        metrics.to_csv(directory / "simulated_metrics.csv", index=False)
        metrics.to_parquet(directory / "simulated_metrics.parquet", index=False)
        counts[name] = int(len(metrics))

    if "baselines" in counts or "main" in counts:
        evaluate_workflow(root, bootstrap_samples)
    empirical = pd.read_parquet(paths["audit"] / "empirical_cascade_metrics.parquet")
    cross_community_dir = run_dirs["cross_community"]
    if "cross_community" in counts:
        simulated = pd.read_parquet(cross_community_dir / "simulated_metrics.parquet")
        fidelity = evaluate_fidelity(empirical, simulated, cross_community_dir / "evaluation", bootstrap_samples)
        ranking = summarize_model_ranking(fidelity)
        ranking.to_csv(cross_community_dir / "evaluation" / "model_ranking.csv", index=False)
        summary = summarize_cross_community_fidelity(fidelity, cross_community_dir / "evaluation", len(simulated))
        write_json(cross_community_dir / "summary.json", summary)
    cross_platform_dir = run_dirs["cross_platform"]
    if "cross_platform" in counts:
        simulated = pd.read_parquet(cross_platform_dir / "simulated_metrics.parquet")
        targets = pd.read_parquet(cross_platform_dir / "empirical_metrics.parquet")
        fidelity = evaluate_fidelity(targets, simulated, cross_platform_dir / "evaluation", bootstrap_samples)
        ranking = summarize_model_ranking(fidelity)
        ranking.to_csv(cross_platform_dir / "evaluation" / "model_ranking.csv", index=False)
        write_json(cross_platform_dir / "summary.json", summarize_cross_platform_fidelity(fidelity, targets, len(simulated)))
    return counts


def _scenario_grid(grid: dict[str, Any], base: RevisionSimulationConfig) -> list[tuple[str, RevisionSimulationConfig]]:
    scenarios = [("reference", base)]
    ignored = {"design", "seed"}
    for key, values in grid.items():
        if key in ignored or key not in base.__dataclass_fields__:
            continue
        for value in values:
            scenarios.append((f"{key}={value}", replace(base, **{key: value})))
    for trait in grid.get("trait_mode", []):
        for ranking in grid.get("ranking", []):
            for viewport in grid.get("viewport_k", []):
                scenarios.append((f"factorial|trait={trait}|ranking={ranking}|viewport={viewport}", replace(base, trait_mode=trait, ranking=ranking, viewport_k=viewport)))
    unique: dict[str, RevisionSimulationConfig] = {}
    for name, scenario in scenarios:
        unique[name] = scenario
    return list(unique.items())


def run_ablations_workflow(root: Path, config: dict[str, Any], grid: dict[str, Any], max_communities: int = 5) -> pd.DataFrame:
    paths = paths_for(root)
    empirical = pd.read_parquet(paths["audit"] / "empirical_cascade_metrics.parquet")
    models = load_models(paths["models"])
    train = empirical[empirical["split"] == "train"]
    test = empirical[empirical["split"] == "test"]
    base = RevisionSimulationConfig.from_dict(config.get("simulation", {}))
    records = []
    communities = list(COMMUNITIES)[:max_communities]
    for scenario_name, scenario in _scenario_grid(grid, base):
        for community in communities:
            reference = test[test["community"] == community].sort_values("post_id").head(1)
            if reference.empty:
                continue
            post_id = str(reference.iloc[0]["post_id"])
            for condition, conflict in (("baseline", 0.0), ("conflict", 1.0)):
                active = replace(scenario, conflict_intensity=conflict)
                nodes = simulate_cascade("learned_bdmtf", models[community], train[train["community"] == community], post_id, int(grid.get("seed", 30371)), active)
                records.append({"scenario": scenario_name, "condition": condition, "community": community, **active.__dict__, **event_metrics(nodes)})
    frame = pd.DataFrame(records)
    paths["ablations"].mkdir(parents=True, exist_ok=True)
    frame.to_csv(paths["ablations"] / "ablation_metrics.csv", index=False)
    baseline = frame[frame["condition"] == "baseline"].set_index(["scenario", "community"])
    conflict = frame[frame["condition"] == "conflict"].set_index(["scenario", "community"])
    comparison = pd.DataFrame(
        {
            "volume_ratio": conflict["size"] / baseline["size"].clip(lower=1),
            "leaf_depth_delta": conflict["mean_leaf_depth"] - baseline["mean_leaf_depth"],
        }
    ).reset_index()
    comparison["shallow_swarm"] = (comparison["volume_ratio"] > 1.0) & (comparison["leaf_depth_delta"] < 0.0)
    comparison["reversal_or_null"] = ~comparison["shallow_swarm"]
    comparison.to_csv(paths["ablations"] / "tradeoff_regions.csv", index=False)
    return frame


def api_workflow(root: Path, execute: bool = False) -> dict[str, Any]:
    paths = paths_for(root)
    manifest = generate_intents(paths["social"], paths["splits"] / "post_splits.csv", root / "configs" / "api_robustness.json", paths["api"], execute)
    if execute and manifest.get("status") == "complete":
        _replay_api_intents(root)
    return manifest


def _replay_api_intents(root: Path) -> pd.DataFrame:
    paths = paths_for(root)
    intents = load_intent_response_frame(
        paths["api"] / "frozen_intents_multimodel.jsonl"
    )
    post_summary = (
        intents.groupby(
            ["family", "model", "community", "post_id"],
            sort=True,
        )[["reply", "antagonistic"]]
        .mean()
        .reset_index()
    )
    empirical = pd.read_parquet(paths["audit"] / "empirical_cascade_metrics.parquet")
    train = empirical[empirical["split"] == "train"]
    models = load_models(paths["models"])
    base_config = RevisionSimulationConfig.from_dict(load_config(root / "configs" / "revision_experiment.json").get("simulation", {}))
    global_reply = max(0.05, float(intents["reply"].mean()))
    strict_records = []
    coupled_records = []
    for item in post_summary.itertuples(index=False):
        post_seed = 30371 ^ int(
            hashlib.sha256(str(item.post_id).encode("utf-8")).hexdigest()[:8],
            16,
        )
        strict_nodes = simulate_cascade(
            "learned_bdmtf",
            models[item.community],
            train[train["community"] == item.community],
            str(item.post_id),
            post_seed,
            base_config,
        )
        strict_records.append(
            {
                "protocol": "strict_shared_dynamics",
                "family": item.family,
                "model": item.model,
                "community": item.community,
                "post_id": str(item.post_id),
                "seed": post_seed,
                "frozen_reply_rate": item.reply,
                "frozen_antagonistic_rate": item.antagonistic,
                **event_metrics(strict_nodes),
            }
        )
        coupled_config = replace(
            base_config,
            activation_multiplier=float(max(0.25, min(2.0, item.reply / global_reply))),
            conflict_intensity=float(max(0.0, min(2.0, item.antagonistic / 0.1))),
        )
        coupled_nodes = simulate_cascade(
            "learned_bdmtf",
            models[item.community],
            train[train["community"] == item.community],
            str(item.post_id),
            post_seed,
            coupled_config,
        )
        coupled_records.append(
            {
                "protocol": "intent_coupled_diagnostic",
                "family": item.family,
                "model": item.model,
                "community": item.community,
                "post_id": str(item.post_id),
                "seed": post_seed,
                "frozen_reply_rate": item.reply,
                "frozen_antagonistic_rate": item.antagonistic,
                **event_metrics(coupled_nodes),
            }
        )
    strict = pd.DataFrame(strict_records)
    coupled = pd.DataFrame(coupled_records)
    metrics = ["size", "max_depth", "mean_leaf_depth", "toxicity_density"]
    strict.to_csv(paths["api"] / "replay_metrics.csv", index=False)
    strict.groupby(["family", "model", "community"])[metrics].mean().reset_index().to_csv(
        paths["api"] / "replay_summary.csv",
        index=False,
    )
    coupled.to_csv(
        paths["api"] / "coupled_replay_metrics.csv",
        index=False,
    )
    coupled.groupby(["family", "model", "community"])[metrics].mean().reset_index().to_csv(
        paths["api"] / "coupled_replay_summary.csv",
        index=False,
    )
    return strict


def provenance_workflow(root: Path) -> dict[str, Any]:
    paths = paths_for(root)
    payload = tree_manifest(root / "data" / "social_paper")
    write_json(paths["provenance"] / "data_manifest.json", payload)
    return payload


def _resolved(root: Path, configured: str | Path) -> Path:
    path = Path(configured)
    return path if path.is_absolute() else root / path


def fetch_tbbt_workflow(
    root: Path,
    config: dict[str, Any],
    execute: bool = False,
    import_data: bool = False,
    max_records_per_archive: int = 0,
) -> dict[str, Any]:
    section = config.get("tbbt", {})
    raw_root = _resolved(root, section.get("raw_root", "data/external/tbbt/raw"))
    manifest = fetch_tbbt(raw_root, execute, section.get("categories"))
    if import_data:
        manifest["import"] = import_tbbt(
            raw_root,
            paths_for(root)["interventions"] / "tbbt",
            max_records_per_archive,
        )
    return manifest


def collect_lemmy_workflow(root: Path, config: dict[str, Any], pages: int | None = None) -> dict[str, Any]:
    section = config.get("lemmy", {})
    instances = section.get("instances", ["https://lemmy.world"])
    return collect_lemmy(
        _resolved(root, section.get("output", "data/external/lemmy")),
        instances,
        int(pages if pages is not None else section.get("pages", 1)),
        int(section.get("page_size", 50)),
        int(section.get("timeout", 60)),
        float(section.get("sleep_seconds", 0.2)),
    )


def collect_lemmy_outcomes_workflow(
    root: Path,
    config: dict[str, Any],
    max_treated_posts: int = 0,
) -> dict[str, Any]:
    section = config.get("lemmy", {})
    output = _resolved(root, section.get("output", "data/external/lemmy"))
    interventions = output / "lemmy_interventions.parquet"
    if not interventions.is_file():
        raise FileNotFoundError("Run collect-lemmy before collect-lemmy-outcomes")
    outcomes_output = _resolved(
        root,
        section.get(
            "outcomes_output",
            str(Path(section.get("output", "data/external/lemmy")) / "outcomes"),
        ),
    )
    return collect_lemmy_outcomes(
        str(section.get("instances", ["https://lemmy.world"])[0]),
        interventions,
        outcomes_output,
        int(section.get("community_pages", 5)),
        max_treated_posts,
        int(section.get("comment_pages", 20)),
        int(section.get("page_size", 50)),
        int(section.get("workers", 8)),
        section.get("intervention_types", ["remove_post", "lock_post"]),
        bool(section.get("deduplicate_content", False)),
        section.get("matching", {}),
        int(section.get("pre_days", 7)),
        int(section.get("post_days", 7)),
    )


def expand_reddit_workflow(root: Path, config: dict[str, Any], inputs: Iterable[str] | None = None) -> dict[str, Any]:
    section = config.get("reddit_expansion", {})
    archive_inputs = list(section.get("archive_inputs", []))
    if archive_inputs and inputs is None:
        return import_reddit_archive_bundles(
            [_resolved(root, value) for value in archive_inputs],
            paths_for(root)["external"] / "reddit_expansion",
            section.get("communities", []),
            int(section.get("minimum_communities", 20)),
            int(section.get("minimum_cascades", 10_000)),
            section.get("community_groups", {}),
        )
    configured = list(inputs or section.get("inputs", []))
    resolved = [_resolved(root, value) for value in configured]
    return import_reddit_expansion(
        resolved,
        paths_for(root)["external"] / "reddit_expansion",
        section.get("column_map", {}),
    )


def reddit_expansion_validation_workflow(
    root: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    section = config.get("reddit_expansion", {})
    return run_reddit_expansion_validation(
        paths_for(root)["external"] / "reddit_expansion",
        section.get("validation", {}),
    )


def _historical_reddit_roots(root: Path) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for path in sorted((root / "data" / "social_paper").glob("*_data/posts_features_*.csv")):
        posts = pd.read_csv(path, encoding="utf-8", encoding_errors="replace")
        for row in posts.itertuples(index=False):
            post_id = str(getattr(row, "post_id"))
            community = str(getattr(row, "subreddit", path.parent.name.removesuffix("_data")))
            records.append(
                {
                    "platform": "reddit",
                    "community": community,
                    "content_id": post_id,
                    "event_id": f"reddit:post:{post_id}",
                    "parent_event_id": "",
                    "created_at": getattr(row, "created_utc"),
                    "author_id": str(getattr(row, "author", "")),
                    "text": str(getattr(row, "title", "")),
                    "title": str(getattr(row, "title", "")),
                    "score": float(getattr(row, "final_score", 0) or 0),
                    "event_type": "post",
                    "removed": False,
                    "content_url": str(getattr(row, "url", "") or ""),
                }
            )
    return pd.DataFrame(records)


def _load_story_events(root: Path, config: dict[str, Any]) -> pd.DataFrame:
    frames = [_historical_reddit_roots(root)]
    section = config.get("story_matching", {})
    default_inputs = [
        "data/external/hackernews/hackernews_events.parquet",
        "data/external/lemmy/lemmy_post_events.parquet",
        "artifacts/external_validation/reddit_expansion/reddit_expanded_events.parquet",
    ]
    for configured in section.get("inputs", default_inputs):
        path = _resolved(root, configured)
        if path.is_file():
            frames.append(read_table(path))
    available = [frame for frame in frames if not frame.empty]
    if not available:
        raise FileNotFoundError("No story-event sources are available")
    combined = pd.concat(available, ignore_index=True, sort=False)
    return combined.drop_duplicates(["platform", "event_id"], keep="first").reset_index(drop=True)


def build_story_matches_workflow(
    root: Path,
    config: dict[str, Any],
    resolve_urls: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    section = config.get("story_matching", {})
    return build_story_matches(
        _load_story_events(root, config),
        paths_for(root)["stories"],
        float(section.get("semantic_threshold", 0.72)),
        float(section.get("max_hours", 72.0)),
        resolve_urls,
        int(section.get("resolution_workers", 8)),
        float(section.get("resolution_timeout_seconds", 10.0)),
    )


def make_prospective_split_workflow(
    root: Path,
    config: dict[str, Any],
    freeze_at: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    section = config.get("prospective", {})
    active_freeze = freeze_at or section.get("freeze_at")
    if not active_freeze:
        output = paths_for(root)["prospective"]
        output.mkdir(parents=True, exist_ok=True)
        result = {
            "status": "not_frozen",
            "resolved": False,
            "claim_allowed": False,
            "reason": "A prospective freeze timestamp must be declared before inspecting test-period data.",
            "configured_weeks": int(section.get("weeks", 8)),
        }
        write_json(output / "prospective_manifest.json", result)
        return pd.DataFrame(), result
    events = _load_story_events(root, config)
    return make_prospective_split(
        events,
        paths_for(root)["prospective"],
        str(active_freeze),
        int(section.get("weeks", 8)),
    )


def _intervention_panel(root: Path, config: dict[str, Any]) -> tuple[Path, pd.DataFrame]:
    section = config.get("causal_analysis", {})
    configured = str(section.get("panel_path", "")).strip()
    if not configured:
        raise FileNotFoundError("No causal_analysis.panel_path is configured")
    path = _resolved(root, configured)
    if not path.is_file():
        raise FileNotFoundError(f"Intervention panel does not exist: {path}")
    return path, read_table(path)


def fit_intervention_models_workflow(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    source, panel = _intervention_panel(root, config)
    section = config.get("causal_analysis", {})
    output = paths_for(root)["interventions"] / str(section.get("forecast_output_name", "forecast"))
    output.mkdir(parents=True, exist_ok=True)
    models = fit_pre_intervention_forecast(panel)
    models.to_csv(output / "pre_intervention_models.csv", index=False)
    manifest = {
        "status": "complete",
        "source": str(source),
        "source_sha256": sha256_file(source),
        "n_models": int(len(models)),
        "training_rule": "relative_period < 0 only",
    }
    write_json(output / "fit_manifest.json", manifest)
    return manifest


def run_natural_experiments_workflow(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    section = config.get("causal_analysis", {})
    output = paths_for(root)["interventions"] / str(section.get("output_name", "natural"))
    try:
        source, panel = _intervention_panel(root, config)
    except FileNotFoundError as exc:
        result = {"status": "needs_data", "resolved": False, "claim_allowed": False, "reason": str(exc)}
        write_json(output / "natural_experiment_summary.json", result)
        return result
    result = run_natural_experiments(
        panel,
        output,
        int(section.get("bootstrap_samples", 1000)),
        int(section.get("seed", 30371)),
        _resolved(root, section.get("r_script", "r/run_causal_analysis.R")),
        bool(section.get("run_synthetic_control", True)),
        section.get("aggregate_synthetic_control"),
    )
    if bool(section.get("confirmatory", False)) and result.get("status") == "complete":
        gates = section.get("claim_gates", {})
        required_outcomes = set(section.get("required_outcomes", []))
        observed_outcomes = set(panel["outcome"].astype(str))
        primary_estimator = str(
            section.get("primary_estimator", "did")
        )
        if (
            primary_estimator
            == "aggregate_preoutcome_calibrated_synthetic_control"
        ):
            aggregate = (
                result.get("aggregate_synthetic_control") or {}
            )
            diagnostics = aggregate.get("diagnostics", [])
        else:
            aggregate = {}
            diagnostics = result.get("diagnostics", [])
        minimum_pairs = int(gates.get("minimum_matched_pairs", 0))
        pair_count = (
            int(panel["intervention_id"].astype(str).nunique())
            if "intervention_id" in panel
            else min(
                int(
                    panel.loc[
                        panel["treated"].astype(int).eq(1),
                        "unit_id",
                    ].nunique()
                ),
                int(
                    panel.loc[
                        panel["treated"].astype(int).eq(0),
                        "unit_id",
                    ].nunique()
                ),
            )
        )
        gate_failures: list[str] = []
        if bool(gates.get("require_source_status_complete", False)):
            configured_manifest = str(
                section.get("source_manifest_path", "")
            ).strip()
            manifest_path = (
                _resolved(root, configured_manifest)
                if configured_manifest
                else None
            )
            if manifest_path is None or not manifest_path.is_file():
                gate_failures.append(
                    "confirmatory source manifest is missing"
                )
            else:
                source_manifest = json.loads(
                    manifest_path.read_text(encoding="utf-8")
                )
                source_status = str(
                    source_manifest.get("status", "unknown")
                )
                if source_status != "complete":
                    gate_failures.append(
                        "confirmatory source status "
                        f"{source_status!r} is not 'complete'"
                    )
        if pair_count < minimum_pairs:
            gate_failures.append(
                f"matched pairs {pair_count} < required {minimum_pairs}"
            )
        if (
            bool(gates.get("require_all_frozen_outcomes", False))
            and not required_outcomes.issubset(observed_outcomes)
        ):
            missing = sorted(required_outcomes.difference(observed_outcomes))
            gate_failures.append(
                "missing frozen outcomes: " + ", ".join(missing)
            )
        minimum_overlap = float(
            gates.get("minimum_treated_overlap", 0.0)
        )
        for diagnostic in diagnostics:
            outcome = str(diagnostic.get("outcome", "unknown"))
            overlap = float(
                diagnostic.get("treated_fraction_in_overlap", 0.0)
            )
            if overlap < minimum_overlap:
                gate_failures.append(
                    f"{outcome} treated overlap {overlap:.3f} "
                    f"< required {minimum_overlap:.3f}"
                )
            if (
                bool(gates.get("require_no_pretrend_flag", False))
                and bool(diagnostic.get("pretrend_flag", False))
            ):
                gate_failures.append(f"{outcome} failed pretrend gate")
        if (
            primary_estimator
            == "aggregate_preoutcome_calibrated_synthetic_control"
        ):
            if not aggregate or aggregate.get("status") != "complete":
                gate_failures.append(
                    "aggregate calibrated synthetic control is incomplete"
                )
            maximum_weight = float(
                aggregate.get("maximum_control_weight", 1.0)
            )
            configured_maximum_weight = float(
                gates.get("maximum_control_weight", 1.0)
            )
            if maximum_weight > configured_maximum_weight:
                gate_failures.append(
                    f"maximum control weight {maximum_weight:.3f} "
                    f"> allowed {configured_maximum_weight:.3f}"
                )
            effective_controls = float(
                aggregate.get("effective_controls", 0.0)
            )
            minimum_effective_controls = float(
                gates.get("minimum_effective_controls", 0.0)
            )
            if effective_controls < minimum_effective_controls:
                gate_failures.append(
                    f"effective controls {effective_controls:.1f} "
                    f"< required {minimum_effective_controls:.1f}"
                )
            pre_rmse = float(
                aggregate.get("standardized_pre_rmse", float("inf"))
            )
            maximum_pre_rmse = float(
                gates.get(
                    "maximum_standardized_pre_rmse",
                    float("inf"),
                )
            )
            if pre_rmse > maximum_pre_rmse:
                gate_failures.append(
                    f"standardized pre-RMSE {pre_rmse:.3f} "
                    f"> allowed {maximum_pre_rmse:.3f}"
                )
        result.update(
            {
                "confirmatory": True,
                "primary_estimator": primary_estimator,
                "primary_diagnostics": diagnostics,
                "matched_pairs": pair_count,
                "claim_gates": gates,
                "gate_failures": gate_failures,
                "status": (
                    "confirmatory_complete"
                    if not gate_failures
                    else "confirmatory_diagnostics_failed"
                ),
                "resolved": not gate_failures,
                "claim_allowed": not gate_failures,
            }
        )
    elif not bool(section.get("confirmatory", False)) and result.get("status") == "complete":
        result.update(
            {
                "status": "pilot_complete",
                "resolved": False,
                "claim_allowed": False,
                "reason": "The configured panel is a pipeline pilot, not the frozen confirmatory intervention sample.",
            }
        )
    elif not bool(section.get("confirmatory", False)):
        result["resolved"] = False
        result["claim_allowed"] = False
        result["confirmatory"] = False
    result["source"] = str(source)
    result["source_sha256"] = sha256_file(source)
    write_json(output / "natural_experiment_summary.json", result)
    return result


def run_human_rct_analysis_workflow(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    section = config.get("human_rct", config)
    configured = str(section.get("data_path", "")).strip()
    output = paths_for(root)["human_rct"]
    if not configured or not _resolved(root, configured).is_file():
        result = {
            "status": "blocked",
            "resolved": False,
            "claim_allowed": False,
            "reason": "No completed, ethics-approved human RCT dataset is configured.",
            "external_prerequisites": [
                "ethics approval",
                "preregistration",
                "participant recruitment",
                "informed consent",
            ],
        }
        write_json(output / "rct_summary.json", result)
        return result
    return analyze_human_rct(read_table(_resolved(root, configured)), section, output)


def evaluate_intervention_fidelity_workflow(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    section = config.get("intervention_fidelity", {})
    prediction_path = _resolved(root, section.get("predictions_path", "artifacts/interventions/predictions.csv"))
    observed_path = _resolved(root, section.get("observed_path", "artifacts/interventions/natural/did_estimates.csv"))
    output = paths_for(root)["interventions"] / "fidelity"
    if not prediction_path.is_file() or not observed_path.is_file():
        result = {
            "status": "needs_data",
            "claim_allowed": False,
            "predictions_available": prediction_path.is_file(),
            "observed_available": observed_path.is_file(),
        }
        write_json(output / "intervention_fidelity.json", result)
        return result
    return evaluate_intervention_fidelity(read_table(prediction_path), read_table(observed_path), output)


def lemmy_intervention_fidelity_workflow(
    root: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    inputs = config.get("inputs", {})
    return run_lemmy_intervention_fidelity(
        _resolved(
            root,
            inputs.get(
                "panel_path",
                "data/external/lemmy/outcomes_confirmatory/lemmy_outcome_panel.parquet",
            ),
        ),
        _resolved(
            root,
            inputs.get(
                "matches_path",
                "data/external/lemmy/outcomes_confirmatory/risk_set_matches.csv",
            ),
        ),
        _resolved(
            root,
            inputs.get(
                "natural_experiment_path",
                "artifacts/interventions/natural_confirmatory_lemmy_calibrated/"
                "natural_experiment_summary.json",
            ),
        ),
        _resolved(
            root,
            config.get(
                "output_dir",
                "artifacts/interventions/fidelity_lemmy",
            ),
        ),
        config,
    )
