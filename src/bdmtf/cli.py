from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data.social_loader import extract_cascade_features, load_comments, resolve_community_paths
from .experiments import config_from_dict, interventions_from_config, run_batch
from .intent_pool import FrozenIntentPool
from .revision.hackernews import collect_hackernews
from .revision.lemmy_agent_replay import run_lemmy_agent_replay
from .revision.lemmy_content_matched_validation import (
    run_lemmy_content_matched_validation,
)
from .revision.workflows import (
    api_workflow,
    audit_external_workflow,
    audit_workflow,
    build_story_matches_workflow,
    collect_lemmy_workflow,
    collect_lemmy_outcomes_workflow,
    cross_community_workflow,
    cross_platform_workflow,
    evaluate_workflow,
    evaluate_intervention_fidelity_workflow,
    expand_reddit_workflow,
    fetch_tbbt_workflow,
    fit_intervention_models_workflow,
    fit_workflow,
    load_config,
    lemmy_intervention_fidelity_workflow,
    make_prospective_split_workflow,
    make_splits_workflow,
    provenance_workflow,
    recompute_saved_metrics_workflow,
    reddit_expansion_validation_workflow,
    run_human_rct_analysis_workflow,
    run_natural_experiments_workflow,
    run_ablations_workflow,
    run_suite_workflow,
)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="BDMTF empirical revision and legacy reproduction tools")
    sub = parser.add_subparsers(dest="cmd", required=True)

    legacy = sub.add_parser("run", help="Run the legacy paper-aligned simulator")
    legacy.add_argument("--social-root", required=True)
    legacy.add_argument("--output", required=True)
    legacy.add_argument("--config", default=None)
    legacy.add_argument("--communities", nargs="+", default=None)
    legacy.add_argument("--posts-per-community", type=int, default=None)
    legacy.add_argument("--seeds", nargs="+", type=int, default=None)

    intents = sub.add_parser("build-intents", help="Build a frozen intent pool from existing comments")
    intents.add_argument("--social-root", required=True)
    intents.add_argument("--community", required=True)
    intents.add_argument("--output", required=True)
    intents.add_argument("--max-intents", type=int, default=2000)

    cascades = sub.add_parser("extract-cascades", help="Extract legacy empirical cascade statistics")
    cascades.add_argument("--social-root", required=True)
    cascades.add_argument("--community", required=True)
    cascades.add_argument("--output", required=True)
    cascades.add_argument("--max-posts", type=int, default=0)

    for name, help_text in (
        ("make-splits", "Create leakage-safe chronological train/validation/test splits"),
        ("audit-data", "Audit all 500 posts and reconstruct empirical cascade metrics"),
        ("fit-models", "Fit aggregate event and target parameters using train data only"),
        ("run-baselines", "Run empirical, branching, Hawkes, and legacy baselines"),
        ("run-main", "Run the learned BDMTF model on the held-out test split"),
        ("run-ablations", "Run reviewer-requested sensitivity and falsification experiments"),
        ("generate-intents", "Prepare or execute three-family API intent generation"),
        ("evaluate-fidelity", "Compare simulations with held-out empirical distributions"),
        ("audit-external-data", "Validate cross-platform and real-intervention inputs without making unsupported claims"),
        ("collect-hackernews", "Collect a cached cross-platform snapshot from the official Hacker News API"),
        ("run-cross-community", "Run leave-one-community-out zero-shot validation"),
        ("run-cross-platform", "Fit on Reddit and evaluate zero-shot transfer to an audited external platform"),
        ("recompute-saved-metrics", "Recompute run summaries from event Parquet after metric-definition changes"),
        ("fetch-tbbt", "Plan, resume, verify, and optionally import the public TBBT archives"),
        ("collect-lemmy", "Collect public Lemmy moderation events and timestamped target posts"),
        ("collect-lemmy-outcomes", "Collect treated threads, risk-set controls, and a Lemmy outcome panel"),
        ("expand-reddit", "Normalize user-supplied Reddit exports for the 20-community expansion"),
        ("run-reddit-expansion-validation", "Evaluate five cascade models across the expanded Reddit communities"),
        ("build-story-matches", "Build separate exact-URL and semantic cross-platform story matches"),
        ("make-prospective-split", "Freeze an eight-week prospective test window"),
        ("fit-intervention-models", "Fit pre-intervention-only forecasting models"),
        ("run-natural-experiments", "Run audited DiD, event-study, and ITS analyses"),
        ("run-human-rct-analysis", "Audit and analyze a completed preregistered human experiment"),
        ("serve-human-rct", "Serve the ethics-gated private randomized thread experiment"),
        ("evaluate-intervention-fidelity", "Compare frozen simulation predictions with observed intervention effects"),
        ("run-lemmy-intervention-fidelity", "Run chronological Lemmy intervention-response prediction validation"),
        ("run-lemmy-agent-replay", "Replay held-out Lemmy interventions with generated agent events"),
        ("run-lemmy-content-matched-validation", "Validate Lemmy portability on content-matched Hacker News stories"),
        ("build-original-manuscript", "Build the sanitized text-transcription fallback"),
        ("build-paper-redline", "Build a reference redline against the text transcription"),
        ("build-final-paper", "Compile the traced revision based on the user-designated original PDF"),
        ("build-revision-package", "Build provenance and the complete revision report package"),
    ):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("--root", default=str(_repository_root()))
        command.add_argument("--config", default=None)
        if name in {"run-baselines", "run-main"}:
            command.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
            command.add_argument("--max-posts-per-community", type=int, default=0)
        if name == "run-ablations":
            command.add_argument("--grid", default=None)
            command.add_argument("--max-communities", type=int, default=5)
        if name == "generate-intents":
            command.add_argument("--execute", action="store_true")
        if name == "evaluate-fidelity":
            command.add_argument("--bootstrap-samples", type=int, default=400)
        if name in {"run-cross-community", "run-cross-platform"}:
            command.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
            command.add_argument("--bootstrap-samples", type=int, default=400)
        if name == "run-cross-community":
            command.add_argument("--max-posts-per-community", type=int, default=0)
        if name == "run-cross-platform":
            command.add_argument("--max-cascades", type=int, default=0)
        if name == "recompute-saved-metrics":
            command.add_argument("--bootstrap-samples", type=int, default=400)
        if name == "collect-hackernews":
            command.add_argument("--stories", type=int, default=50)
            command.add_argument("--max-comments-per-story", type=int, default=300)
            command.add_argument("--feed", choices=["topstories", "newstories", "beststories"], default="topstories")
            command.add_argument("--workers", type=int, default=8)
            command.add_argument(
                "--historical-scan-multiplier",
                type=int,
                default=8,
                help="Maximum reverse maxitem candidates per requested story when the live feed is too small",
            )
            command.add_argument(
                "--discovery",
                choices=["auto", "algolia", "official_maxitem"],
                default="auto",
                help="Historical story discovery backend; comments always come from official Firebase",
            )
        if name == "fetch-tbbt":
            command.add_argument("--execute", action="store_true")
            command.add_argument("--import-data", action="store_true")
            command.add_argument("--max-records-per-archive", type=int, default=0)
        if name == "collect-lemmy":
            command.add_argument("--pages", type=int, default=None)
        if name == "collect-lemmy-outcomes":
            command.add_argument("--max-treated-posts", type=int, default=0)
        if name == "expand-reddit":
            command.add_argument("--inputs", nargs="*", default=None)
        if name == "build-story-matches":
            command.add_argument(
                "--resolve-urls",
                action="store_true",
                help="Resolve public redirects and HTML canonical links into the offline cache",
            )
        if name == "make-prospective-split":
            command.add_argument("--freeze-at", default=None)
        if name == "serve-human-rct":
            command.add_argument("--host", default="127.0.0.1")
            command.add_argument("--port", type=int, default=8765)
            command.add_argument("--database", default=None)

    args = parser.parse_args()
    if args.cmd == "run":
        _legacy_run(args)
    elif args.cmd == "build-intents":
        _build_intents(args)
    elif args.cmd == "extract-cascades":
        _extract_cascades(args)
    else:
        _revision_command(args)


def _revision_command(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    external_commands = {
        "audit-external-data",
        "collect-hackernews",
        "run-cross-community",
        "run-cross-platform",
        "fetch-tbbt",
        "collect-lemmy",
        "collect-lemmy-outcomes",
        "expand-reddit",
        "run-reddit-expansion-validation",
        "build-story-matches",
        "make-prospective-split",
    }
    causal_commands = {
        "fit-intervention-models",
        "run-natural-experiments",
        "evaluate-intervention-fidelity",
    }
    default_config = (
        "external_sources.json"
        if args.cmd in external_commands
        else "intervention_analysis.json"
        if args.cmd in causal_commands
        else "lemmy_intervention_fidelity.json"
        if args.cmd == "run-lemmy-intervention-fidelity"
        else "lemmy_agent_intervention_replay.json"
        if args.cmd == "run-lemmy-agent-replay"
        else "lemmy_content_matched_validation.json"
        if args.cmd == "run-lemmy-content-matched-validation"
        else "human_rct.json"
        if args.cmd in {"run-human-rct-analysis", "serve-human-rct"}
        else "revision_experiment.json"
    )
    config_path = Path(args.config) if args.config else root / "configs" / default_config
    config = load_config(config_path)
    if args.cmd == "make-splits":
        frame = make_splits_workflow(root, config)
        print(f"Created splits for {len(frame)} posts")
    elif args.cmd == "audit-data":
        frame, audit = audit_workflow(root, config)
        print(f"Audited {audit['n_posts']} posts; reconstructed {len(frame)} cascades")
    elif args.cmd == "fit-models":
        payload = fit_workflow(root)
        print(f"Fitted {len(payload['communities'])} community models from training data")
    elif args.cmd == "run-baselines":
        metrics, _ = run_suite_workflow(root, config, ("empirical_bootstrap", "branching_process", "hawkes", "legacy_heuristic"), "baselines", args.seeds, args.max_posts_per_community)
        print(f"Completed {len(metrics)} baseline simulations")
    elif args.cmd == "run-main":
        metrics, _ = run_suite_workflow(root, config, ("learned_bdmtf",), "main", args.seeds, args.max_posts_per_community)
        print(f"Completed {len(metrics)} learned-model simulations")
    elif args.cmd == "run-ablations":
        grid_path = Path(args.grid) if args.grid else root / "configs" / "ablation_grid.json"
        frame = run_ablations_workflow(root, config, load_config(grid_path), args.max_communities)
        print(f"Completed {len(frame)} paired ablation simulations")
    elif args.cmd == "generate-intents":
        manifest = api_workflow(root, execute=args.execute)
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    elif args.cmd == "evaluate-fidelity":
        frame = evaluate_workflow(root, bootstrap_samples=args.bootstrap_samples)
        print(f"Computed {len(frame)} held-out fidelity comparisons")
    elif args.cmd == "audit-external-data":
        audit = audit_external_workflow(root, config)
        print(json.dumps(audit, ensure_ascii=False, indent=2))
    elif args.cmd == "collect-hackernews":
        _, manifest = collect_hackernews(
            root / "data" / "external" / "hackernews",
            args.stories,
            args.max_comments_per_story,
            args.feed,
            args.workers,
            historical_scan_multiplier=args.historical_scan_multiplier,
            discovery=args.discovery,
        )
        print(
            json.dumps(
                {
                    key: manifest.get(key)
                    for key in (
                        "status",
                        "data_complete",
                        "selection_mode",
                        "story_count_requested",
                        "story_count_collected",
                        "target_met",
                        "structurally_complete_stories",
                        "incomplete_stories_excluded",
                        "temporally_reparented_stories_excluded",
                        "request_failure_count",
                        "event_count",
                        "events_sha256",
                    )
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    elif args.cmd == "run-cross-community":
        metrics, fidelity = cross_community_workflow(
            root,
            config,
            args.seeds,
            args.max_posts_per_community,
            args.bootstrap_samples,
        )
        print(f"Completed {len(metrics)} zero-shot simulations and {len(fidelity)} fidelity comparisons")
    elif args.cmd == "run-cross-platform":
        metrics, fidelity = cross_platform_workflow(
            root,
            config,
            args.seeds,
            args.max_cascades,
            args.bootstrap_samples,
        )
        print(f"Completed {len(metrics)} cross-platform simulations and {len(fidelity)} fidelity comparisons")
    elif args.cmd == "recompute-saved-metrics":
        counts = recompute_saved_metrics_workflow(root, args.bootstrap_samples)
        print(json.dumps(counts, ensure_ascii=False, indent=2))
    elif args.cmd == "fetch-tbbt":
        result = fetch_tbbt_workflow(
            root,
            config,
            args.execute,
            args.import_data,
            args.max_records_per_archive,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.cmd == "collect-lemmy":
        print(json.dumps(collect_lemmy_workflow(root, config, args.pages), ensure_ascii=False, indent=2))
    elif args.cmd == "collect-lemmy-outcomes":
        print(
            json.dumps(
                collect_lemmy_outcomes_workflow(root, config, args.max_treated_posts),
                ensure_ascii=False,
                indent=2,
            )
        )
    elif args.cmd == "expand-reddit":
        print(json.dumps(expand_reddit_workflow(root, config, args.inputs), ensure_ascii=False, indent=2))
    elif args.cmd == "run-reddit-expansion-validation":
        print(json.dumps(reddit_expansion_validation_workflow(root, config), ensure_ascii=False, indent=2))
    elif args.cmd == "build-story-matches":
        _, result = build_story_matches_workflow(root, config, args.resolve_urls)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.cmd == "make-prospective-split":
        _, result = make_prospective_split_workflow(root, config, args.freeze_at)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.cmd == "fit-intervention-models":
        print(json.dumps(fit_intervention_models_workflow(root, config), ensure_ascii=False, indent=2))
    elif args.cmd == "run-natural-experiments":
        print(json.dumps(run_natural_experiments_workflow(root, config), ensure_ascii=False, indent=2))
    elif args.cmd == "run-human-rct-analysis":
        print(json.dumps(run_human_rct_analysis_workflow(root, config), ensure_ascii=False, indent=2))
    elif args.cmd == "serve-human-rct":
        from .revision.human_experiment import serve_human_experiment

        database = Path(args.database).resolve() if args.database else None
        serve_human_experiment(
            root,
            config.get("human_rct", config),
            args.host,
            args.port,
            database,
        )
    elif args.cmd == "evaluate-intervention-fidelity":
        print(json.dumps(evaluate_intervention_fidelity_workflow(root, config), ensure_ascii=False, indent=2))
    elif args.cmd == "run-lemmy-intervention-fidelity":
        print(json.dumps(lemmy_intervention_fidelity_workflow(root, config), ensure_ascii=False, indent=2))
    elif args.cmd == "run-lemmy-agent-replay":
        print(
            json.dumps(
                run_lemmy_agent_replay(root, config),
                ensure_ascii=False,
                indent=2,
            )
        )
    elif args.cmd == "run-lemmy-content-matched-validation":
        print(
            json.dumps(
                run_lemmy_content_matched_validation(root, config),
                ensure_ascii=False,
                indent=2,
            )
        )
    elif args.cmd in {"build-original-manuscript", "build-paper-redline", "build-final-paper"}:
        from .revision.manuscript import run_manuscript_command

        print(json.dumps(run_manuscript_command(root, args.cmd), ensure_ascii=False, indent=2))
    elif args.cmd == "build-revision-package":
        provenance = provenance_workflow(root)
        from .revision.reporting import build_revision_package

        build_revision_package(root)
        print(f"Revision package built; data hash {provenance['tree_sha256']}")


def _load_legacy_config(path: str | None) -> dict:
    if path:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    return {}


def _legacy_run(args: argparse.Namespace) -> None:
    raw = _load_legacy_config(args.config)
    config = config_from_dict(raw)
    communities = args.communities or raw.get("communities") or ["funny"]
    posts_per_community = args.posts_per_community or int(raw.get("posts_per_community", 5))
    seeds = args.seeds or raw.get("seeds") or [0]
    records = run_batch(args.social_root, args.output, communities, posts_per_community, seeds, config, interventions_from_config(raw))
    print(f"Completed {len(records)} legacy runs. Output: {args.output}")


def _build_intents(args: argparse.Namespace) -> None:
    paths = resolve_community_paths(args.social_root, args.community)
    pool = FrozenIntentPool.from_comments(load_comments(paths), max_intents=args.max_intents)
    pool.save_jsonl(args.output)
    print(f"Frozen intents written to {args.output}")


def _extract_cascades(args: argparse.Namespace) -> None:
    paths = resolve_community_paths(args.social_root, args.community)
    comments = load_comments(paths)
    post_ids = comments["post_id"].drop_duplicates().tolist()
    if args.max_posts > 0:
        post_ids = post_ids[: args.max_posts]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for post_id in post_ids:
            item = extract_cascade_features(comments, post_id)
            if item:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Fixed cascade features written to {output}")


if __name__ == "__main__":
    main()
