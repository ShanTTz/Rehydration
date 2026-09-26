from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bdmtf.platform_adapter_validation import (
    chronological_platform_split,
    run_platform_adapter_test,
    select_platform_adapter,
)
from bdmtf.revision.data_pipeline import METRIC_COLUMNS
from bdmtf.revision.evaluation import evaluate_fidelity, summarize_model_ranking
from bdmtf.revision.fitted_models import fit_pooled_model
from bdmtf.revision.provenance import sha256_file, write_json
from bdmtf.revision.simulation import (
    RevisionSimulationConfig,
    event_metrics,
    nodes_to_records,
    simulate_cascade,
)


ADAPTED_LABEL = "platform_adapted_learned_bdmtf"
FIXED_MODEL_LABELS = (
    "reddit_zero_shot_learned_bdmtf",
    "target_default_learned_bdmtf",
    "target_empirical_bootstrap",
    "target_branching_process",
    "target_hawkes",
)


def nested_validation_subset(
    platform_metrics: pd.DataFrame,
    budget: int,
) -> pd.DataFrame:
    if budget < 0:
        raise ValueError("Validation budget must be non-negative")
    validation = platform_metrics[
        platform_metrics["split"].astype(str).eq("validation")
    ].sort_values(["created_at", "community", "post_id"])
    if budget > len(validation):
        raise ValueError(
            f"Validation budget {budget} exceeds {len(validation)} cascades"
        )
    return validation.head(budget).copy()


def frame_with_validation_budget(
    platform_metrics: pd.DataFrame,
    budget: int,
) -> pd.DataFrame:
    subset = nested_validation_subset(platform_metrics, budget)
    non_validation = platform_metrics[
        ~platform_metrics["split"].astype(str).eq("validation")
    ]
    return pd.concat([non_validation, subset], ignore_index=True)


def validation_subset_hash(frame: pd.DataFrame) -> str:
    columns = ["community", "post_id", "created_at"]
    serial = frame.loc[:, columns].copy()
    serial["created_at"] = pd.to_datetime(
        serial["created_at"], utc=True, errors="coerce"
    ).astype(str)
    payload = serial.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def split_assignment_hash(frame: pd.DataFrame) -> str:
    serial = frame.loc[:, ["community", "post_id", "split"]].copy()
    serial = serial.sort_values(["community", "post_id", "split"])
    payload = serial.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def simulate_adapted_budget(
    platform_metrics: pd.DataFrame,
    selected_config: RevisionSimulationConfig,
    budget: int,
    seeds: Iterable[int],
    max_test_cascades: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    profile = fit_pooled_model(platform_metrics, (), "target_platform")
    train = platform_metrics[
        platform_metrics["split"].astype(str).eq("train")
    ].copy()
    test = platform_metrics[
        platform_metrics["split"].astype(str).eq("test")
    ].sort_values(["community", "created_at", "post_id"])
    if max_test_cascades > 0:
        test = test.groupby("community", group_keys=False).head(
            max_test_cascades
        )
    label = f"platform_adapted_n{budget}"
    metric_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    for target in test.itertuples(index=False):
        for seed in map(int, seeds):
            # The seed basis is intentionally common across budgets for paired tests.
            run_seed = seed + _stable_seed(
                str(target.post_id), ADAPTED_LABEL, "platform_test"
            )
            nodes = simulate_cascade(
                "learned_bdmtf",
                profile,
                train,
                str(target.post_id),
                run_seed,
                selected_config,
            )
            context = {
                "platform": str(target.platform),
                "community": str(target.community),
                "post_id": str(target.post_id),
                "model": label,
                "seed": seed,
                "split": "test",
                "validation_budget": int(budget),
            }
            metric_rows.append({**context, **event_metrics(nodes)})
            event_rows.extend(nodes_to_records(nodes, context))
    return pd.DataFrame(metric_rows), pd.DataFrame(event_rows)


def summarize_budget_curve(
    fidelity: pd.DataFrame,
    budgets: Iterable[int],
    selection_metrics: Iterable[str],
) -> pd.DataFrame:
    primary_metrics = set(selection_metrics)
    rows: list[dict[str, Any]] = []
    for budget in map(int, budgets):
        model = f"platform_adapted_n{budget}"
        selected = fidelity[fidelity["model"].astype(str).eq(model)]
        primary = selected[selected["metric"].isin(primary_metrics)]
        rows.append(
            {
                "validation_budget": budget,
                "model": model,
                "all_metric_median_distance": _median_distance(selected),
                "primary_metric_median_distance": _median_distance(primary),
                "all_metric_mean_distance": _mean_distance(selected),
                "primary_metric_mean_distance": _mean_distance(primary),
                "n_all_metrics": int(selected["metric"].nunique()),
                "n_primary_metrics": int(primary["metric"].nunique()),
            }
        )
    result = pd.DataFrame(rows).sort_values("validation_budget")
    fixed = (
        fidelity.groupby("model")["normalized_wasserstein"]
        .median()
        .to_dict()
    )
    default_distance = float(
        fixed.get("target_default_learned_bdmtf", np.nan)
    )
    zero_shot_distance = float(
        fixed.get("reddit_zero_shot_learned_bdmtf", np.nan)
    )
    result["target_default_median_distance"] = default_distance
    result["reddit_zero_shot_median_distance"] = zero_shot_distance
    result["gain_over_target_default"] = (
        default_distance - result["all_metric_median_distance"]
    ) / default_distance
    result["gain_over_reddit_zero_shot"] = (
        zero_shot_distance - result["all_metric_median_distance"]
    ) / zero_shot_distance
    return result


def run_platform_adapter_curve_workflow(
    root: str | Path,
    config: Mapping[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    project = Path(root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    inputs = config["inputs"]
    external_metrics_path = project / inputs["external_metrics"]
    external_events_path = project / inputs["external_events"]
    reddit_metrics_path = project / inputs["reddit_metrics"]
    simulation_config_path = project / inputs["simulation_config"]
    external_metrics = pd.read_parquet(external_metrics_path)
    external_events = pd.read_parquet(external_events_path)
    reddit_metrics = pd.read_parquet(reddit_metrics_path)
    raw_simulation = json.loads(
        simulation_config_path.read_text(encoding="utf-8")
    )
    base_config = RevisionSimulationConfig.from_dict(
        raw_simulation["simulation"]
    )
    budgets = sorted({int(value) for value in config["validation_budgets"]})
    if not budgets or budgets[0] != 0:
        raise ValueError("validation_budgets must include zero")

    platform_metrics, split_manifest = chronological_platform_split(
        external_metrics,
        external_events,
        str(config["platform"]),
        config["split_fractions"],
    )
    if len(platform_metrics) < int(config["minimum_cascades"]):
        raise ValueError("Target platform does not meet minimum cascade count")
    if budgets[-1] > int(
        platform_metrics["split"].astype(str).eq("validation").sum()
    ):
        raise ValueError("Largest budget exceeds the validation split")
    temporal_path = output / "platform_temporal_metrics.parquet"
    platform_metrics.to_parquet(temporal_path, index=False)
    write_json(output / "platform_split_manifest.json", split_manifest)

    running_manifest = {
        "schema_version": 1,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "platform": str(config["platform"]),
        "validation_budgets": budgets,
        "split_assignment_sha256": split_assignment_hash(platform_metrics),
        "test_used_for_selection": False,
    }
    write_json(output / "platform_adapter_curve_manifest.json", running_manifest)

    profile = fit_pooled_model(platform_metrics, (), "target_platform")
    selected_configs: dict[int, RevisionSimulationConfig] = {0: base_config}
    selection_manifests: dict[int, dict[str, Any]] = {
        0: {
            "selection_split": None,
            "test_used_for_selection": False,
            "candidate_count": 0,
            "selected": _adapter_fields(base_config),
            "selection_score": None,
            "n_validation_cascades": 0,
            "selection_metrics": list(config["selection"]["metrics"]),
        }
    }
    score_frames: list[pd.DataFrame] = []
    reuse = _load_primary_reuse(
        project,
        config,
        platform_metrics,
        external_metrics_path,
        external_events_path,
        reddit_metrics_path,
        simulation_config_path,
    )
    primary_budget = int(config["reuse"]["primary_budget"])
    if reuse is not None and primary_budget in budgets:
        selected_configs[primary_budget] = replace(
            base_config, **reuse["selected"]
        )
        reused_scores = reuse["scores"].copy()
        reused_scores["validation_budget"] = primary_budget
        reused_scores["reused"] = True
        score_frames.append(reused_scores)
        reused_selection = dict(reuse["selection_manifest"])
        reused_selection["reused"] = True
        selection_manifests[primary_budget] = reused_selection

    for budget in budgets:
        if budget in selected_configs:
            continue
        score_path = checkpoint_dir / f"selection_scores_n{budget}.csv"
        selection_path = checkpoint_dir / f"selection_manifest_n{budget}.json"
        if bool(config.get("resume", True)) and score_path.exists() and selection_path.exists():
            scores = pd.read_csv(score_path)
            selection_manifest = json.loads(
                selection_path.read_text(encoding="utf-8")
            )
            selected = selection_manifest["selected"]
            selected_config = replace(
                base_config,
                ranking=str(selected["ranking"]),
                viewport_k=int(selected["viewport_k"]),
                deep_drill_lambda=float(selected["deep_drill_lambda"]),
            )
        else:
            budget_frame = frame_with_validation_budget(
                platform_metrics, budget
            )
            selected_config, scores, selection_manifest = (
                select_platform_adapter(
                    budget_frame,
                    profile,
                    base_config,
                    config["adapter_grid"],
                    config["selection"]["metrics"],
                    0,
                    int(config["selection"]["seed"]),
                )
            )
            scores.to_csv(score_path, index=False)
            write_json(selection_path, selection_manifest)
        scores = scores.copy()
        scores["validation_budget"] = budget
        scores["reused"] = False
        score_frames.append(scores)
        selection_manifests[budget] = selection_manifest
        selected_configs[budget] = selected_config

    scores = pd.concat(score_frames, ignore_index=True)
    scores.to_csv(output / "budget_selection_scores.csv", index=False)
    selection_rows = []
    for budget in budgets:
        subset = nested_validation_subset(platform_metrics, budget)
        selected = selected_configs[budget]
        details = selection_manifests[budget]
        selection_rows.append(
            {
                "validation_budget": budget,
                **_adapter_fields(selected),
                "selection_score": details.get("selection_score"),
                "candidate_count": int(details.get("candidate_count", 0)),
                "validation_subset_sha256": validation_subset_hash(subset),
                "test_used_for_selection": bool(
                    details.get("test_used_for_selection", False)
                ),
                "reused": bool(details.get("reused", False)),
            }
        )
    selection_summary = pd.DataFrame(selection_rows)
    selection_summary.to_csv(
        output / "budget_selection_summary.csv", index=False
    )

    fixed_metrics, fixed_events, reused_adapted = _fixed_test_outputs(
        platform_metrics,
        reddit_metrics,
        base_config,
        selected_configs[primary_budget],
        config,
        reuse,
    )
    adapted_metric_frames: list[pd.DataFrame] = []
    adapted_event_frames: list[pd.DataFrame] = []
    for budget in budgets:
        if budget == primary_budget and reused_adapted is not None:
            metrics, events = reused_adapted
        else:
            metrics_path = checkpoint_dir / f"adapted_metrics_n{budget}.parquet"
            events_path = checkpoint_dir / f"adapted_events_n{budget}.parquet"
            if (
                bool(config.get("resume", True))
                and metrics_path.exists()
                and events_path.exists()
            ):
                metrics = pd.read_parquet(metrics_path)
                events = pd.read_parquet(events_path)
            else:
                metrics, events = simulate_adapted_budget(
                    platform_metrics,
                    selected_configs[budget],
                    budget,
                    config["test"]["seeds"],
                    int(config["test"]["max_test_cascades"]),
                )
                metrics.to_parquet(metrics_path, index=False)
                events.to_parquet(events_path, index=False)
        adapted_metric_frames.append(metrics)
        adapted_event_frames.append(events)

    simulations = pd.concat(
        [fixed_metrics, *adapted_metric_frames], ignore_index=True
    )
    events = pd.concat(
        [fixed_events, *adapted_event_frames], ignore_index=True
    )
    simulations_path = output / "simulated_metrics.parquet"
    events_path = output / "events.parquet"
    simulations.to_parquet(simulations_path, index=False)
    simulations.to_csv(output / "simulated_metrics.csv", index=False)
    events.to_parquet(events_path, index=False)

    fidelity_dir = output / "evaluation"
    fidelity = evaluate_fidelity(
        platform_metrics,
        simulations,
        fidelity_dir,
        bootstrap_samples=int(config["test"]["bootstrap_samples"]),
        seed=int(config["selection"]["seed"]),
    )
    ranking = summarize_model_ranking(fidelity)
    ranking.to_csv(fidelity_dir / "model_ranking.csv", index=False)
    curve = summarize_budget_curve(
        fidelity, budgets, config["selection"]["metrics"]
    )
    curve = curve.merge(
        selection_summary[
            [
                "validation_budget",
                "ranking",
                "viewport_k",
                "deep_drill_lambda",
                "selection_score",
            ]
        ],
        on="validation_budget",
        how="left",
        validate="one_to_one",
    )
    curve_path = output / "adapter_budget_curve.csv"
    curve.to_csv(curve_path, index=False)
    _plot_budget_curve(curve, output / "adapter_budget_curve")
    _plot_adapter_choices(curve, output / "adapter_choices_by_budget")

    report_path = output / "PLATFORM_ADAPTER_BUDGET_REPORT.md"
    report_path.write_text(
        _build_report(curve, config, len(platform_metrics)),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "platform": str(config["platform"]),
        "validation_budgets": budgets,
        "n_train": int(
            platform_metrics["split"].astype(str).eq("train").sum()
        ),
        "n_validation": int(
            platform_metrics["split"].astype(str).eq("validation").sum()
        ),
        "n_test": int(
            platform_metrics["split"].astype(str).eq("test").sum()
        ),
        "n_simulations": int(len(simulations)),
        "n_events": int(len(events)),
        "split_assignment_sha256": split_assignment_hash(platform_metrics),
        "nested_validation_subsets": _nested_subset_audit(
            platform_metrics, budgets
        ),
        "test_used_for_selection": False,
        "training_filter": (
            "fit_pooled_model and simulation profiles use split=train only"
        ),
        "common_random_numbers_across_budgets": True,
        "primary_budget_reused": bool(reuse is not None),
        "selected_adapters": {
            str(budget): _adapter_fields(selected_configs[budget])
            for budget in budgets
        },
        "evidence_scope": dict(config["evidence_scope"]),
        "inputs": {
            "external_metrics_sha256": sha256_file(external_metrics_path),
            "external_events_sha256": sha256_file(external_events_path),
            "reddit_metrics_sha256": sha256_file(reddit_metrics_path),
            "simulation_config_sha256": sha256_file(simulation_config_path),
        },
        "outputs": {
            "temporal_metrics_sha256": sha256_file(temporal_path),
            "selection_scores_sha256": sha256_file(
                output / "budget_selection_scores.csv"
            ),
            "selection_summary_sha256": sha256_file(
                output / "budget_selection_summary.csv"
            ),
            "simulated_metrics_sha256": sha256_file(simulations_path),
            "events_sha256": sha256_file(events_path),
            "fidelity_distances_sha256": sha256_file(
                fidelity_dir / "fidelity_distances.csv"
            ),
            "budget_curve_sha256": sha256_file(curve_path),
            "report_sha256": sha256_file(report_path),
        },
    }
    write_json(output / "platform_adapter_curve_manifest.json", manifest)
    return manifest


def _fixed_test_outputs(
    platform_metrics: pd.DataFrame,
    reddit_metrics: pd.DataFrame,
    base_config: RevisionSimulationConfig,
    primary_config: RevisionSimulationConfig,
    config: Mapping[str, Any],
    reuse: dict[str, Any] | None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    tuple[pd.DataFrame, pd.DataFrame] | None,
]:
    primary_budget = int(config["reuse"]["primary_budget"])
    if reuse is not None:
        simulations = reuse["simulations"].copy()
        events = reuse["events"].copy()
        fixed_metrics = simulations[
            simulations["model"].isin(FIXED_MODEL_LABELS)
        ].copy()
        fixed_events = events[events["model"].isin(FIXED_MODEL_LABELS)].copy()
        adapted_metrics = simulations[
            simulations["model"].astype(str).eq(ADAPTED_LABEL)
        ].copy()
        adapted_events = events[
            events["model"].astype(str).eq(ADAPTED_LABEL)
        ].copy()
        adapted_label = f"platform_adapted_n{primary_budget}"
        adapted_metrics["model"] = adapted_label
        adapted_metrics["validation_budget"] = primary_budget
        adapted_events["model"] = adapted_label
        adapted_events["validation_budget"] = primary_budget
        return (
            fixed_metrics,
            fixed_events,
            (adapted_metrics, adapted_events),
        )
    simulations, events = run_platform_adapter_test(
        platform_metrics,
        reddit_metrics,
        primary_config,
        base_config,
        config["test"]["seeds"],
        int(config["test"]["max_test_cascades"]),
    )
    fixed_metrics = simulations[
        simulations["model"].isin(FIXED_MODEL_LABELS)
    ].copy()
    fixed_events = events[events["model"].isin(FIXED_MODEL_LABELS)].copy()
    adapted_metrics = simulations[
        simulations["model"].astype(str).eq(ADAPTED_LABEL)
    ].copy()
    adapted_events = events[
        events["model"].astype(str).eq(ADAPTED_LABEL)
    ].copy()
    adapted_label = f"platform_adapted_n{primary_budget}"
    adapted_metrics["model"] = adapted_label
    adapted_metrics["validation_budget"] = primary_budget
    adapted_events["model"] = adapted_label
    adapted_events["validation_budget"] = primary_budget
    return fixed_metrics, fixed_events, (adapted_metrics, adapted_events)


def _load_primary_reuse(
    project: Path,
    config: Mapping[str, Any],
    platform_metrics: pd.DataFrame,
    external_metrics_path: Path,
    external_events_path: Path,
    reddit_metrics_path: Path,
    simulation_config_path: Path,
) -> dict[str, Any] | None:
    reuse_config = config.get("reuse", {})
    if not bool(reuse_config.get("enabled", False)):
        return None
    source = project / str(reuse_config["source_output"])
    manifest_path = source / "platform_adapter_manifest.json"
    selection_path = source / "adapter_selection_manifest.json"
    score_path = source / "adapter_validation_scores.csv"
    split_path = source / "platform_temporal_metrics.parquet"
    simulations_path = source / "simulated_metrics.parquet"
    events_path = source / "events.parquet"
    required = (
        manifest_path,
        selection_path,
        score_path,
        split_path,
        simulations_path,
        events_path,
    )
    if not all(path.exists() for path in required):
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_hashes = {
        "external_metrics_sha256": sha256_file(external_metrics_path),
        "external_events_sha256": sha256_file(external_events_path),
        "reddit_metrics_sha256": sha256_file(reddit_metrics_path),
        "simulation_config_sha256": sha256_file(simulation_config_path),
    }
    if manifest.get("inputs") != expected_hashes:
        return None
    old_split = pd.read_parquet(split_path)
    if split_assignment_hash(old_split) != split_assignment_hash(platform_metrics):
        return None
    selection_manifest = json.loads(selection_path.read_text(encoding="utf-8"))
    if int(selection_manifest["n_validation_cascades"]) != int(
        reuse_config["primary_budget"]
    ):
        return None
    return {
        "selected": {
            "ranking": str(selection_manifest["selected"]["ranking"]),
            "viewport_k": int(selection_manifest["selected"]["viewport_k"]),
            "deep_drill_lambda": float(
                selection_manifest["selected"]["deep_drill_lambda"]
            ),
        },
        "selection_manifest": selection_manifest,
        "scores": pd.read_csv(score_path),
        "simulations": pd.read_parquet(simulations_path),
        "events": pd.read_parquet(events_path),
    }


def _nested_subset_audit(
    platform_metrics: pd.DataFrame,
    budgets: Iterable[int],
) -> list[dict[str, Any]]:
    rows = []
    previous_ids: set[str] = set()
    for budget in sorted(map(int, budgets)):
        subset = nested_validation_subset(platform_metrics, budget)
        ids = set(subset["post_id"].astype(str))
        if not previous_ids.issubset(ids):
            raise AssertionError("Validation budget subsets are not nested")
        rows.append(
            {
                "budget": budget,
                "n_cascades": int(len(subset)),
                "sha256": validation_subset_hash(subset),
                "contains_previous_budget": True,
            }
        )
        previous_ids = ids
    return rows


def _adapter_fields(config: RevisionSimulationConfig) -> dict[str, Any]:
    return {
        "ranking": config.ranking,
        "viewport_k": int(config.viewport_k),
        "deep_drill_lambda": float(config.deep_drill_lambda),
    }


def _median_distance(frame: pd.DataFrame) -> float:
    if frame.empty:
        return float("nan")
    return float(frame["normalized_wasserstein"].median())


def _mean_distance(frame: pd.DataFrame) -> float:
    if frame.empty:
        return float("nan")
    return float(frame["normalized_wasserstein"].mean())


def _stable_seed(*parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big", signed=False)


def _plot_budget_curve(curve: pd.DataFrame, path: Path) -> None:
    figure, ax = plt.subplots(figsize=(8.5, 5.2))
    x = curve["validation_budget"].to_numpy(dtype=float)
    ax.plot(
        x,
        curve["all_metric_median_distance"],
        marker="o",
        linewidth=2,
        color="#176B87",
        label="Adapted BDMTF: all metrics",
    )
    ax.plot(
        x,
        curve["primary_metric_median_distance"],
        marker="s",
        linewidth=2,
        color="#D95F59",
        label="Adapted BDMTF: prespecified metrics",
    )
    ax.axhline(
        float(curve["target_default_median_distance"].iloc[0]),
        color="#555555",
        linestyle="--",
        label="Target default",
    )
    ax.axhline(
        float(curve["reddit_zero_shot_median_distance"].iloc[0]),
        color="#999999",
        linestyle=":",
        label="Reddit zero-shot",
    )
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(value)) for value in x])
    ax.set_xlabel("HN validation cascades used for adapter selection")
    ax.set_ylabel("Median normalized Wasserstein distance (lower is better)")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path.with_suffix(".png"), dpi=220)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def _plot_adapter_choices(curve: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    x = curve["validation_budget"].to_numpy(dtype=float)
    axes[0].step(
        x,
        curve["viewport_k"].replace(-1, 40),
        where="mid",
        color="#176B87",
        linewidth=2,
    )
    axes[0].scatter(
        x,
        curve["viewport_k"].replace(-1, 40),
        color="#176B87",
    )
    axes[0].set_xscale("symlog", linthresh=10)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([str(int(value)) for value in x])
    axes[0].set_yticks([10, 20, 40])
    axes[0].set_yticklabels(["10", "20", "all"])
    axes[0].set_xlabel("Validation budget")
    axes[0].set_ylabel("Selected viewport")
    ranking_codes = {"top": 0, "best": 1, "hot": 2}
    ranking = curve["ranking"].map(ranking_codes)
    axes[1].scatter(
        x,
        ranking,
        s=60,
        c=curve["deep_drill_lambda"],
        cmap="viridis",
        vmin=0,
        vmax=max(float(curve["deep_drill_lambda"].max()), 1.0),
    )
    axes[1].set_xscale("symlog", linthresh=10)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(int(value)) for value in x])
    axes[1].set_yticks(list(ranking_codes.values()))
    axes[1].set_yticklabels(list(ranking_codes.keys()))
    axes[1].set_xlabel("Validation budget")
    axes[1].set_ylabel("Selected ranking")
    figure.tight_layout()
    figure.savefig(path.with_suffix(".png"), dpi=220)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def _build_report(
    curve: pd.DataFrame,
    config: Mapping[str, Any],
    n_cascades: int,
) -> str:
    table = curve[
        [
            "validation_budget",
            "ranking",
            "viewport_k",
            "deep_drill_lambda",
            "all_metric_median_distance",
            "primary_metric_median_distance",
            "gain_over_target_default",
            "gain_over_reddit_zero_shot",
        ]
    ].copy()
    for column in (
        "all_metric_median_distance",
        "primary_metric_median_distance",
        "gain_over_target_default",
        "gain_over_reddit_zero_shot",
    ):
        table[column] = table[column].map(lambda value: f"{value:.4f}")
    lines = [
        "# HN Platform Adapter Data-Budget Curve",
        "",
        f"- Platform cascades: {n_cascades:,}.",
        "- Split: chronological 60% train, 20% validation, 20% test.",
        "- Adapter selection uses validation only; the test set is opened once.",
        "- Lower normalized Wasserstein distance is better.",
        "",
        table.to_markdown(index=False),
        "",
        "## Interpretation Boundary",
        "",
        f"- Allowed: {config['evidence_scope']['allowed_claim']}",
        f"- Forbidden: {config['evidence_scope']['forbidden_claim']}",
        "- The curve is diagnostic. Monotonic improvement was not imposed and "
        "unsuccessful budgets remain in the report.",
        "",
    ]
    return "\n".join(lines)
