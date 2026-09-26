from __future__ import annotations

import hashlib
import json
import math
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bdmtf.revision.provenance import sha256_file, write_json


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**32 - 1)


def _qualified_rows(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "intervention_id",
        "community",
        "intervention_type",
        "treatment_date",
        "qualified",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Qualification diagnostics are missing: {sorted(missing)}")
    qualified = frame["qualified"].astype(str).str.lower().eq("true")
    result = frame.loc[qualified, sorted(required)].copy()
    if result.empty:
        raise ValueError("No qualified TBBT interventions were found")
    return result.sort_values("intervention_id").reset_index(drop=True)


def _json_parser() -> tuple[Any, str]:
    try:
        import ujson  # type: ignore[import-not-found]

        return ujson.loads, "ujson"
    except ImportError:
        return json.loads, "stdlib-json"


def _extract_pre_features(
    archive_path: Path,
    member: str,
    treatment_date: str,
    feature_days: int,
) -> dict[str, Any]:
    loads, parser_name = _json_parser()
    cutoff = int(
        pd.Timestamp(treatment_date, tz="UTC").timestamp()
        if pd.Timestamp(treatment_date).tzinfo is None
        else pd.Timestamp(treatment_date).tz_convert("UTC").timestamp()
    )
    feature_start = cutoff - int(feature_days) * 86400
    first_seen: dict[str, int] = {}
    author_comments: defaultdict[str, int] = defaultdict(int)
    first_thread: dict[str, str] = {}
    multi_thread_authors: set[str] = set()
    thread_comments: defaultdict[str, int] = defaultdict(int)
    daily_comments: defaultdict[int, int] = defaultdict(int)
    records_read = 0
    feature_comments = 0

    with zipfile.ZipFile(archive_path) as archive:
        if member not in archive.namelist():
            raise FileNotFoundError(f"{member} is absent from {archive_path}")
        with archive.open(member) as stream:
            for raw_line in stream:
                record = loads(raw_line)
                created = int(record["created_utc"])
                author = str(record.get("author") or "")
                thread = str(record.get("link_id") or "")
                records_read += 1
                if author and created < cutoff:
                    previous = first_seen.get(author)
                    if previous is None or created < previous:
                        first_seen[author] = created
                if not (feature_start <= created < cutoff):
                    continue
                feature_comments += 1
                day = (created - feature_start) // 86400
                daily_comments[int(day)] += 1
                if thread:
                    thread_comments[thread] += 1
                if not author:
                    continue
                author_comments[author] += 1
                if author not in first_thread:
                    first_thread[author] = thread
                elif thread and first_thread[author] != thread:
                    multi_thread_authors.add(author)

    if feature_comments == 0 or not author_comments:
        raise ValueError(f"No usable pre-intervention records in {member}")
    identified_comments = int(sum(author_comments.values()))
    novel_comments = int(
        sum(
            count
            for author, count in author_comments.items()
            if first_seen.get(author, cutoff) >= feature_start
        )
    )
    multi_thread_comments = int(
        sum(
            count
            for author, count in author_comments.items()
            if author in multi_thread_authors
        )
    )
    repeat_comments = int(
        sum(count for count in author_comments.values() if count >= 2)
    )
    thread_hhi = float(
        sum((count / feature_comments) ** 2 for count in thread_comments.values())
    )
    daily = np.asarray(
        [daily_comments.get(day, 0) for day in range(feature_days)],
        dtype=float,
    )
    return {
        "records_read": records_read,
        "feature_comments": feature_comments,
        "identified_comments": identified_comments,
        "feature_authors": len(author_comments),
        "feature_threads": len(thread_comments),
        "novel_comments": novel_comments,
        "novel_comment_share": novel_comments / identified_comments,
        "multi_thread_comments": multi_thread_comments,
        "multi_thread_comment_share": multi_thread_comments / identified_comments,
        "repeat_comments": repeat_comments,
        "repeat_comment_share": repeat_comments / identified_comments,
        "thread_hhi": thread_hhi,
        "mean_daily_comments": float(daily.mean()),
        "daily_comment_cv": float(daily.std(ddof=1) / max(daily.mean(), 1.0)),
        "parser": parser_name,
    }


def mechanism_ratio(
    intervention_type: str,
    *,
    novel_share: float,
    switch_share: float,
    removed_rate: float,
    external_traffic_multiplier: float,
    conflict_response_coefficient: float,
    hostile_dropout_probability: float,
    toxic_activation_boost: float,
    visibility_remaining: float,
) -> float:
    if intervention_type == "quarantine":
        external_retention = 1.0 / max(external_traffic_multiplier, 1e-9)
        discovery_retention = 1.0 - novel_share * (
            1.0 - visibility_remaining
        )
        incumbent_retention = 1.0 - hostile_dropout_probability
        return float(
            max(
                1e-9,
                external_retention
                * discovery_retention
                * incumbent_retention,
            )
        )
    if intervention_type == "post_removal":
        direct_retention = 1.0 - removed_rate * (
            1.0 - visibility_remaining
        ) * (1.0 - switch_share)
        rehydration_multiplier = 1.0 + (
            conflict_response_coefficient
            * toxic_activation_boost
            * switch_share
        )
        return float(max(1e-9, direct_retention * rehydration_multiplier))
    raise ValueError(f"Unsupported TBBT intervention type: {intervention_type}")


def _beta_draw(
    rng: np.random.Generator,
    successes: int,
    trials: int,
    size: int,
) -> np.ndarray:
    failures = max(0, int(trials) - int(successes))
    return rng.beta(successes + 0.5, failures + 0.5, size=size)


def _prediction_draws(
    row: Mapping[str, Any],
    parameters: Mapping[str, float],
    uncertainty: Mapping[str, Any],
) -> pd.DataFrame:
    draws = int(uncertainty["draws"])
    rng = np.random.default_rng(
        _stable_seed(uncertainty["seed"], row["intervention_id"])
    )
    novel = _beta_draw(
        rng,
        int(row["novel_comments"]),
        int(row["identified_comments"]),
        draws,
    )
    switching = _beta_draw(
        rng,
        int(row["multi_thread_comments"]),
        int(row["identified_comments"]),
        draws,
    )
    low_scale = float(uncertainty["local_scale_low"])
    high_scale = float(uncertainty["local_scale_high"])
    external = 1.0 + (
        float(parameters["external_traffic_multiplier"]) - 1.0
    ) * rng.uniform(low_scale, high_scale, size=draws)
    conflict = (
        float(parameters["conflict_response_coefficient"])
        * rng.uniform(low_scale, high_scale, size=draws)
    )
    activation = (
        float(parameters["toxic_activation_boost"])
        * rng.uniform(low_scale, high_scale, size=draws)
    )
    dropout = rng.triangular(
        float(uncertainty["dropout_low"]),
        float(uncertainty["dropout_mode"]),
        float(uncertainty["dropout_high"]),
        size=draws,
    )
    visibility = rng.triangular(
        float(uncertainty["visibility_low"]),
        float(uncertainty["visibility_mode"]),
        float(uncertainty["visibility_high"]),
        size=draws,
    )
    removed_rate = (
        float(row["removed_rate"])
        * rng.uniform(
            float(uncertainty["removed_rate_scale_low"]),
            float(uncertainty["removed_rate_scale_high"]),
            size=draws,
        )
    )
    if row["intervention_type"] == "quarantine":
        ratios = (
            (1.0 / external)
            * (1.0 - novel * (1.0 - visibility))
            * (1.0 - dropout)
        )
    elif row["intervention_type"] == "post_removal":
        ratios = (
            1.0 - removed_rate * (1.0 - visibility) * (1.0 - switching)
        ) * (1.0 + conflict * activation * switching)
    else:
        raise ValueError(row["intervention_type"])
    ratios = np.clip(ratios, 1e-9, None)
    return pd.DataFrame(
        {
            "intervention_id": str(row["intervention_id"]),
            "draw": np.arange(draws, dtype=int),
            "predicted_log_effect": np.log(ratios),
            "predicted_percent_effect": 100.0 * (ratios - 1.0),
        }
    )


def _load_frozen_parameters(
    project: Path,
    config: Mapping[str, Any],
) -> tuple[dict[str, float], dict[str, Any]]:
    inputs = config["inputs"]
    mechanism = config["mechanism"]
    paper_config = _read_json(project / str(inputs["paper_model_config"]))
    lemmy = _read_json(project / str(inputs["lemmy_agent_replay_manifest"]))
    ablations = _read_json(project / str(inputs["ablation_grid"]))

    activation = float(
        paper_config["simulation"]["toxic_activation_boost"]
    )
    visibility = float(
        lemmy["selected_validation_parameters"]["remove_visibility_remaining"]
    )
    dropout = float(mechanism["hostile_dropout_probability"])
    if not math.isclose(
        activation,
        float(mechanism["expected_toxic_activation_boost"]),
    ):
        raise ValueError("The frozen paper activation parameter changed")
    if not math.isclose(
        visibility,
        float(mechanism["expected_visibility_remaining"]),
    ):
        raise ValueError("The independently selected visibility parameter changed")
    if dropout not in set(map(float, ablations["hostile_dropout_probability"])):
        raise ValueError("The dropout parameter is absent from the frozen ablation grid")
    parameters = {
        "external_traffic_multiplier": float(
            mechanism["external_traffic_multiplier"]
        ),
        "conflict_response_coefficient": float(
            mechanism["conflict_response_coefficient"]
        ),
        "hostile_dropout_probability": dropout,
        "toxic_activation_boost": activation,
        "visibility_remaining": visibility,
    }
    provenance = {
        key: {
            "path": str((project / str(path)).resolve()),
            "sha256": sha256_file(project / str(path)),
        }
        for key, path in inputs.items()
        if key != "observed_effects"
    }
    return parameters, provenance


def freeze_tbbt_predictions(
    root: str | Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    project = Path(root)
    output = project / str(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    diagnostics_path = project / str(
        config["inputs"]["qualification_diagnostics"]
    )
    qualified = _qualified_rows(diagnostics_path)
    parameters, input_provenance = _load_frozen_parameters(project, config)
    community_models = _read_json(
        project / str(config["inputs"]["community_models"])
    )["communities"]
    raw_events = config["raw_events"]
    feature_rows: list[dict[str, Any]] = []
    archive_hashes: dict[str, str] = {}

    for event in qualified.itertuples(index=False):
        event_id = str(event.intervention_id)
        if event_id not in raw_events:
            raise ValueError(f"Missing raw pre-event mapping for {event_id}")
        source = raw_events[event_id]
        archive_path = project / str(source["archive"])
        archive_key = str(archive_path.resolve())
        if archive_key not in archive_hashes:
            archive_hashes[archive_key] = sha256_file(archive_path)
        features = _extract_pre_features(
            archive_path,
            str(source["member"]),
            str(event.treatment_date),
            int(config["mechanism"]["pre_feature_days"]),
        )
        removed_rate = float(
            community_models.get(str(event.community), {}).get("removed_rate", 0.0)
        )
        feature_rows.append(
            {
                "intervention_id": event_id,
                "community": str(event.community),
                "intervention_type": str(event.intervention_type),
                "treatment_date": str(event.treatment_date),
                "archive": str(source["archive"]),
                "archive_sha256": archive_hashes[archive_key],
                "member": str(source["member"]),
                "removed_rate": removed_rate,
                **features,
            }
        )

    features_frame = pd.DataFrame(feature_rows).sort_values("intervention_id")
    features_path = output / "pre_intervention_features.csv"
    features_frame.to_csv(features_path, index=False)
    prediction_rows: list[dict[str, Any]] = []
    all_draws: list[pd.DataFrame] = []
    for row in features_frame.to_dict(orient="records"):
        ratio = mechanism_ratio(
            str(row["intervention_type"]),
            novel_share=float(row["novel_comment_share"]),
            switch_share=float(row["multi_thread_comment_share"]),
            removed_rate=float(row["removed_rate"]),
            external_traffic_multiplier=parameters[
                "external_traffic_multiplier"
            ],
            conflict_response_coefficient=parameters[
                "conflict_response_coefficient"
            ],
            hostile_dropout_probability=parameters[
                "hostile_dropout_probability"
            ],
            toxic_activation_boost=parameters["toxic_activation_boost"],
            visibility_remaining=parameters["visibility_remaining"],
        )
        draws = _prediction_draws(row, parameters, config["uncertainty"])
        all_draws.append(draws)
        prediction_rows.append(
            {
                "intervention_id": str(row["intervention_id"]),
                "community": str(row["community"]),
                "intervention_type": str(row["intervention_type"]),
                "predicted_log_effect": math.log(ratio),
                "predicted_percent_effect": 100.0 * (ratio - 1.0),
                "prediction_log_low": float(
                    draws["predicted_log_effect"].quantile(0.025)
                ),
                "prediction_log_high": float(
                    draws["predicted_log_effect"].quantile(0.975)
                ),
                "prediction_percent_low": float(
                    draws["predicted_percent_effect"].quantile(0.025)
                ),
                "prediction_percent_high": float(
                    draws["predicted_percent_effect"].quantile(0.975)
                ),
                "novel_comment_share": float(row["novel_comment_share"]),
                "multi_thread_comment_share": float(
                    row["multi_thread_comment_share"]
                ),
                "removed_rate": float(row["removed_rate"]),
            }
        )
    predictions = pd.DataFrame(prediction_rows).sort_values("intervention_id")
    predictions_path = output / "frozen_predictions.csv"
    predictions.to_csv(predictions_path, index=False)
    draws_path = output / "prediction_draws.parquet"
    pd.concat(all_draws, ignore_index=True).to_parquet(draws_path, index=False)

    freeze_digest = hashlib.sha256(
        (
            predictions_path.read_text(encoding="utf-8")
            + json.dumps(config["protocol"], sort_keys=True)
            + json.dumps(parameters, sort_keys=True)
        ).encode("utf-8")
    ).hexdigest()
    manifest = {
        "status": "frozen",
        "protocol": dict(config["protocol"]),
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "freeze_id": freeze_digest,
        "qualified_interventions": int(len(predictions)),
        "parameters": parameters,
        "parameter_sources": dict(
            config["mechanism"].get("parameter_sources", {})
        ),
        "analyst_blinding": {
            "prospective_or_preregistered": False,
            "analyst_had_seen_aggregate_observed_effects": True,
            "code_level_outcome_firewall": True,
            "observed_effects_opened_by_freezer": False,
            "allowed_claim": (
                "Retrospective protocol-frozen mechanism-prediction check"
            ),
            "forbidden_claim": (
                "Prospective, preregistered, or analyst-blind prediction"
            ),
        },
        "inputs_available_to_freezer": input_provenance,
        "raw_archive_sha256": archive_hashes,
        "forbidden_scoring_input": str(config["inputs"]["observed_effects"]),
        "outputs": {
            "pre_intervention_features": {
                "path": str(features_path.resolve()),
                "sha256": sha256_file(features_path),
            },
            "frozen_predictions": {
                "path": str(predictions_path.resolve()),
                "sha256": sha256_file(predictions_path),
            },
            "prediction_draws": {
                "path": str(draws_path.resolve()),
                "sha256": sha256_file(draws_path),
            },
        },
    }
    manifest_path = output / "prediction_freeze_manifest.json"
    write_json(manifest_path, manifest)
    return manifest


def _interval_overlap(
    left_low: pd.Series,
    left_high: pd.Series,
    right_low: pd.Series,
    right_high: pd.Series,
) -> pd.Series:
    return np.maximum(left_low, right_low) <= np.minimum(left_high, right_high)


def _bootstrap_improvement(
    comparison: pd.DataFrame,
    draws: int,
    seed: int,
) -> dict[str, float]:
    improvement = (
        comparison["zero_absolute_error_pp"]
        - comparison["absolute_error_pp"]
    ).to_numpy(float)
    rng = np.random.default_rng(seed)
    sampled = rng.choice(improvement, size=(draws, len(improvement)), replace=True)
    means = sampled.mean(axis=1)
    return {
        "mean": float(improvement.mean()),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
    }


def _plot_fidelity(frame: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 5.2))
    colors = frame["intervention_type"].map(
        {"quarantine": "#2B6CB0", "post_removal": "#C05621"}
    )
    axis.errorbar(
        frame["percent_effect"],
        frame["predicted_percent_effect"],
        xerr=np.vstack(
            [
                frame["percent_effect"] - frame["percent_ci_low"],
                frame["percent_ci_high"] - frame["percent_effect"],
            ]
        ),
        yerr=np.vstack(
            [
                frame["predicted_percent_effect"]
                - frame["prediction_percent_low"],
                frame["prediction_percent_high"]
                - frame["predicted_percent_effect"],
            ]
        ),
        fmt="none",
        ecolor="#A0AEC0",
        elinewidth=1.2,
        capsize=3,
        zorder=1,
    )
    axis.scatter(
        frame["percent_effect"],
        frame["predicted_percent_effect"],
        c=colors,
        s=58,
        edgecolor="white",
        linewidth=0.8,
        zorder=2,
    )
    for row in frame.itertuples(index=False):
        axis.annotate(
            str(row.community),
            (row.percent_effect, row.predicted_percent_effect),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    limits = [
        float(
            min(
                frame["percent_ci_low"].min(),
                frame["prediction_percent_low"].min(),
            )
        )
        - 3.0,
        float(
            max(
                frame["percent_ci_high"].max(),
                frame["prediction_percent_high"].max(),
            )
        )
        + 3.0,
    ]
    axis.plot(limits, limits, color="#4A5568", linestyle="--", linewidth=1)
    axis.axhline(0, color="#CBD5E0", linewidth=0.8)
    axis.axvline(0, color="#CBD5E0", linewidth=0.8)
    axis.set_xlim(limits)
    axis.set_ylim(limits)
    axis.set_xlabel("Observed qualified-control effect (%)")
    axis.set_ylabel("Frozen BDMTF prediction (%)")
    axis.set_title("TBBT intervention-prediction fidelity")
    axis.grid(alpha=0.16)
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def score_tbbt_predictions(
    root: str | Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    project = Path(root)
    output = project / str(config["output_dir"])
    freeze_path = output / "prediction_freeze_manifest.json"
    predictions_path = output / "frozen_predictions.csv"
    if not freeze_path.is_file() or not predictions_path.is_file():
        raise FileNotFoundError("Freeze predictions before unsealing outcomes")
    freeze = _read_json(freeze_path)
    if freeze.get("status") != "frozen":
        raise ValueError("Prediction manifest is not frozen")
    expected_prediction_hash = freeze["outputs"]["frozen_predictions"]["sha256"]
    if sha256_file(predictions_path) != expected_prediction_hash:
        raise ValueError("Frozen predictions changed after sealing")

    observed_path = project / str(config["inputs"]["observed_effects"])
    observed = pd.read_csv(observed_path)
    observed = observed[
        observed["qualified"].astype(str).str.lower().eq("true")
        & observed["window_days"].astype(int).eq(30)
    ].copy()
    predictions = pd.read_csv(predictions_path)
    comparison = predictions.merge(
        observed[
            [
                "intervention_id",
                "log_effect",
                "ci_low",
                "ci_high",
                "percent_effect",
                "percent_ci_low",
                "percent_ci_high",
            ]
        ],
        on="intervention_id",
        how="inner",
        validate="one_to_one",
    )
    if len(comparison) != len(predictions):
        raise ValueError("Every frozen prediction must have one observed effect")
    comparison["direction_correct"] = np.sign(
        comparison["predicted_log_effect"]
    ).eq(np.sign(comparison["log_effect"]))
    comparison["absolute_error_log"] = (
        comparison["predicted_log_effect"] - comparison["log_effect"]
    ).abs()
    comparison["absolute_error_pp"] = (
        comparison["predicted_percent_effect"] - comparison["percent_effect"]
    ).abs()
    comparison["zero_absolute_error_log"] = comparison["log_effect"].abs()
    comparison["zero_absolute_error_pp"] = comparison["percent_effect"].abs()
    comparison["observed_in_prediction_interval"] = comparison[
        "percent_effect"
    ].between(
        comparison["prediction_percent_low"],
        comparison["prediction_percent_high"],
    )
    comparison["prediction_in_observed_interval"] = comparison[
        "predicted_percent_effect"
    ].between(comparison["percent_ci_low"], comparison["percent_ci_high"])
    comparison["intervals_overlap"] = _interval_overlap(
        comparison["prediction_percent_low"],
        comparison["prediction_percent_high"],
        comparison["percent_ci_low"],
        comparison["percent_ci_high"],
    )
    comparison_path = output / "prediction_observed_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    figure_path = output / "tbbt_intervention_fidelity.png"
    _plot_fidelity(comparison, figure_path)

    improvement = _bootstrap_improvement(
        comparison,
        int(config["uncertainty"]["bootstrap_draws"]),
        int(config["uncertainty"]["seed"]),
    )
    summary = {
        "status": "complete",
        "freeze_id": freeze["freeze_id"],
        "n_interventions": int(len(comparison)),
        "direction_accuracy": float(comparison["direction_correct"].mean()),
        "mae_log_effect": float(comparison["absolute_error_log"].mean()),
        "mae_percentage_points": float(comparison["absolute_error_pp"].mean()),
        "zero_effect_mae_log": float(
            comparison["zero_absolute_error_log"].mean()
        ),
        "zero_effect_mae_percentage_points": float(
            comparison["zero_absolute_error_pp"].mean()
        ),
        "mae_improvement_over_zero_percentage_points": improvement,
        "observed_point_prediction_interval_coverage": float(
            comparison["observed_in_prediction_interval"].mean()
        ),
        "prediction_point_observed_ci_coverage": float(
            comparison["prediction_in_observed_interval"].mean()
        ),
        "interval_overlap_rate": float(comparison["intervals_overlap"].mean()),
        "spearman_rank_correlation": float(
            comparison["predicted_percent_effect"].corr(
                comparison["percent_effect"], method="spearman"
            )
        ),
        "by_intervention_type": {
            str(kind): {
                "n": int(len(group)),
                "direction_accuracy": float(group["direction_correct"].mean()),
                "mae_percentage_points": float(group["absolute_error_pp"].mean()),
                "observed_point_prediction_interval_coverage": float(
                    group["observed_in_prediction_interval"].mean()
                ),
            }
            for kind, group in comparison.groupby("intervention_type")
        },
        "interpretation": (
            "The frozen mechanism map is directionally and quantitatively "
            "compatible with these five qualified TBBT effects if it improves "
            "on the zero-effect baseline. The sample remains small and the "
            "exercise is retrospective rather than prospectively blinded."
        ),
        "evidence_boundary": freeze["analyst_blinding"],
        "inputs": {
            "prediction_freeze_manifest": {
                "path": str(freeze_path.resolve()),
                "sha256": sha256_file(freeze_path),
            },
            "observed_effects": {
                "path": str(observed_path.resolve()),
                "sha256": sha256_file(observed_path),
            },
        },
        "outputs": {
            "comparison": {
                "path": str(comparison_path.resolve()),
                "sha256": sha256_file(comparison_path),
            },
            "figure": {
                "path": str(figure_path.resolve()),
                "sha256": sha256_file(figure_path),
            },
        },
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    summary_path = output / "fidelity_summary.json"
    write_json(summary_path, summary)
    report_rows = "\n".join(
        "| {community} | {kind} | {pred:+.1f}% [{low:+.1f}, {high:+.1f}] | "
        "{obs:+.1f}% [{obs_low:+.1f}, {obs_high:+.1f}] | {direction} | "
        "{error:.1f} |".format(
            community=row.community,
            kind=row.intervention_type,
            pred=row.predicted_percent_effect,
            low=row.prediction_percent_low,
            high=row.prediction_percent_high,
            obs=row.percent_effect,
            obs_low=row.percent_ci_low,
            obs_high=row.percent_ci_high,
            direction="yes" if row.direction_correct else "no",
            error=row.absolute_error_pp,
        )
        for row in comparison.itertuples(index=False)
    )
    report = f"""# TBBT Frozen BDMTF Intervention-Prediction Fidelity

## Protocol boundary

Predictions were generated by a code-level outcome firewall from TBBT
`IN-BEFORE` activity and previously frozen mechanism parameters. The scorer
then unsealed the five qualified 30-day effects. Because the analyst had seen
aggregate outcomes before this protocol was written, this is retrospective
protocol freezing, not prospective or preregistered prediction.

## Event-level results

| Community | Intervention | Frozen prediction | Observed effect | Direction | Absolute error (pp) |
|---|---|---:|---:|---:|---:|
{report_rows}

## Aggregate fidelity

- Direction accuracy: {summary['direction_accuracy']:.1%}.
- Mean absolute error: {summary['mae_percentage_points']:.2f} percentage points.
- Zero-effect baseline MAE: {summary['zero_effect_mae_percentage_points']:.2f} percentage points.
- MAE improvement over zero: {improvement['mean']:.2f} pp
  [{improvement['ci_low']:.2f}, {improvement['ci_high']:.2f}].
- Observed point inside the BDMTF prediction interval:
  {summary['observed_point_prediction_interval_coverage']:.1%}.
- Frozen prediction inside the observed 95% interval:
  {summary['prediction_point_observed_ci_coverage']:.1%}.
- Spearman rank correlation: {summary['spearman_rank_correlation']:.3f}.

The result evaluates whether a fixed mechanism map can anticipate direction,
magnitude, and uncertainty for these observed interventions. It does not
establish universal moderation effects, and five events are insufficient for a
standalone causal generalization claim.
"""
    (output / "TBBT_INTERVENTION_FIDELITY_REPORT.md").write_text(
        report,
        encoding="utf-8",
    )
    return summary


def run_tbbt_intervention_fidelity(
    root: str | Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    freeze_tbbt_predictions(root, config)
    return score_tbbt_predictions(root, config)
