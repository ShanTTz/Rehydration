from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bdmtf.revision.provenance import sha256_file, write_json


PRIMARY_OUTCOMES = ("reply_count", "active_authors", "max_depth")
SECONDARY_OUTCOMES = ("removed_replies",)
MODEL_ORDER = (
    "zero_effect",
    "frozen_mechanism",
    "type_median",
    "additive_ridge",
    "bdmtf_interaction",
    "gradient_boosting",
)


def _frame_hash(frame: pd.DataFrame) -> str:
    ordered = frame.sort_values(list(frame.columns)).reset_index(drop=True)
    return hashlib.sha256(
        pd.util.hash_pandas_object(ordered, index=True).values.tobytes()
    ).hexdigest()


def _slope(values: np.ndarray) -> float:
    if len(values) < 2 or np.allclose(values, values[0]):
        return 0.0
    return float(np.polyfit(np.arange(len(values), dtype=float), values, 1)[0])


def _stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    transformed = np.log1p(np.maximum(values, 0.0))
    return {
        "mean": float(transformed.mean()),
        "last": float(transformed[-1]),
        "slope": _slope(transformed),
        "std": float(transformed.std()),
        "sum": float(transformed.sum()),
        "zero_share": float(np.mean(values <= 0)),
    }


def _finite_conformal_quantile(errors: np.ndarray, coverage: float) -> float:
    clean = np.sort(np.asarray(errors, dtype=float)[np.isfinite(errors)])
    if not len(clean):
        return 0.0
    rank = min(
        len(clean),
        max(1, int(math.ceil((len(clean) + 1) * float(coverage)))),
    )
    return float(clean[rank - 1])


def _assign_time_splits(
    matches: pd.DataFrame,
    train_fraction: float,
    validation_fraction: float,
) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    for _, group in matches.groupby("intervention_type", sort=True):
        ordered = group.sort_values(["event_time", "intervention_id"]).copy()
        count = len(ordered)
        train_end = int(np.floor(count * train_fraction))
        validation_end = train_end + int(np.floor(count * validation_fraction))
        assignments = np.full(count, "test", dtype=object)
        assignments[:train_end] = "train"
        assignments[train_end:validation_end] = "validation"
        ordered["split"] = assignments
        records.append(ordered)
    result = pd.concat(records, ignore_index=True)
    if result["intervention_id"].duplicated().any():
        raise ValueError("An intervention was assigned to multiple splits")
    return result


def _arm_values(
    panel_index: pd.DataFrame,
    intervention_id: str,
    outcome: str,
    treated: int,
    periods: Iterable[int],
) -> np.ndarray:
    group = panel_index[
        panel_index["intervention_id"].eq(intervention_id)
        & panel_index["outcome"].eq(outcome)
        & panel_index["treated"].eq(treated)
    ].set_index("relative_period")["value"]
    return np.asarray(
        [float(group.get(int(period), 0.0)) for period in periods],
        dtype=float,
    )


def _mechanism_prediction(
    treated_pre: np.ndarray,
    intervention_type: str,
    outcome: str,
    post_periods: list[int],
    remove_visibility_remaining: float,
) -> float:
    values = np.asarray(treated_pre, dtype=float)
    x = np.arange(-len(values), 0, dtype=float)
    transformed = np.log1p(np.maximum(values, 0.0))
    if np.allclose(transformed, transformed[0]):
        forecast = np.repeat(transformed[-1], len(post_periods))
    else:
        slope, intercept = np.polyfit(x, transformed, 1)
        forecast = intercept + slope * np.asarray(post_periods, dtype=float)
    upper = float(np.log1p(max(1.0, 4.0 * float(values.max()) + 1.0)))
    future = np.expm1(np.clip(forecast, 0.0, upper))
    remaining = (
        0.0
        if intervention_type == "lock_post"
        else float(remove_visibility_remaining)
    )
    suppression = 1.0 - float(np.clip(remaining, 0.0, 1.0))
    if outcome == "removed_replies":
        suppression *= 0.5
    return -suppression * float(future.mean())


def prepare_intervention_dataset(
    panel: pd.DataFrame,
    matches: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    required_panel = {
        "intervention_id",
        "intervention_type",
        "treated",
        "relative_period",
        "outcome",
        "value",
    }
    required_matches = {
        "intervention_id",
        "intervention_type",
        "event_time",
    }
    missing = sorted(
        required_panel.difference(panel.columns)
        | required_matches.difference(matches.columns)
    )
    if missing:
        raise ValueError("Lemmy fidelity inputs are missing: " + ", ".join(missing))

    primary = list(config.get("protocol", {}).get("primary_outcomes", PRIMARY_OUTCOMES))
    secondary = list(
        config.get("protocol", {}).get("secondary_outcomes", SECONDARY_OUTCOMES)
    )
    outcomes = primary + secondary
    pre_periods = [int(value) for value in config.get("pre_periods", range(-7, 0))]
    post_periods = [int(value) for value in config.get("post_periods", range(0, 8))]

    prepared_panel = panel.copy()
    prepared_panel["intervention_id"] = prepared_panel["intervention_id"].astype(str)
    prepared_panel["treated"] = pd.to_numeric(
        prepared_panel["treated"], errors="raise"
    ).astype(int)
    prepared_panel["relative_period"] = pd.to_numeric(
        prepared_panel["relative_period"], errors="raise"
    ).astype(int)
    prepared_panel["value"] = pd.to_numeric(
        prepared_panel["value"], errors="raise"
    ).astype(float)
    prepared_panel = prepared_panel[
        prepared_panel["outcome"].astype(str).isin(outcomes)
    ].copy()

    prepared_matches = matches.copy()
    prepared_matches["intervention_id"] = prepared_matches[
        "intervention_id"
    ].astype(str)
    prepared_matches["event_time"] = pd.to_datetime(
        prepared_matches["event_time"],
        utc=True,
        errors="coerce",
    )
    if prepared_matches["event_time"].isna().any():
        raise ValueError("Lemmy matches contain invalid event_time values")
    prepared_matches = prepared_matches.drop_duplicates("intervention_id")
    prepared_matches = _assign_time_splits(
        prepared_matches,
        float(config.get("train_fraction", 0.6)),
        float(config.get("validation_fraction", 0.2)),
    )

    feature_records: list[dict[str, Any]] = []
    effect_records: list[dict[str, Any]] = []
    numeric_match_columns = (
        "treated_age_hours",
        "control_age_hours",
        "distance",
        "pretrajectory_distance",
        "match_score",
    )
    for match in prepared_matches.itertuples(index=False):
        intervention_id = str(match.intervention_id)
        intervention_type = str(match.intervention_type)
        feature: dict[str, Any] = {
            "intervention_id": intervention_id,
            "intervention_type": intervention_type,
            "event_time": match.event_time,
            "split": str(match.split),
            "community_id": str(getattr(match, "community_id", "")),
            "is_remove_post": float(intervention_type == "remove_post"),
        }
        for column in numeric_match_columns:
            value = getattr(match, column, 0.0)
            feature[column] = float(value) if pd.notna(value) else 0.0

        outcome_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for outcome in outcomes:
            treated_pre = _arm_values(
                prepared_panel,
                intervention_id,
                outcome,
                1,
                pre_periods,
            )
            control_pre = _arm_values(
                prepared_panel,
                intervention_id,
                outcome,
                0,
                pre_periods,
            )
            treated_post = _arm_values(
                prepared_panel,
                intervention_id,
                outcome,
                1,
                post_periods,
            )
            control_post = _arm_values(
                prepared_panel,
                intervention_id,
                outcome,
                0,
                post_periods,
            )
            outcome_arrays[outcome] = (treated_pre, control_pre)
            treated_stats = _stats(treated_pre)
            control_stats = _stats(control_pre)
            difference_stats = _stats(np.maximum(treated_pre - control_pre, 0.0))
            for name, value in treated_stats.items():
                feature[f"{outcome}__treated_{name}"] = value
            for name, value in control_stats.items():
                feature[f"{outcome}__control_{name}"] = value
            for name, value in difference_stats.items():
                feature[f"{outcome}__positive_difference_{name}"] = value
            feature[f"{outcome}__raw_mean_difference"] = float(
                treated_pre.mean() - control_pre.mean()
            )
            feature[f"{outcome}__raw_slope_difference"] = float(
                _slope(treated_pre) - _slope(control_pre)
            )
            observed_effect = float(
                (treated_post - control_post).mean()
                - (treated_pre - control_pre).mean()
            )
            effect_records.append(
                {
                    "intervention_id": intervention_id,
                    "intervention_type": intervention_type,
                    "event_time": match.event_time,
                    "split": str(match.split),
                    "community_id": feature["community_id"],
                    "outcome": outcome,
                    "observed_effect": observed_effect,
                    "frozen_mechanism_prediction": _mechanism_prediction(
                        treated_pre,
                        intervention_type,
                        outcome,
                        post_periods,
                        float(config.get("remove_post_visibility_remaining", 0.2)),
                    ),
                }
            )
        feature_records.append(feature)

    features = pd.DataFrame(feature_records).sort_values(
        ["event_time", "intervention_id"]
    )
    effects = pd.DataFrame(effect_records).sort_values(
        ["event_time", "intervention_id", "outcome"]
    )
    metadata_columns = {
        "intervention_id",
        "intervention_type",
        "event_time",
        "split",
        "community_id",
    }
    additive_columns = [
        column
        for column in features.columns
        if column not in metadata_columns
    ]
    interaction = features[additive_columns].copy()
    selected_states = [
        "reply_count__treated_mean",
        "reply_count__treated_last",
        "reply_count__treated_slope",
        "active_authors__treated_mean",
        "active_authors__treated_last",
        "active_authors__treated_slope",
        "max_depth__treated_mean",
        "max_depth__treated_last",
        "max_depth__treated_slope",
        "reply_count__raw_mean_difference",
        "active_authors__raw_mean_difference",
        "max_depth__raw_mean_difference",
    ]
    selected_states = [name for name in selected_states if name in interaction]
    for column in selected_states:
        interaction[f"remove_x__{column}"] = (
            interaction["is_remove_post"] * interaction[column]
        )
    state_pairs = (
        ("reply_count__treated_mean", "max_depth__treated_mean"),
        ("reply_count__treated_last", "max_depth__treated_last"),
        ("active_authors__treated_mean", "max_depth__treated_mean"),
        ("reply_count__treated_slope", "max_depth__treated_slope"),
        (
            "reply_count__raw_mean_difference",
            "max_depth__raw_mean_difference",
        ),
        (
            "active_authors__raw_mean_difference",
            "max_depth__raw_mean_difference",
        ),
    )
    for left, right in state_pairs:
        if left in interaction and right in interaction:
            interaction[f"interaction__{left}__{right}"] = (
                interaction[left] * interaction[right]
            )
    interaction_columns = list(interaction.columns)
    for column in interaction_columns:
        if column not in features:
            features[column] = interaction[column]
    numeric_columns = list(dict.fromkeys(additive_columns + interaction_columns))
    features[numeric_columns] = features[numeric_columns].replace(
        [np.inf, -np.inf],
        np.nan,
    ).fillna(0.0)
    return features, effects, additive_columns, interaction_columns


def _fit_ridge(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_validation: pd.DataFrame,
    y_validation: np.ndarray,
    alphas: Iterable[float],
) -> tuple[Any, float, float]:
    best_model = None
    best_alpha = None
    best_mae = float("inf")
    for alpha in alphas:
        model = make_pipeline(
            StandardScaler(),
            Ridge(alpha=float(alpha)),
        )
        model.fit(x_train, y_train)
        prediction = model.predict(x_validation)
        score = float(mean_absolute_error(y_validation, prediction))
        if score < best_mae:
            best_model = model
            best_alpha = float(alpha)
            best_mae = score
    if best_model is None or best_alpha is None:
        raise RuntimeError("Ridge validation did not produce a fitted model")
    return best_model, best_alpha, best_mae


def _type_median_predictions(
    train: pd.DataFrame,
    targets: pd.DataFrame,
) -> np.ndarray:
    global_value = float(train["observed_effect"].median())
    medians = train.groupby("intervention_type")["observed_effect"].median()
    return targets["intervention_type"].map(medians).fillna(global_value).to_numpy(float)


def _model_predictions(
    features: pd.DataFrame,
    effects: pd.DataFrame,
    outcome: str,
    additive_columns: list[str],
    interaction_columns: list[str],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    frame = effects[effects["outcome"].eq(outcome)].merge(
        features,
        on=[
            "intervention_id",
            "intervention_type",
            "event_time",
            "split",
            "community_id",
        ],
        validate="one_to_one",
    )
    train = frame[frame["split"].eq("train")].copy()
    validation = frame[frame["split"].eq("validation")].copy()
    test = frame[frame["split"].eq("test")].copy()
    if train.empty or validation.empty or test.empty:
        raise ValueError(f"{outcome} lacks a non-empty chronological split")

    alphas = [float(value) for value in config.get("ridge_grid", [0.1, 1.0, 10.0])]
    additive, additive_alpha, additive_validation_mae = _fit_ridge(
        train[additive_columns],
        train["observed_effect"].to_numpy(float),
        validation[additive_columns],
        validation["observed_effect"].to_numpy(float),
        alphas,
    )
    interaction, interaction_alpha, interaction_validation_mae = _fit_ridge(
        train[interaction_columns],
        train["observed_effect"].to_numpy(float),
        validation[interaction_columns],
        validation["observed_effect"].to_numpy(float),
        alphas,
    )
    boosting = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=200,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        loss="absolute_error",
        random_state=int(config.get("seed", 30371)),
    )
    boosting.fit(
        train[additive_columns],
        train["observed_effect"].to_numpy(float),
    )

    models: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "zero_effect": (
            np.zeros(len(validation)),
            np.zeros(len(test)),
        ),
        "frozen_mechanism": (
            validation["frozen_mechanism_prediction"].to_numpy(float),
            test["frozen_mechanism_prediction"].to_numpy(float),
        ),
        "type_median": (
            _type_median_predictions(train, validation),
            _type_median_predictions(train, test),
        ),
        "additive_ridge": (
            additive.predict(validation[additive_columns]),
            additive.predict(test[additive_columns]),
        ),
        "bdmtf_interaction": (
            interaction.predict(validation[interaction_columns]),
            interaction.predict(test[interaction_columns]),
        ),
        "gradient_boosting": (
            boosting.predict(validation[additive_columns]),
            boosting.predict(test[additive_columns]),
        ),
    }
    coverage = float(config.get("prediction_interval", 0.95))
    prediction_rows: list[pd.DataFrame] = []
    tuning = [
        {
            "outcome": outcome,
            "model": "additive_ridge",
            "selected_alpha": additive_alpha,
            "validation_mae": additive_validation_mae,
        },
        {
            "outcome": outcome,
            "model": "bdmtf_interaction",
            "selected_alpha": interaction_alpha,
            "validation_mae": interaction_validation_mae,
        },
    ]
    validation_observed = validation["observed_effect"].to_numpy(float)
    for model_name, (validation_prediction, test_prediction) in models.items():
        radius = _finite_conformal_quantile(
            np.abs(validation_prediction - validation_observed),
            coverage,
        )
        part = test[
            [
                "intervention_id",
                "intervention_type",
                "event_time",
                "community_id",
                "split",
                "observed_effect",
            ]
        ].copy()
        part["outcome"] = outcome
        part["model"] = model_name
        part["predicted_effect"] = test_prediction
        part["ci_low"] = test_prediction - radius
        part["ci_high"] = test_prediction + radius
        part["conformal_radius"] = radius
        prediction_rows.append(part)
    return pd.concat(prediction_rows, ignore_index=True), tuning


def _bootstrap_metric_intervals(
    frame: pd.DataFrame,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    cluster = (
        frame.groupby("intervention_id", sort=False)[
            ["absolute_error", "direction_correct", "interval_covers"]
        ]
        .mean()
        .to_numpy(dtype=float)
    )
    indices = rng.integers(
        0,
        len(cluster),
        size=(int(bootstrap_samples), len(cluster)),
    )
    draws = cluster[indices].mean(axis=1)
    values = {
        "mae": draws[:, 0],
        "direction_accuracy": draws[:, 1],
        "interval_coverage": draws[:, 2],
    }
    return {
        metric: (
            float(np.quantile(samples, 0.025)),
            float(np.quantile(samples, 0.975)),
        )
        for metric, samples in values.items()
    }


def summarize_predictions(
    predictions: pd.DataFrame,
    bootstrap_samples: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = predictions.copy()
    frame["error"] = frame["predicted_effect"] - frame["observed_effect"]
    frame["absolute_error"] = frame["error"].abs()
    frame["squared_error"] = frame["error"] ** 2
    frame["direction_correct"] = (
        np.sign(frame["predicted_effect"]) == np.sign(frame["observed_effect"])
    )
    frame["interval_covers"] = (
        (frame["observed_effect"] >= frame["ci_low"])
        & (frame["observed_effect"] <= frame["ci_high"])
    )
    rows: list[dict[str, Any]] = []
    scopes = [
        *[
            (outcome, group)
            for outcome, group in frame.groupby("outcome", sort=True)
        ],
        ("all_primary", frame[frame["outcome"].isin(PRIMARY_OUTCOMES)]),
    ]
    for scope, scope_frame in scopes:
        for model, group in scope_frame.groupby("model", sort=False):
            intervals = _bootstrap_metric_intervals(
                group,
                bootstrap_samples,
                seed + int(hashlib.sha256(f"{scope}|{model}".encode()).hexdigest()[:8], 16),
            )
            rows.append(
                {
                    "scope": scope,
                    "model": model,
                    "n_predictions": int(len(group)),
                    "n_interventions": int(group["intervention_id"].nunique()),
                    "mae": float(group["absolute_error"].mean()),
                    "mae_ci_low": intervals["mae"][0],
                    "mae_ci_high": intervals["mae"][1],
                    "rmse": float(np.sqrt(group["squared_error"].mean())),
                    "mean_error": float(group["error"].mean()),
                    "direction_accuracy": float(group["direction_correct"].mean()),
                    "direction_accuracy_ci_low": intervals["direction_accuracy"][0],
                    "direction_accuracy_ci_high": intervals["direction_accuracy"][1],
                    "interval_coverage": float(group["interval_covers"].mean()),
                    "interval_coverage_ci_low": intervals["interval_coverage"][0],
                    "interval_coverage_ci_high": intervals["interval_coverage"][1],
                }
            )
    return frame, pd.DataFrame(rows)


def paired_model_improvements(
    evaluated: pd.DataFrame,
    bootstrap_samples: int,
    seed: int,
) -> pd.DataFrame:
    comparisons = (
        "zero_effect",
        "frozen_mechanism",
        "type_median",
        "additive_ridge",
        "gradient_boosting",
    )
    primary = evaluated[evaluated["outcome"].isin(PRIMARY_OUTCOMES)]
    full = primary[primary["model"].eq("bdmtf_interaction")][
        ["intervention_id", "outcome", "absolute_error", "direction_correct"]
    ].rename(
        columns={
            "absolute_error": "bdmtf_error",
            "direction_correct": "bdmtf_direction",
        }
    )
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    for baseline_name in comparisons:
        baseline = primary[primary["model"].eq(baseline_name)][
            ["intervention_id", "outcome", "absolute_error", "direction_correct"]
        ].rename(
            columns={
                "absolute_error": "baseline_error",
                "direction_correct": "baseline_direction",
            }
        )
        paired = full.merge(
            baseline,
            on=["intervention_id", "outcome"],
            validate="one_to_one",
        )
        paired["mae_improvement"] = (
            paired["baseline_error"] - paired["bdmtf_error"]
        )
        paired["direction_improvement"] = (
            paired["bdmtf_direction"].astype(float)
            - paired["baseline_direction"].astype(float)
        )
        cluster = (
            paired.groupby("intervention_id", sort=False)[
                ["mae_improvement", "direction_improvement"]
            ]
            .mean()
            .to_numpy(dtype=float)
        )
        indices = rng.integers(
            0,
            len(cluster),
            size=(int(bootstrap_samples), len(cluster)),
        )
        draws = cluster[indices].mean(axis=1)
        bootstrap_mae = draws[:, 0]
        bootstrap_direction = draws[:, 1]
        rows.append(
            {
                "full_model": "bdmtf_interaction",
                "baseline_model": baseline_name,
                "n_interventions": int(len(cluster)),
                "mean_absolute_error_improvement": float(
                    paired["mae_improvement"].mean()
                ),
                "mae_improvement_ci_low": float(
                    np.quantile(bootstrap_mae, 0.025)
                ),
                "mae_improvement_ci_high": float(
                    np.quantile(bootstrap_mae, 0.975)
                ),
                "direction_accuracy_improvement": float(
                    paired["direction_improvement"].mean()
                ),
                "direction_improvement_ci_low": float(
                    np.quantile(bootstrap_direction, 0.025)
                ),
                "direction_improvement_ci_high": float(
                    np.quantile(bootstrap_direction, 0.975)
                ),
                "pairwise_win_share": float(
                    (paired["bdmtf_error"] < paired["baseline_error"]).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def _plot_results(
    evaluated: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    primary = summary[summary["scope"].eq("all_primary")].copy()
    order = [model for model in MODEL_ORDER if model in set(primary["model"])]
    primary = primary.set_index("model").loc[order].reset_index()
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(primary["model"], primary["mae"], color="#3B6F8F")
    axes[0].set_ylabel("Mean absolute error (lower is better)")
    axes[0].tick_params(axis="x", rotation=35)
    axes[1].bar(
        primary["model"],
        primary["direction_accuracy"],
        color="#B45F4B",
    )
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Direction accuracy")
    axes[1].tick_params(axis="x", rotation=35)
    figure.tight_layout()
    figure.savefig(output_dir / "model_fidelity.png", dpi=180)
    figure.savefig(output_dir / "model_fidelity.pdf")
    plt.close(figure)

    bdmtf = evaluated[
        evaluated["model"].eq("bdmtf_interaction")
        & evaluated["outcome"].isin(PRIMARY_OUTCOMES)
    ]
    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    for axis, outcome in zip(axes, PRIMARY_OUTCOMES, strict=True):
        group = bdmtf[bdmtf["outcome"].eq(outcome)]
        axis.scatter(
            group["observed_effect"],
            group["predicted_effect"],
            alpha=0.55,
            s=20,
            color="#3B6F8F",
        )
        low = float(min(group["observed_effect"].min(), group["predicted_effect"].min()))
        high = float(max(group["observed_effect"].max(), group["predicted_effect"].max()))
        axis.plot([low, high], [low, high], color="#202020", linewidth=1)
        axis.set_title(outcome.replace("_", " "))
        axis.set_xlabel("Observed pair effect")
        axis.set_ylabel("Predicted effect")
    figure.tight_layout()
    figure.savefig(output_dir / "bdmtf_observed_vs_predicted.png", dpi=180)
    figure.savefig(output_dir / "bdmtf_observed_vs_predicted.pdf")
    plt.close(figure)


def run_lemmy_intervention_fidelity(
    panel_path: Path,
    matches_path: Path,
    natural_experiment_path: Path,
    output_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(panel_path)
    matches = pd.read_csv(matches_path)
    natural = json.loads(natural_experiment_path.read_text(encoding="utf-8"))
    features, effects, additive_columns, interaction_columns = (
        prepare_intervention_dataset(panel, matches, config)
    )
    predictions: list[pd.DataFrame] = []
    tuning_records: list[dict[str, Any]] = []
    outcomes = list(config.get("protocol", {}).get("primary_outcomes", PRIMARY_OUTCOMES))
    outcomes += list(
        config.get("protocol", {}).get("secondary_outcomes", SECONDARY_OUTCOMES)
    )
    for outcome in outcomes:
        part, tuning = _model_predictions(
            features,
            effects,
            outcome,
            additive_columns,
            interaction_columns,
            config,
        )
        predictions.append(part)
        tuning_records.extend(tuning)

    prediction_frame = pd.concat(predictions, ignore_index=True)
    evaluated, summary = summarize_predictions(
        prediction_frame,
        int(config.get("bootstrap_samples", 2000)),
        int(config.get("seed", 30371)),
    )
    improvements = paired_model_improvements(
        evaluated,
        int(config.get("bootstrap_samples", 2000)),
        int(config.get("seed", 30371)) + 1,
    )
    splits = features[
        [
            "intervention_id",
            "intervention_type",
            "event_time",
            "community_id",
            "split",
        ]
    ].copy()
    splits.to_csv(output_dir / "intervention_splits.csv", index=False)
    features.to_parquet(output_dir / "pre_intervention_features.parquet", index=False)
    effects.to_parquet(output_dir / "pair_effects.parquet", index=False)
    evaluated.to_parquet(output_dir / "test_predictions.parquet", index=False)
    evaluated.to_csv(output_dir / "test_predictions.csv", index=False)
    summary.to_csv(output_dir / "fidelity_summary.csv", index=False)
    improvements.to_csv(output_dir / "paired_improvements.csv", index=False)
    pd.DataFrame(tuning_records).to_csv(
        output_dir / "model_selection.csv",
        index=False,
    )
    _plot_results(evaluated, summary, output_dir)

    primary = summary[summary["scope"].eq("all_primary")].set_index("model")
    bdmtf = primary.loc["bdmtf_interaction"]
    best_model = str(primary["mae"].idxmin())
    best_direction_model = str(primary["direction_accuracy"].idxmax())
    additive_comparison = improvements[
        improvements["baseline_model"].eq("additive_ridge")
    ].iloc[0]
    mechanism_comparison = improvements[
        improvements["baseline_model"].eq("frozen_mechanism")
    ].iloc[0]
    test_splits = splits[splits["split"].eq("test")]
    chronological = True
    for _, group in splits.groupby("intervention_type"):
        train_time = group[group["split"].eq("train")]["event_time"]
        validation_time = group[group["split"].eq("validation")]["event_time"]
        test_time = group[group["split"].eq("test")]["event_time"]
        chronological &= bool(
            train_time.max() <= validation_time.min()
            and validation_time.max() <= test_time.min()
        )
    result = {
        "status": "complete",
        "claim_allowed": bool(chronological and len(test_splits) >= 100),
        "protocol": config.get("protocol", {}),
        "n_pairs": int(features["intervention_id"].nunique()),
        "split_counts": {
            str(key): int(value)
            for key, value in splits["split"].value_counts().items()
        },
        "test_pairs": int(test_splits["intervention_id"].nunique()),
        "test_predictions": int(
            len(evaluated[evaluated["outcome"].isin(PRIMARY_OUTCOMES)])
        ),
        "models": list(MODEL_ORDER),
        "primary_outcomes": list(PRIMARY_OUTCOMES),
        "chronological_no_test_leakage": bool(chronological),
        "best_model_by_primary_mae": best_model,
        "best_model_by_direction_accuracy": best_direction_model,
        "bdmtf_primary_mae": float(bdmtf["mae"]),
        "bdmtf_primary_mae_ci": [
            float(bdmtf["mae_ci_low"]),
            float(bdmtf["mae_ci_high"]),
        ],
        "bdmtf_direction_accuracy": float(bdmtf["direction_accuracy"]),
        "bdmtf_direction_accuracy_ci": [
            float(bdmtf["direction_accuracy_ci_low"]),
            float(bdmtf["direction_accuracy_ci_high"]),
        ],
        "bdmtf_interval_coverage": float(bdmtf["interval_coverage"]),
        "bdmtf_interval_coverage_ci": [
            float(bdmtf["interval_coverage_ci_low"]),
            float(bdmtf["interval_coverage_ci_high"]),
        ],
        "bdmtf_vs_additive_mae_improvement": float(
            additive_comparison["mean_absolute_error_improvement"]
        ),
        "bdmtf_vs_additive_mae_improvement_ci": [
            float(additive_comparison["mae_improvement_ci_low"]),
            float(additive_comparison["mae_improvement_ci_high"]),
        ],
        "bdmtf_vs_frozen_mechanism_mae_improvement": float(
            mechanism_comparison["mean_absolute_error_improvement"]
        ),
        "bdmtf_vs_frozen_mechanism_mae_improvement_ci": [
            float(mechanism_comparison["mae_improvement_ci_low"]),
            float(mechanism_comparison["mae_improvement_ci_high"]),
        ],
        "framework_support": (
            "supported"
            if float(additive_comparison["mae_improvement_ci_low"]) > 0
            else "competitive_but_not_significantly_better"
            if float(additive_comparison["mean_absolute_error_improvement"]) >= 0
            else "not_supported_by_primary_mae"
        ),
        "observed_confirmatory_att": {
            str(item["outcome"]): {
                "effect": float(item["effect"]),
                "ci_low": float(item["ci_low"]),
                "ci_high": float(item["ci_high"]),
            }
            for item in natural.get("primary_diagnostics", [])
        },
        "evidence_boundary": (
            "The frozen mechanism model uses no post-intervention outcomes. "
            "The calibrated adapters use outcomes from earlier interventions but "
            "never read validation or chronological test outcomes during fitting. "
            "This is retrospective temporal validation, not a prospective trial."
        ),
        "inputs": {
            "panel": str(panel_path.resolve()),
            "panel_sha256": sha256_file(panel_path),
            "matches": str(matches_path.resolve()),
            "matches_sha256": sha256_file(matches_path),
            "natural_experiment": str(natural_experiment_path.resolve()),
            "natural_experiment_sha256": sha256_file(natural_experiment_path),
            "panel_frame_sha256": _frame_hash(panel),
        },
    }
    write_json(output_dir / "intervention_fidelity.json", result)
    return result
