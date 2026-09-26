from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import optimize, stats

from bdmtf.revision.provenance import sha256_file, write_json


PANEL_COLUMNS = (
    "unit_id",
    "period",
    "treated",
    "relative_period",
    "outcome",
    "value",
)

RCT_COLUMNS = (
    "participant_id",
    "arm",
    "ranking",
    "context",
    "correction",
    "consent",
    "eligible",
    "completed",
)


def panel_readiness(frame: pd.DataFrame) -> dict[str, Any]:
    missing = [name for name in PANEL_COLUMNS if name not in frame]
    issues: list[str] = []
    if missing:
        issues.append(f"missing columns: {', '.join(missing)}")
    if not missing:
        treated = pd.to_numeric(frame["treated"], errors="coerce")
        relative = pd.to_numeric(frame["relative_period"], errors="coerce")
        values = pd.to_numeric(frame["value"], errors="coerce")
        periods = pd.to_datetime(frame["period"], utc=True, errors="coerce")
        if treated.isna().any() or not set(treated.dropna().astype(int)).issubset({0, 1}):
            issues.append("treated must be binary")
        if treated.nunique() < 2:
            issues.append("both treated and comparison units are required")
        if relative.isna().any():
            issues.append("relative_period must be numeric")
        if values.isna().any():
            issues.append("value must be numeric")
        if periods.isna().any():
            issues.append("period contains invalid timestamps")
        if not (relative < 0).any() or not (relative >= 0).any():
            issues.append("both pre- and post-intervention periods are required")
        duplicate_key = ["unit_id", "period", "outcome"]
        if frame.duplicated(duplicate_key).any():
            issues.append("unit-period-outcome keys must be unique")
    return {
        "status": "ready" if not issues else "invalid",
        "claim_allowed": not issues,
        "issues": issues,
        "n_rows": int(len(frame)),
        "n_units": int(frame["unit_id"].nunique()) if "unit_id" in frame else 0,
    }


def _cluster_bootstrap_did(
    frame: pd.DataFrame,
    samples: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    changes = pd.DataFrame(
        [
            {
                "treated": int(treated),
                "change": _unit_change(group),
            }
            for (_, treated), group in frame.groupby(
                ["unit_id", "treated"],
                sort=False,
            )
        ]
    )
    treated_changes = changes.loc[
        changes["treated"].eq(1),
        "change",
    ].to_numpy(float)
    control_changes = changes.loc[
        changes["treated"].eq(0),
        "change",
    ].to_numpy(float)
    treated_draws = rng.integers(
        0,
        len(treated_changes),
        size=(samples, len(treated_changes)),
    )
    control_draws = rng.integers(
        0,
        len(control_changes),
        size=(samples, len(control_changes)),
    )
    return (
        treated_changes[treated_draws].mean(axis=1)
        - control_changes[control_draws].mean(axis=1)
    )


def _unit_change(group: pd.DataFrame) -> float:
    pre = group.loc[group["relative_period"] < 0, "value"].astype(float)
    post = group.loc[group["relative_period"] >= 0, "value"].astype(float)
    return float(post.mean() - pre.mean())


def estimate_panel_did(
    frame: pd.DataFrame,
    bootstrap_samples: int = 1000,
    seed: int = 30371,
) -> dict[str, Any]:
    readiness = panel_readiness(frame)
    if not readiness["claim_allowed"]:
        raise ValueError(f"Panel is not analysis-ready: {readiness['issues']}")
    outcomes = frame["outcome"].astype(str).unique()
    if len(outcomes) != 1:
        raise ValueError("estimate_panel_did accepts exactly one outcome at a time")
    unit = pd.DataFrame(
        [
            {"unit_id": unit_id, "treated": treated, "change": _unit_change(group)}
            for (unit_id, treated), group in frame.assign(value=pd.to_numeric(frame["value"])).groupby(
                ["unit_id", "treated"]
            )
        ]
    )
    treated_changes = unit.loc[unit["treated"].astype(int) == 1, "change"].to_numpy(float)
    control_changes = unit.loc[unit["treated"].astype(int) == 0, "change"].to_numpy(float)
    effect = float(treated_changes.mean() - control_changes.mean())
    bootstrap = _cluster_bootstrap_did(frame, bootstrap_samples, seed)
    event_study = event_study_table(frame)
    pre = event_study[event_study["relative_period"] < 0]
    pretrend_slope = float(np.polyfit(pre["relative_period"], pre["difference"], 1)[0]) if len(pre) >= 2 else np.nan
    overlap = _outcome_overlap(frame)
    placebo = _placebo_effect(frame)
    return {
        **readiness,
        "estimator": "unit-cluster bootstrap difference-in-differences",
        "estimand": "ATT under parallel trends and no anticipation",
        "outcome": str(outcomes[0]),
        "effect": effect,
        "ci_low": float(np.quantile(bootstrap, 0.025)),
        "ci_high": float(np.quantile(bootstrap, 0.975)),
        "bootstrap_samples": int(bootstrap_samples),
        "pretrend_slope": pretrend_slope,
        "pretrend_flag": bool(np.isfinite(pretrend_slope) and abs(pretrend_slope) > 0.1),
        "placebo_effect": placebo,
        "pre_outcome_overlap": overlap,
        "causal_caveat": (
            "The estimate is causal only if treatment timing is well measured, parallel trends and overlap hold, "
            "and no unmeasured time-varying confounding or spillover remains."
        ),
    }


def event_study_table(frame: pd.DataFrame, reference_period: int = -1) -> pd.DataFrame:
    grouped = (
        frame.assign(
            treated=pd.to_numeric(frame["treated"]).astype(int),
            relative_period=pd.to_numeric(frame["relative_period"]).astype(int),
            value=pd.to_numeric(frame["value"]),
        )
        .groupby(["relative_period", "treated"])["value"]
        .agg(["mean", "count", "std"])
        .reset_index()
    )
    pivot = grouped.pivot(index="relative_period", columns="treated", values="mean")
    counts = grouped.pivot(index="relative_period", columns="treated", values="count")
    result = pd.DataFrame(
        {
            "relative_period": pivot.index,
            "treated_mean": pivot.get(1),
            "control_mean": pivot.get(0),
            "treated_n": counts.get(1),
            "control_n": counts.get(0),
        }
    ).reset_index(drop=True)
    result["difference"] = result["treated_mean"] - result["control_mean"]
    reference = result.loc[result["relative_period"] == reference_period, "difference"]
    baseline = float(reference.iloc[0]) if not reference.empty else float(result[result["relative_period"] < 0]["difference"].mean())
    result["event_study_effect"] = result["difference"] - baseline
    return result


def _outcome_overlap(frame: pd.DataFrame) -> dict[str, float]:
    pre = (
        frame[frame["relative_period"] < 0]
        .groupby(["unit_id", "treated"])["value"]
        .mean()
        .reset_index()
    )
    treated = pre.loc[pre["treated"].astype(int) == 1, "value"].astype(float)
    control = pre.loc[pre["treated"].astype(int) == 0, "value"].astype(float)
    lower = max(float(treated.min()), float(control.min()))
    upper = min(float(treated.max()), float(control.max()))
    covered = float(((treated >= lower) & (treated <= upper)).mean()) if upper >= lower else 0.0
    return {"lower": lower, "upper": upper, "treated_fraction_in_overlap": covered}


def _placebo_effect(frame: pd.DataFrame) -> float | None:
    pre = frame[frame["relative_period"] < 0].copy()
    periods = sorted(pre["relative_period"].astype(int).unique())
    if len(periods) < 4:
        return None
    midpoint = periods[len(periods) // 2]
    pre["relative_period"] = pre["relative_period"].astype(int) - midpoint
    if not (pre["relative_period"] < 0).any() or not (pre["relative_period"] >= 0).any():
        return None
    unit = pd.DataFrame(
        [
            {"unit_id": unit_id, "treated": treated, "change": _unit_change(group)}
            for (unit_id, treated), group in pre.groupby(["unit_id", "treated"])
        ]
    )
    treated = unit.loc[unit["treated"].astype(int) == 1, "change"]
    control = unit.loc[unit["treated"].astype(int) == 0, "change"]
    return float(treated.mean() - control.mean())


def controlled_interrupted_time_series(frame: pd.DataFrame) -> dict[str, Any]:
    readiness = panel_readiness(frame)
    if not readiness["claim_allowed"]:
        raise ValueError(f"Panel is not analysis-ready: {readiness['issues']}")
    aggregated = (
        frame.groupby(["relative_period", "treated"], as_index=False)["value"]
        .mean()
        .sort_values(["treated", "relative_period"])
    )
    t = aggregated["relative_period"].to_numpy(float)
    treated = aggregated["treated"].to_numpy(float)
    post = (t >= 0).astype(float)
    design = np.column_stack(
        [
            np.ones(len(aggregated)),
            t,
            treated,
            post,
            t * post,
            treated * post,
            treated * t,
            treated * t * post,
        ]
    )
    coefficients, _, rank, _ = np.linalg.lstsq(design, aggregated["value"].to_numpy(float), rcond=None)
    return {
        "status": "complete",
        "estimator": "controlled segmented least squares",
        "rank": int(rank),
        "treated_level_change": float(coefficients[5]),
        "treated_slope_change": float(coefficients[7]),
        "coefficient_order": [
            "intercept",
            "time",
            "treated",
            "post",
            "time_post",
            "treated_post",
            "treated_time",
            "treated_time_post",
        ],
    }


def uncontrolled_interrupted_time_series(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"unit_id", "relative_period", "outcome", "value"}
    if required.difference(frame.columns):
        return pd.DataFrame()
    records: list[dict[str, Any]] = []
    for (unit_id, outcome), group in frame.groupby(["unit_id", "outcome"]):
        series = (
            group.assign(
                relative_period=pd.to_numeric(group["relative_period"], errors="coerce"),
                value=pd.to_numeric(group["value"], errors="coerce"),
            )
            .dropna(subset=["relative_period", "value"])
            .groupby("relative_period", as_index=False)["value"]
            .mean()
            .sort_values("relative_period")
        )
        if len(series) < 4:
            continue
        t = series["relative_period"].to_numpy(float)
        if not (t < 0).any() or not (t >= 0).any():
            continue
        post = (t >= 0).astype(float)
        design = np.column_stack([np.ones(len(series)), t, post, t * post])
        coefficients, _, rank, _ = np.linalg.lstsq(
            design,
            series["value"].to_numpy(float),
            rcond=None,
        )
        fitted = design @ coefficients
        records.append(
            {
                "unit_id": str(unit_id),
                "outcome": str(outcome),
                "level_change": float(coefficients[2]),
                "slope_change": float(coefficients[3]),
                "rmse": float(np.sqrt(np.mean((series["value"].to_numpy(float) - fitted) ** 2))),
                "rank": int(rank),
                "n_periods": int(len(series)),
                "causal_claim_allowed": False,
            }
        )
    return pd.DataFrame(records)


def synthetic_control_estimate(frame: pd.DataFrame) -> pd.DataFrame:
    readiness = panel_readiness(frame)
    if not readiness["claim_allowed"]:
        raise ValueError(f"Panel is not analysis-ready: {readiness['issues']}")
    outcomes = frame["outcome"].astype(str).unique()
    if len(outcomes) != 1:
        raise ValueError("synthetic_control_estimate accepts exactly one outcome at a time")
    prepared = frame.assign(
        treated=pd.to_numeric(frame["treated"]).astype(int),
        relative_period=pd.to_numeric(frame["relative_period"]).astype(int),
        value=pd.to_numeric(frame["value"]),
        unit_id=frame["unit_id"].astype(str),
    )
    treated_units = prepared.loc[prepared["treated"] == 1, "unit_id"].unique().tolist()
    control_units = prepared.loc[prepared["treated"] == 0, "unit_id"].unique().tolist()
    records: list[dict[str, Any]] = []
    if not treated_units or not control_units:
        return pd.DataFrame()
    pivot = prepared.pivot_table(
        index="relative_period",
        columns="unit_id",
        values="value",
        aggfunc="mean",
    )
    for treated_unit in treated_units:
        selected_columns = [treated_unit, *control_units]
        pre = pivot.loc[pivot.index < 0, selected_columns].dropna()
        post = pivot.loc[pivot.index >= 0, selected_columns].dropna()
        if len(pre) < 2 or post.empty:
            continue
        target = pre[treated_unit].to_numpy(float)
        donors = pre[control_units].to_numpy(float)
        initial = np.repeat(1.0 / len(control_units), len(control_units))
        fit = optimize.minimize(
            lambda weights: float(np.mean((target - donors @ weights) ** 2)),
            initial,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * len(control_units),
            constraints=[{"type": "eq", "fun": lambda weights: float(weights.sum() - 1.0)}],
            options={"maxiter": 1000, "ftol": 1e-12},
        )
        if not fit.success:
            continue
        weights = fit.x
        pre_gap = target - donors @ weights
        post_gap = post[treated_unit].to_numpy(float) - post[control_units].to_numpy(float) @ weights
        records.append(
            {
                "unit_id": treated_unit,
                "outcome": str(outcomes[0]),
                "effect": float(post_gap.mean()),
                "pre_rmse": float(np.sqrt(np.mean(pre_gap**2))),
                "post_rmse": float(np.sqrt(np.mean(post_gap**2))),
                "max_donor_weight": float(weights.max()),
                "effective_donors": float(1.0 / np.sum(weights**2)),
                "donor_count": int(len(control_units)),
                "donor_weights": json.dumps(
                    {unit: float(weight) for unit, weight in zip(control_units, weights) if weight > 1e-8},
                    sort_keys=True,
                ),
            }
        )
    return pd.DataFrame(records)


def _calibrated_control_weights(
    donor_features: np.ndarray,
    target_features: np.ndarray,
    ridge: float,
    scale_floor: float,
    max_iterations: int = 1000,
    tolerance: float = 1e-12,
) -> tuple[np.ndarray, dict[str, Any]]:
    donors = np.asarray(donor_features, dtype=float)
    target = np.asarray(target_features, dtype=float)
    if donors.ndim != 2 or target.shape != (donors.shape[0],):
        raise ValueError("Calibration features have incompatible dimensions")
    scale = np.maximum(np.std(donors, axis=1), float(scale_floor))
    design = donors / scale[:, None]
    response = target / scale
    uniform = np.repeat(1.0 / donors.shape[1], donors.shape[1])
    def objective(weights: np.ndarray) -> float:
        residual = design @ weights - response
        return float(
            np.mean(residual**2)
            + float(ridge) * np.sum((weights - uniform) ** 2)
        )

    def gradient(weights: np.ndarray) -> np.ndarray:
        return (
            2.0
            * design.T
            @ (design @ weights - response)
            / max(1, design.shape[0])
            + 2.0 * float(ridge) * (weights - uniform)
        )

    fitted = optimize.minimize(
        objective,
        uniform,
        jac=gradient,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * len(uniform),
        constraints=[
            {
                "type": "eq",
                "fun": lambda weights: float(weights.sum() - 1.0),
                "jac": lambda weights: np.ones_like(weights),
            }
        ],
        options={
            "maxiter": int(max_iterations),
            "ftol": float(tolerance),
        },
    )
    if not fitted.success:
        raise RuntimeError(
            "Calibrated control optimization failed: "
            + str(fitted.message)
        )
    weights = fitted.x
    residual = design @ weights - response
    return weights, {
        "optimizer": "SLSQP",
        "optimizer_converged": True,
        "optimizer_message": str(fitted.message),
        "iterations": int(fitted.nit),
        "standardized_pre_rmse": float(
            np.sqrt(np.mean(residual**2))
        ),
        "maximum_control_weight": float(weights.max()),
        "effective_controls": float(1.0 / np.sum(weights**2)),
        "ridge": float(ridge),
        "scale_floor": float(scale_floor),
    }


def aggregate_calibrated_synthetic_control(
    panel: pd.DataFrame,
    balance_outcomes: Iterable[str] | None = None,
    ridge: float = 0.1,
    scale_floor: float = 0.1,
    bootstrap_samples: int = 1000,
    seed: int = 30371,
    pretrend_threshold: float = 0.1,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Calibrate one shared control pool using only pre-intervention trajectories."""
    readiness = panel_readiness(panel)
    if not readiness["claim_allowed"]:
        raise ValueError(
            f"Panel is not analysis-ready: {readiness['issues']}"
        )
    prepared = panel.assign(
        unit_id=panel["unit_id"].astype(str),
        treated=pd.to_numeric(panel["treated"]).astype(int),
        relative_period=pd.to_numeric(
            panel["relative_period"]
        ).astype(int),
        value=pd.to_numeric(panel["value"]).astype(float),
        outcome=panel["outcome"].astype(str),
    )
    outcomes = sorted(prepared["outcome"].unique())
    requested = (
        list(balance_outcomes)
        if balance_outcomes is not None
        else outcomes
    )
    missing = sorted(set(requested).difference(outcomes))
    if missing:
        raise ValueError(
            "Calibration outcomes are missing: " + ", ".join(missing)
        )
    treated_ids = sorted(
        prepared.loc[prepared["treated"].eq(1), "unit_id"].unique()
    )
    control_ids = sorted(
        prepared.loc[prepared["treated"].eq(0), "unit_id"].unique()
    )
    pre_periods = sorted(
        prepared.loc[
            prepared["relative_period"] < 0,
            "relative_period",
        ].unique()
    )
    all_periods = sorted(prepared["relative_period"].unique())
    pivots: dict[str, pd.DataFrame] = {}
    for outcome in outcomes:
        pivot = prepared[
            prepared["outcome"].eq(outcome)
        ].pivot_table(
            index="relative_period",
            columns="unit_id",
            values="value",
            aggfunc="mean",
        )
        required_columns = set(treated_ids + control_ids)
        if not required_columns.issubset(pivot.columns):
            raise ValueError(
                f"{outcome} does not contain every treated and control unit"
            )
        pivots[outcome] = pivot.reindex(all_periods)
    target_features = np.concatenate(
        [
            pivots[outcome]
            .loc[pre_periods, treated_ids]
            .to_numpy(float)
            .mean(axis=1)
            for outcome in requested
        ]
    )
    donor_features = np.vstack(
        [
            pivots[outcome]
            .loc[pre_periods, control_ids]
            .to_numpy(float)
            for outcome in requested
        ]
    )
    weights, fit = _calibrated_control_weights(
        donor_features,
        target_features,
        ridge,
        scale_floor,
    )
    weight_table = pd.DataFrame(
        {
            "control_unit_id": control_ids,
            "weight": weights,
        }
    ).sort_values(
        ["weight", "control_unit_id"],
        ascending=[False, True],
    )
    rng = np.random.default_rng(seed)
    treated_multipliers = rng.exponential(
        1.0,
        size=(int(bootstrap_samples), len(treated_ids)),
    )
    treated_multipliers /= treated_multipliers.sum(
        axis=1,
        keepdims=True,
    )
    control_multipliers = rng.exponential(
        1.0,
        size=(int(bootstrap_samples), len(control_ids)),
    )
    control_multipliers *= weights[None, :]
    control_multipliers /= control_multipliers.sum(
        axis=1,
        keepdims=True,
    )
    diagnostics: list[dict[str, Any]] = []
    event_tables: list[pd.DataFrame] = []
    for outcome in outcomes:
        pivot = pivots[outcome]
        treated_values = pivot.loc[
            all_periods,
            treated_ids,
        ].to_numpy(float)
        control_values = pivot.loc[
            all_periods,
            control_ids,
        ].to_numpy(float)
        treated_mean = treated_values.mean(axis=1)
        control_mean = control_values @ weights
        difference = treated_mean - control_mean
        event_table = pd.DataFrame(
            {
                "outcome": outcome,
                "relative_period": all_periods,
                "treated_mean": treated_mean,
                "calibrated_control_mean": control_mean,
                "difference": difference,
            }
        )
        event_tables.append(event_table)
        pre_mask = np.asarray(all_periods) < 0
        post_mask = ~pre_mask
        pre_gap = float(difference[pre_mask].mean())
        effect = float(
            difference[post_mask].mean() - pre_gap
        )
        treated_changes = (
            treated_values[post_mask].mean(axis=0)
            - treated_values[pre_mask].mean(axis=0)
        )
        control_changes = (
            control_values[post_mask].mean(axis=0)
            - control_values[pre_mask].mean(axis=0)
        )
        bootstrap = (
            treated_multipliers @ treated_changes
            - control_multipliers @ control_changes
        )
        pretrend_slope = float(
            np.polyfit(
                np.asarray(all_periods)[pre_mask],
                difference[pre_mask],
                1,
            )[0]
        )
        overlap = _outcome_overlap(
            prepared[prepared["outcome"].eq(outcome)]
        )
        diagnostics.append(
            {
                "outcome": outcome,
                "effect": effect,
                "ci_low": float(np.quantile(bootstrap, 0.025)),
                "ci_high": float(np.quantile(bootstrap, 0.975)),
                "pretrend_slope": pretrend_slope,
                "pretrend_flag": bool(
                    abs(pretrend_slope) > float(pretrend_threshold)
                ),
                "treated_fraction_in_overlap": overlap[
                    "treated_fraction_in_overlap"
                ],
            }
        )
    return (
        {
            "status": "complete",
            "estimator": (
                "aggregate pre-outcome calibrated synthetic control "
                "with fixed-weight unit-cluster Bayesian bootstrap"
            ),
            "estimand": "calibrated ATT under conditional parallel trends",
            "balance_outcomes": requested,
            "pre_periods": [int(value) for value in pre_periods],
            "post_periods": [
                int(value)
                for value in all_periods
                if int(value) >= 0
            ],
            "bootstrap_samples": int(bootstrap_samples),
            "pretrend_threshold": float(pretrend_threshold),
            **fit,
            "diagnostics": diagnostics,
            "weight_training_rule": (
                "Control weights use only treated and control outcomes "
                "with relative_period < 0; post outcomes are never read "
                "during calibration."
            ),
        },
        pd.concat(event_tables, ignore_index=True),
        weight_table,
    )


def fit_pre_intervention_forecast(frame: pd.DataFrame) -> pd.DataFrame:
    """Fit transparent pre-period linear forecasts without reading post-treatment rows."""
    required = {"unit_id", "relative_period", "outcome", "value"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Forecast panel is missing columns: {', '.join(missing)}")
    numeric_period = pd.to_numeric(frame["relative_period"], errors="coerce")
    numeric_value = pd.to_numeric(frame["value"], errors="coerce")
    if numeric_period.isna().any() or numeric_value.isna().any():
        raise ValueError("Forecast panel contains invalid relative_period or value entries")
    prepared = frame.assign(relative_period=numeric_period, value=numeric_value)
    records = []
    for (unit_id, outcome), group in prepared[prepared["relative_period"] < 0].groupby(
        ["unit_id", "outcome"]
    ):
        x = group["relative_period"].to_numpy(float)
        y = group["value"].to_numpy(float)
        if len(group) < 2:
            continue
        slope, intercept = np.polyfit(x, y, 1)
        records.append(
            {
                "unit_id": str(unit_id),
                "outcome": str(outcome),
                "intercept": float(intercept),
                "slope": float(slope),
                "n_pre_periods": int(len(group)),
                "max_training_period": int(group["relative_period"].max()),
            }
        )
    return pd.DataFrame(records)


def run_natural_experiments(
    panel: pd.DataFrame,
    output_dir: Path,
    bootstrap_samples: int = 1000,
    seed: int = 30371,
    r_script: Path | None = None,
    run_synthetic_control: bool = True,
    aggregate_synthetic_control: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    readiness = panel_readiness(panel)
    write_json(output_dir / "panel_readiness.json", readiness)
    uncontrolled_its = uncontrolled_interrupted_time_series(panel)
    if not uncontrolled_its.empty:
        uncontrolled_its.to_csv(output_dir / "uncontrolled_its.csv", index=False)
    if not readiness["claim_allowed"]:
        result = {
            **readiness,
            "status": "descriptive_only" if not uncontrolled_its.empty else "needs_controls_or_panel",
            "resolved": False,
            "reason": "Natural-experiment claims are disabled until treated and comparison panels pass audit.",
            "uncontrolled_its_series": int(len(uncontrolled_its)),
        }
        write_json(output_dir / "natural_experiment_summary.json", result)
        return result
    estimates = []
    event_tables = []
    controlled_its_records = []
    synthetic_records = []
    for outcome, group in panel.groupby("outcome"):
        estimate = estimate_panel_did(group, bootstrap_samples, seed)
        estimates.append(estimate)
        table = event_study_table(group)
        table.insert(0, "outcome", outcome)
        event_tables.append(table)
        controlled_its_records.append({"outcome": outcome, **controlled_interrupted_time_series(group)})
        if run_synthetic_control:
            synthetic = synthetic_control_estimate(group)
            if not synthetic.empty:
                synthetic_records.append(synthetic)
    pd.DataFrame(estimates).to_csv(output_dir / "did_estimates.csv", index=False)
    pd.concat(event_tables, ignore_index=True).to_csv(output_dir / "event_study.csv", index=False)
    pd.DataFrame(controlled_its_records).to_csv(output_dir / "controlled_its.csv", index=False)
    if synthetic_records:
        pd.concat(synthetic_records, ignore_index=True).to_csv(
            output_dir / "synthetic_control_estimates.csv",
            index=False,
        )
    aggregate_result = None
    aggregate_config = aggregate_synthetic_control or {}
    if bool(aggregate_config.get("enabled", False)):
        (
            aggregate_result,
            aggregate_events,
            aggregate_weights,
        ) = aggregate_calibrated_synthetic_control(
            panel,
            aggregate_config.get("balance_outcomes"),
            float(aggregate_config.get("ridge", 0.1)),
            float(aggregate_config.get("scale_floor", 0.1)),
            int(
                aggregate_config.get(
                    "bootstrap_samples",
                    bootstrap_samples,
                )
            ),
            int(aggregate_config.get("seed", seed)),
            float(
                aggregate_config.get(
                    "pretrend_threshold",
                    0.1,
                )
            ),
        )
        aggregate_events.to_csv(
            output_dir / "aggregate_calibrated_event_study.csv",
            index=False,
        )
        aggregate_weights.to_csv(
            output_dir / "aggregate_calibrated_control_weights.csv",
            index=False,
        )
        write_json(
            output_dir / "aggregate_calibrated_summary.json",
            aggregate_result,
        )
    r_result = _run_r_reference(panel, output_dir, r_script) if r_script else {"status": "not_requested"}
    result = {
        "status": "complete",
        "resolved": True,
        "claim_allowed": True,
        "n_outcomes": len(estimates),
        "n_units": int(panel["unit_id"].nunique()),
        "diagnostics": [
            {
                "outcome": item["outcome"],
                "effect": item["effect"],
                "ci_low": item["ci_low"],
                "ci_high": item["ci_high"],
                "pretrend_flag": item["pretrend_flag"],
                "placebo_effect": item["placebo_effect"],
                "treated_fraction_in_overlap": item[
                    "pre_outcome_overlap"
                ]["treated_fraction_in_overlap"],
            }
            for item in estimates
        ],
        "estimators": [
            "unit-cluster bootstrap DiD",
            "event study",
            "controlled interrupted time series",
            *(
                ["nonnegative simplex synthetic control"]
                if run_synthetic_control
                else []
            ),
        ],
        "synthetic_control_units": int(sum(len(frame) for frame in synthetic_records)),
        "aggregate_synthetic_control": aggregate_result,
        "uncontrolled_its_series": int(len(uncontrolled_its)),
        "r_reference": r_result,
        "note": "Substantive claims still depend on the saved diagnostics for each outcome.",
    }
    write_json(output_dir / "natural_experiment_summary.json", result)
    return result


def _run_r_reference(panel: pd.DataFrame, output_dir: Path, script: Path | None) -> dict[str, Any]:
    if not script or not script.is_file():
        return {"status": "missing_script"}
    input_path = output_dir / "r_input_panel.csv"
    panel.to_csv(input_path, index=False)
    try:
        completed = subprocess.run(
            ["Rscript", str(script), str(input_path), str(output_dir / "r_reference")],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=3600,
        )
    except FileNotFoundError:
        return {"status": "unavailable", "reason": "Rscript executable not found"}
    except subprocess.TimeoutExpired:
        return {"status": "unavailable", "reason": "R reference analysis exceeded 3600 seconds"}
    return {
        "status": "complete" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-2000:],
    }


def rct_readiness(frame: pd.DataFrame, protocol: dict[str, Any]) -> dict[str, Any]:
    missing = [name for name in RCT_COLUMNS if name not in frame]
    issues: list[str] = []
    if missing:
        issues.append(f"missing columns: {', '.join(missing)}")
    if not protocol.get("ethics_approval_id"):
        issues.append("ethics approval ID is missing")
    if not protocol.get("preregistration_url"):
        issues.append("preregistration URL is missing")
    if not protocol.get("frozen_primary_outcomes"):
        issues.append("primary outcomes are not frozen")
    if not missing:
        eligible = frame["eligible"].astype(bool) & frame["consent"].astype(bool)
        if eligible.sum() == 0:
            issues.append("no eligible consenting participants")
        if frame.loc[eligible, "arm"].nunique() < 2:
            issues.append("fewer than two randomized arms")
        duplicates = frame.loc[eligible, "participant_id"].astype(str).duplicated()
        if duplicates.any():
            issues.append("participant_id is not unique")
    return {
        "status": "ready" if not issues else "blocked",
        "claim_allowed": not issues,
        "issues": issues,
        "n_records": int(len(frame)),
        "external_prerequisites": [
            "ethics approval",
            "informed consent",
            "preregistration",
            "participant recruitment and compensation",
        ],
    }


def analyze_human_rct(
    frame: pd.DataFrame,
    protocol: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    readiness = rct_readiness(frame, protocol)
    write_json(output_dir / "rct_readiness.json", readiness)
    if not readiness["claim_allowed"]:
        result = {**readiness, "resolved": False}
        write_json(output_dir / "rct_summary.json", result)
        return result
    eligible = frame[frame["eligible"].astype(bool) & frame["consent"].astype(bool)].copy()
    completion = eligible.groupby("arm")["completed"].agg(["mean", "count"]).reset_index()
    completion.to_csv(output_dir / "attrition_by_arm.csv", index=False)
    balance = _randomization_balance(eligible, protocol.get("baseline_covariates", []))
    balance.to_csv(output_dir / "randomization_balance.csv", index=False)
    estimates = []
    outcomes = list(protocol["frozen_primary_outcomes"]) + list(protocol.get("secondary_outcomes", []))
    reference = str(protocol.get("reference_arm") or sorted(eligible["arm"].astype(str).unique())[0])
    completed = eligible[eligible["completed"].astype(bool)]
    for outcome in outcomes:
        if outcome not in completed:
            continue
        reference_values = pd.to_numeric(completed.loc[completed["arm"].astype(str) == reference, outcome], errors="coerce").dropna()
        for arm, group in completed.groupby("arm"):
            if str(arm) == reference:
                continue
            values = pd.to_numeric(group[outcome], errors="coerce").dropna()
            if not len(values) or not len(reference_values):
                continue
            test = stats.ttest_ind(values, reference_values, equal_var=False)
            estimates.append(
                {
                    "outcome": outcome,
                    "arm": str(arm),
                    "reference_arm": reference,
                    "effect": float(values.mean() - reference_values.mean()),
                    "p_value": float(test.pvalue),
                    "n_arm": int(len(values)),
                    "n_reference": int(len(reference_values)),
                }
            )
    estimates_frame = pd.DataFrame(estimates)
    if not estimates_frame.empty:
        estimates_frame["holm_p_value"] = _holm(estimates_frame["p_value"].to_numpy(float))
        estimates_frame.to_csv(output_dir / "rct_estimates.csv", index=False)
    result = {
        **readiness,
        "status": "complete",
        "resolved": True,
        "n_eligible": int(len(eligible)),
        "n_completed": int(len(completed)),
        "reference_arm": reference,
        "n_contrasts": int(len(estimates_frame)),
        "ethics_approval_id": protocol["ethics_approval_id"],
        "preregistration_url": protocol["preregistration_url"],
    }
    write_json(output_dir / "rct_summary.json", result)
    return result


def _randomization_balance(frame: pd.DataFrame, covariates: Iterable[str]) -> pd.DataFrame:
    records = []
    for covariate in covariates:
        if covariate not in frame:
            continue
        values = pd.to_numeric(frame[covariate], errors="coerce")
        for arm, group in frame.assign(_value=values).groupby("arm"):
            records.append(
                {
                    "covariate": covariate,
                    "arm": str(arm),
                    "mean": float(group["_value"].mean()),
                    "std": float(group["_value"].std()),
                    "n": int(group["_value"].notna().sum()),
                }
            )
    return pd.DataFrame(records)


def _holm(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (len(p_values) - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def evaluate_intervention_fidelity(
    predictions: pd.DataFrame,
    observed: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Any]:
    required_predictions = {"intervention_id", "outcome", "predicted_effect", "ci_low", "ci_high"}
    required_observed = {"intervention_id", "outcome", "observed_effect"}
    missing = sorted(required_predictions.difference(predictions.columns) | required_observed.difference(observed.columns))
    output_dir.mkdir(parents=True, exist_ok=True)
    if missing:
        result = {
            "status": "invalid",
            "claim_allowed": False,
            "missing_columns": missing,
        }
        write_json(output_dir / "intervention_fidelity.json", result)
        return result
    merged = predictions.merge(observed, on=["intervention_id", "outcome"], how="inner")
    if merged.empty:
        result = {"status": "needs_overlap", "claim_allowed": False, "n_comparisons": 0}
        write_json(output_dir / "intervention_fidelity.json", result)
        return result
    merged["absolute_error"] = (merged["predicted_effect"] - merged["observed_effect"]).abs()
    merged["direction_correct"] = np.sign(merged["predicted_effect"]) == np.sign(merged["observed_effect"])
    merged["interval_covers"] = (
        (merged["observed_effect"] >= merged["ci_low"])
        & (merged["observed_effect"] <= merged["ci_high"])
    )
    merged.to_csv(output_dir / "intervention_fidelity_pairs.csv", index=False)
    result = {
        "status": "complete",
        "claim_allowed": True,
        "n_comparisons": int(len(merged)),
        "direction_accuracy": float(merged["direction_correct"].mean()),
        "mae": float(merged["absolute_error"].mean()),
        "interval_coverage": float(merged["interval_covers"].mean()),
        "prediction_sha256": _frame_hash(predictions),
        "observed_sha256": _frame_hash(observed),
    }
    write_json(output_dir / "intervention_fidelity.json", result)
    return result


def _frame_hash(frame: pd.DataFrame) -> str:
    return __import__("hashlib").sha256(
        pd.util.hash_pandas_object(frame.sort_index(axis=1), index=True).values.tobytes()
    ).hexdigest()
