from __future__ import annotations

import hashlib
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import requests
from scipy.optimize import minimize
from scipy.stats import linregress

from bdmtf.revision.provenance import sha256_file, write_json


def _request_json(
    url: str,
    params: Mapping[str, Any],
    retries: int = 5,
) -> Any:
    headers = {"User-Agent": "bdmtf-tbbt-qualified-controls/1.0"}
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.get(
                url,
                params=dict(params),
                headers=headers,
                timeout=60,
            )
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            time.sleep(0.5 * (2**attempt))
    raise RuntimeError(f"Time-series request failed: {params}") from last_error


def fetch_arctic_shift_series(
    base_url: str,
    subreddit: str,
    metric: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cache_dir: Path,
) -> pd.Series:
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(
        f"{subreddit}|{metric}|{start.date()}|{end.date()}".encode("utf-8")
    ).hexdigest()[:20]
    cache_path = cache_dir / f"{key}.json"
    params = {
        "key": f"r/{subreddit}/comments/{metric}",
        "precision": "day",
        "after": start.strftime("%Y-%m-%d"),
        "before": (end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
    }
    if cache_path.is_file():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        payload = _request_json(base_url, params)
        temporary = cache_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(
                {
                    "request": params,
                    "response": payload,
                    "retrieved_at": datetime.now(timezone.utc).isoformat(),
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        temporary.replace(cache_path)
    if "response" in payload:
        payload = payload["response"]
    records = payload.get("data", []) if isinstance(payload, dict) else []
    values = {
        pd.to_datetime(int(record["date"]), unit="s", utc=True).floor("D"):
        float(record["value"])
        for record in records
        if record.get("date") is not None and record.get("value") is not None
    }
    index = pd.date_range(start.floor("D"), end.floor("D"), freq="D", tz="UTC")
    return pd.Series(values, index=index, dtype=float).reindex(index).fillna(0.0)


def _log_count(values: np.ndarray | pd.Series) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return np.log1p(np.clip(array, 0.0, None))


def _simplex_weights(
    target: np.ndarray,
    donors: np.ndarray,
    ridge: float,
) -> np.ndarray:
    count = donors.shape[1]
    initial = np.repeat(1.0 / count, count)
    target_centered = target - target.mean()
    donor_centered = donors - donors.mean(axis=0, keepdims=True)
    result = minimize(
        lambda weights: float(
            np.mean((target_centered - donor_centered @ weights) ** 2)
            + float(ridge) * np.sum(weights**2)
        ),
        initial,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * count,
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
        options={"maxiter": 2000, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"Synthetic-control optimization failed: {result.message}")
    weights = np.clip(np.asarray(result.x, dtype=float), 0.0, None)
    return weights / weights.sum()


def _predict_with_weights(
    target_pre: np.ndarray,
    donor_pre: np.ndarray,
    donors: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    intercept = float(target_pre.mean() - donor_pre.mean(axis=0) @ weights)
    return intercept + donors @ weights


def _select_and_fit(
    target: pd.Series,
    donor_frame: pd.DataFrame,
    relative_period: pd.Series,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    train_mask = (
        (relative_period >= -int(config["pre_days"]))
        & (relative_period <= -int(config["validation_days"]) - 2)
    ).to_numpy()
    validation_mask = (
        (relative_period >= -int(config["validation_days"]) - 1)
        & (relative_period <= -2)
    ).to_numpy()
    pre_mask = train_mask | validation_mask
    target_values = _log_count(target.to_numpy(float))
    donor_values = np.log1p(donor_frame.clip(lower=0).to_numpy(float))
    correlations: list[tuple[int, float]] = []
    for index in range(donor_values.shape[1]):
        left = target_values[train_mask]
        right = donor_values[train_mask, index]
        if np.std(right) < 1e-8:
            continue
        correlation = float(np.corrcoef(left, right)[0, 1])
        if np.isfinite(correlation):
            correlations.append((index, correlation))
    correlations.sort(key=lambda item: (-item[1], donor_frame.columns[item[0]]))
    selected_indices = [
        index
        for index, correlation in correlations
        if correlation >= float(config["minimum_pre_correlation"])
    ][: int(config["maximum_donors"])]
    if len(selected_indices) < int(config["minimum_donors"]):
        selected_indices = [
            index for index, _correlation in correlations
        ][: int(config["maximum_donors"])]
    if len(selected_indices) < int(config["minimum_donors"]):
        raise ValueError("Insufficient non-degenerate donor series")

    selected_names = [str(donor_frame.columns[index]) for index in selected_indices]
    selected = donor_values[:, selected_indices]
    target_train_mean = float(target_values[train_mask].mean())
    target_train_scale = max(float(target_values[train_mask].std()), 0.1)
    donor_train_mean = selected[train_mask].mean(axis=0)
    donor_train_scale = np.maximum(selected[train_mask].std(axis=0), 1e-6)
    selected = (
        target_train_mean
        + (selected - donor_train_mean) / donor_train_scale * target_train_scale
    )
    target_scale = max(float(np.std(target_values[train_mask])), 0.1)
    candidates: list[dict[str, Any]] = []
    for ridge in map(float, config["ridge_grid"]):
        weights = _simplex_weights(
            target_values[train_mask],
            selected[train_mask],
            ridge,
        )
        prediction = _predict_with_weights(
            target_values[train_mask],
            selected[train_mask],
            selected,
            weights,
        )
        validation_rmse = float(
            np.sqrt(
                np.mean(
                    (
                        target_values[validation_mask]
                        - prediction[validation_mask]
                    )
                    ** 2
                )
            )
        )
        validation_nrmse = validation_rmse / target_scale
        validation_baseline_rmse = max(
            float(
                np.sqrt(
                    np.mean(
                        (
                            target_values[validation_mask]
                            - target_train_mean
                        )
                        ** 2
                    )
                )
            ),
            1e-8,
        )
        candidates.append(
            {
                "ridge": ridge,
                "validation_nrmse": validation_nrmse,
                "validation_rmspe_ratio": (
                    validation_rmse / validation_baseline_rmse
                ),
                "weights": weights,
            }
        )
    best = min(
        candidates,
        key=lambda item: (item["validation_rmspe_ratio"], item["ridge"]),
    )
    weights = _simplex_weights(
        target_values[pre_mask],
        selected[pre_mask],
        float(best["ridge"]),
    )
    prediction = _predict_with_weights(
        target_values[pre_mask],
        selected[pre_mask],
        selected,
        weights,
    )
    gap = target_values - prediction
    pre_nrmse = float(
        np.sqrt(np.mean(gap[pre_mask] ** 2))
        / max(float(np.std(target_values[pre_mask])), 0.1)
    )
    pre_baseline_rmse = max(
        float(
            np.sqrt(
                np.mean(
                    (
                        target_values[pre_mask]
                        - target_values[pre_mask].mean()
                    )
                    ** 2
                )
            )
        ),
        1e-8,
    )
    pre_rmspe_ratio = float(
        np.sqrt(np.mean(gap[pre_mask] ** 2)) / pre_baseline_rmse
    )
    trend = linregress(
        relative_period.to_numpy(float)[pre_mask],
        gap[pre_mask],
    )
    return {
        "selected_names": selected_names,
        "weights": weights,
        "prediction": prediction,
        "target_transformed": target_values,
        "gap": gap,
        "pre_mask": pre_mask,
        "train_mask": train_mask,
        "validation_mask": validation_mask,
        "ridge": float(best["ridge"]),
        "validation_nrmse": float(best["validation_nrmse"]),
        "validation_rmspe_ratio": float(best["validation_rmspe_ratio"]),
        "pre_nrmse": pre_nrmse,
        "pre_rmspe_ratio": pre_rmspe_ratio,
        "pretrend_slope": float(trend.slope),
        "pretrend_ci_low": float(trend.slope - 1.96 * trend.stderr),
        "pretrend_ci_high": float(trend.slope + 1.96 * trend.stderr),
        "pretrend_p_value": float(trend.pvalue),
        "maximum_weight": float(weights.max()),
        "effective_donors": float(1.0 / np.sum(weights**2)),
        "correlations": {
            str(donor_frame.columns[index]): correlation
            for index, correlation in correlations
        },
    }


def _hac_mean_interval(
    values: np.ndarray,
    max_lag: int,
) -> tuple[float, float, float, float]:
    clean = np.asarray(values, dtype=float)
    estimate = float(clean.mean())
    centered = clean - estimate
    count = len(clean)
    long_run = float(np.dot(centered, centered) / count)
    for lag in range(1, min(int(max_lag), count - 1) + 1):
        covariance = float(
            np.dot(centered[lag:], centered[:-lag]) / count
        )
        long_run += 2.0 * (1.0 - lag / (max_lag + 1.0)) * covariance
    standard_error = math.sqrt(max(long_run, 0.0) / count)
    return (
        estimate,
        estimate - 1.96 * standard_error,
        estimate + 1.96 * standard_error,
        standard_error,
    )


def _source_overlap_audit(
    tbbt: pd.Series,
    arctic: pd.Series,
    relative_period: pd.Series,
    pre_days: int,
    validation_days: int,
) -> dict[str, float]:
    train = (
        (relative_period >= -int(pre_days))
        & (relative_period <= -int(validation_days) - 2)
    ).to_numpy()
    validation = (
        (relative_period >= -int(validation_days) - 1)
        & (relative_period <= -2)
    ).to_numpy()
    left = _log_count(tbbt)
    right = _log_count(arctic)
    train_correlation = float(np.corrcoef(left[train], right[train])[0, 1])
    validation_correlation = float(
        np.corrcoef(left[validation], right[validation])[0, 1]
    )
    full_pre = train | validation
    log_ratio = left[full_pre] - right[full_pre]
    ratio_center = float(np.median(log_ratio))
    ratio_mad = float(np.median(np.abs(log_ratio - ratio_center)))
    return {
        "source_train_correlation": train_correlation,
        "source_validation_correlation": validation_correlation,
        "source_log_ratio_median": ratio_center,
        "source_log_ratio_mad": ratio_mad,
    }


def _placebo_distribution(
    gap: np.ndarray,
    relative_period: pd.Series,
    window_days: int,
) -> np.ndarray:
    relative = relative_period.to_numpy(int)
    means: list[float] = []
    for end in range(-int(window_days) - 8, -7):
        start = end - int(window_days) + 1
        mask = (relative >= start) & (relative <= end)
        if int(mask.sum()) == int(window_days):
            means.append(float(gap[mask].mean()))
    return np.asarray(means, dtype=float)


def _random_effects_meta(
    effects: np.ndarray,
    standard_errors: np.ndarray,
) -> dict[str, float]:
    variances = np.maximum(np.asarray(standard_errors, dtype=float) ** 2, 1e-8)
    effects = np.asarray(effects, dtype=float)
    fixed_weights = 1.0 / variances
    fixed_mean = float(np.sum(fixed_weights * effects) / fixed_weights.sum())
    q = float(np.sum(fixed_weights * (effects - fixed_mean) ** 2))
    degrees = max(len(effects) - 1, 1)
    denominator = float(
        fixed_weights.sum()
        - np.sum(fixed_weights**2) / fixed_weights.sum()
    )
    tau_squared = max(0.0, (q - degrees) / max(denominator, 1e-12))
    weights = 1.0 / (variances + tau_squared)
    estimate = float(np.sum(weights * effects) / weights.sum())
    standard_error = math.sqrt(1.0 / float(weights.sum()))
    return {
        "log_effect": estimate,
        "ci_low": estimate - 1.96 * standard_error,
        "ci_high": estimate + 1.96 * standard_error,
        "standard_error": standard_error,
        "tau_squared": tau_squared,
        "i_squared": max(0.0, 100.0 * (q - degrees) / max(q, 1e-12)),
    }


def _summarize_effect_group(frame: pd.DataFrame) -> dict[str, Any]:
    summary = _random_effects_meta(
        frame["log_effect"].to_numpy(float),
        frame["standard_error"].to_numpy(float),
    )
    summary["percent_effect"] = 100.0 * math.expm1(summary["log_effect"])
    summary["percent_ci_low"] = 100.0 * math.expm1(summary["ci_low"])
    summary["percent_ci_high"] = 100.0 * math.expm1(summary["ci_high"])
    summary["qualified_interventions"] = int(len(frame))
    summary["direction_negative_share"] = float(
        (frame["log_effect"].astype(float) < 0).mean()
    )
    return summary


def run_tbbt_qualified_control_analysis(
    root: str | Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    project = Path(root)
    panel_path = project / str(config["inputs"]["panel"])
    catalog_path = project / str(config["inputs"]["catalog"])
    output = project / str(config["output_dir"])
    cache_dir = project / str(config["cache_dir"])
    output.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(panel_path)
    catalog = pd.read_csv(catalog_path)
    catalog["date"] = pd.to_datetime(catalog["date"], utc=True)
    panel_period = pd.to_datetime(panel["period"], utc=True).dt.floor("D")
    after_rows = panel[
        panel["period_label"].astype(str).str.contains("after", case=False)
    ].copy()
    after_rows["period"] = pd.to_datetime(
        after_rows["period"], utc=True
    ).dt.floor("D")
    inferred_dates = (
        after_rows.groupby("intervention_id")["period"].min()
        + pd.Timedelta(days=1)
    ).to_dict()
    known_treatments = {
        str(row.community).lower(): inferred_dates.get(
            str(row.intervention_id),
            row.date,
        )
        for row in catalog.itertuples(index=False)
    }
    analysis = config["analysis"]
    gates = config["claim_gates"]
    eligible_ids = list(map(str, config["eligible_interventions"]))
    diagnostics: list[dict[str, Any]] = []
    effect_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []

    for intervention_id in eligible_ids:
        metadata = catalog[catalog["intervention_id"].eq(intervention_id)].iloc[0]
        treatment_date = pd.Timestamp(
            inferred_dates.get(intervention_id, metadata["date"])
        ).floor("D")
        start = treatment_date - pd.Timedelta(days=int(analysis["pre_days"]))
        end = treatment_date + pd.Timedelta(days=max(map(int, analysis["post_windows"])))
        treated_rows = panel[
            panel["intervention_id"].astype(str).eq(intervention_id)
            & panel["platform"].astype(str).str.lower().eq("reddit")
            & panel["scope"].astype(str).str.lower().eq("in")
            & panel["outcome"].astype(str).eq("messages")
        ].copy()
        treated_rows["period"] = pd.to_datetime(
            treated_rows["period"], utc=True
        ).dt.floor("D")
        treated_rows = treated_rows[
            treated_rows["period"].between(start, end)
        ]
        index = pd.date_range(start, end, freq="D", tz="UTC")
        treated = (
            treated_rows.groupby("period")["value"]
            .sum()
            .astype(float)
            .reindex(index)
        )
        coverage = float(treated.notna().mean())
        treated = treated.fillna(0.0)
        relative = pd.Series(
            (index - treatment_date).days,
            index=index,
            dtype=int,
        )
        candidate_controls = [
            str(value)
            for value in config["candidate_controls"]
            if str(value).lower() != str(metadata["community"]).lower()
            and not (
                str(value).lower() in known_treatments
                and start
                <= known_treatments[str(value).lower()]
                <= end
            )
        ]
        arctic_target = fetch_arctic_shift_series(
            str(config["arctic_shift"]["base_url"]),
            str(metadata["community"]),
            "count",
            start,
            end,
            cache_dir,
        )
        donor_series: dict[str, pd.Series] = {}
        with ThreadPoolExecutor(max_workers=int(config["arctic_shift"]["workers"])) as executor:
            futures = {
                executor.submit(
                    fetch_arctic_shift_series,
                    str(config["arctic_shift"]["base_url"]),
                    subreddit,
                    "count",
                    start,
                    end,
                    cache_dir,
                ): subreddit
                for subreddit in candidate_controls
            }
            for future in as_completed(futures):
                subreddit = futures[future]
                donor_series[subreddit] = future.result()
        donor_frame = pd.DataFrame(donor_series, index=index).sort_index(axis=1)
        nonzero = donor_frame.sum(axis=0) > 0
        donor_frame = donor_frame.loc[:, nonzero]
        source_audit = _source_overlap_audit(
            treated,
            arctic_target,
            relative,
            int(analysis["pre_days"]),
            int(analysis["validation_days"]),
        )
        issues: list[str] = []
        try:
            fitted = _select_and_fit(
                arctic_target,
                donor_frame,
                relative,
                analysis,
            )
        except (ValueError, RuntimeError) as exc:
            diagnostics.append(
                {
                    "intervention_id": intervention_id,
                    "community": str(metadata["community"]),
                    "intervention_type": str(metadata["type"]),
                    "treatment_date": treatment_date.date().isoformat(),
                    "qualified": False,
                    "issues": str(exc),
                    "treated_coverage": coverage,
                    **source_audit,
                }
            )
            continue
        if coverage < float(gates["minimum_treated_coverage"]):
            issues.append("treated_coverage")
        if source_audit["source_train_correlation"] < float(
            gates["minimum_source_correlation"]
        ):
            issues.append("source_correlation")
        if source_audit["source_validation_correlation"] < float(
            gates["minimum_source_validation_correlation"]
        ):
            issues.append("source_validation_correlation")
        if source_audit["source_log_ratio_mad"] > float(
            gates["maximum_source_log_ratio_mad"]
        ):
            issues.append("source_ratio_instability")
        if fitted["validation_rmspe_ratio"] >= float(
            gates["maximum_validation_rmspe_ratio"]
        ):
            issues.append("validation_rmspe_ratio")
        if fitted["pre_rmspe_ratio"] >= float(
            gates["maximum_pre_rmspe_ratio"]
        ):
            issues.append("pre_rmspe_ratio")
        equivalence_margin = float(gates["maximum_abs_pretrend_slope"])
        if (
            fitted["pretrend_ci_low"] <= -equivalence_margin
            or fitted["pretrend_ci_high"] >= equivalence_margin
        ):
            issues.append("pretrend_equivalence")
        if fitted["maximum_weight"] > float(gates["maximum_donor_weight"]):
            issues.append("donor_concentration")
        if fitted["effective_donors"] < float(gates["minimum_effective_donors"]):
            issues.append("effective_donors")
        qualified = not issues
        placebo = _placebo_distribution(
            fitted["gap"],
            relative,
            int(analysis["primary_post_window"]),
        )
        for donor, weight in zip(fitted["selected_names"], fitted["weights"]):
            weight_rows.append(
                {
                    "intervention_id": intervention_id,
                    "community": str(metadata["community"]),
                    "donor": donor,
                    "weight": float(weight),
                    "pre_correlation": fitted["correlations"].get(donor),
                    "qualified_intervention": qualified,
                }
            )
        for date, rel, observed, predicted, gap in zip(
            index,
            relative,
            fitted["target_transformed"],
            fitted["prediction"],
            fitted["gap"],
        ):
            event_rows.append(
                {
                    "intervention_id": intervention_id,
                    "community": str(metadata["community"]),
                    "date": date.isoformat(),
                    "relative_period": int(rel),
                    "observed_log_messages": float(observed),
                    "synthetic_log_messages": float(predicted),
                    "gap": float(gap),
                    "qualified_intervention": qualified,
                }
            )
        for window in map(int, analysis["post_windows"]):
            mask = (relative.to_numpy() >= 1) & (
                relative.to_numpy() <= window
            )
            estimate = _hac_mean_interval(
                fitted["gap"][mask],
                int(analysis["hac_max_lag"]),
            )
            placebo_p = (
                float(
                    (1 + np.sum(np.abs(placebo) >= abs(estimate[0])))
                    / (1 + len(placebo))
                )
                if len(placebo)
                else float("nan")
            )
            effect_rows.append(
                {
                    "intervention_id": intervention_id,
                    "community": str(metadata["community"]),
                "intervention_type": str(metadata["type"]),
                "treatment_date": treatment_date.date().isoformat(),
                "window_days": window,
                    "qualified": qualified,
                    "log_effect": estimate[0],
                    "ci_low": estimate[1],
                    "ci_high": estimate[2],
                    "standard_error": estimate[3],
                    "percent_effect": 100.0 * math.expm1(estimate[0]),
                    "percent_ci_low": 100.0 * math.expm1(estimate[1]),
                    "percent_ci_high": 100.0 * math.expm1(estimate[2]),
                    "placebo_p_value": placebo_p,
                }
            )
        diagnostics.append(
            {
                "intervention_id": intervention_id,
                "community": str(metadata["community"]),
                "intervention_type": str(metadata["type"]),
                "treatment_date": treatment_date.date().isoformat(),
                "qualified": qualified,
                "issues": ",".join(issues),
                "treated_coverage": coverage,
                **source_audit,
                "donor_candidates": int(donor_frame.shape[1]),
                "selected_donors": int(len(fitted["selected_names"])),
                "ridge": fitted["ridge"],
                "validation_nrmse": fitted["validation_nrmse"],
                "validation_rmspe_ratio": fitted[
                    "validation_rmspe_ratio"
                ],
                "pre_nrmse": fitted["pre_nrmse"],
                "pre_rmspe_ratio": fitted["pre_rmspe_ratio"],
                "pretrend_slope": fitted["pretrend_slope"],
                "pretrend_ci_low": fitted["pretrend_ci_low"],
                "pretrend_ci_high": fitted["pretrend_ci_high"],
                "pretrend_p_value": fitted["pretrend_p_value"],
                "maximum_donor_weight": fitted["maximum_weight"],
                "effective_donors": fitted["effective_donors"],
                "pre_placebo_windows": int(len(placebo)),
            }
        )

    diagnostics_frame = pd.DataFrame(diagnostics)
    effects_frame = pd.DataFrame(effect_rows)
    weights_frame = pd.DataFrame(weight_rows)
    events_frame = pd.DataFrame(event_rows)
    diagnostics_frame.to_csv(output / "qualification_diagnostics.csv", index=False)
    effects_frame.to_csv(output / "intervention_effects.csv", index=False)
    weights_frame.to_csv(output / "donor_weights.csv", index=False)
    events_frame.to_csv(output / "event_study.csv", index=False)

    primary = effects_frame[
        effects_frame["qualified"].astype(bool)
        & effects_frame["window_days"].eq(int(analysis["primary_post_window"]))
    ]
    meta = (
        _summarize_effect_group(primary)
        if len(primary)
        else None
    )
    effects_by_type = {
        str(intervention_type): _summarize_effect_group(group)
        for intervention_type, group in primary.groupby(
            "intervention_type",
            sort=True,
        )
    }
    minimum = int(gates["minimum_qualified_interventions"])
    claim_allowed = bool(
        meta
        and len(primary) >= minimum
        and float(meta["ci_high"]) < 0.0
    )
    result = {
        "status": "complete",
        "protocol": config.get("protocol", {}),
        "eligible_interventions": int(len(eligible_ids)),
        "qualified_interventions": int(
            diagnostics_frame["qualified"].astype(bool).sum()
        ),
        "primary_window_days": int(analysis["primary_post_window"]),
        "aggregate_effect": meta,
        "aggregate_effects_by_intervention_type": effects_by_type,
        "claim_allowed": claim_allowed,
        "claim_rule": (
            f"At least {minimum} interventions must pass pre-outcome gates and "
            "the random-effects 95% interval must be below zero."
        ),
        "selection_is_outcome_blind": True,
        "outcome_used_for_qualification": False,
        "source_boundary": (
            "TBBT supplies intervention definitions and an independent overlap "
            "audit. Arctic Shift supplies both treated-community outcomes and "
            "donor-community time series on one measurement scale; affected-user "
            "OUT slices are never used as controls."
        ),
        "inputs": {
            "panel": {
                "path": panel_path.as_posix(),
                "sha256": sha256_file(panel_path),
            },
            "catalog": {
                "path": catalog_path.as_posix(),
                "sha256": sha256_file(catalog_path),
            },
            "arctic_shift_api": str(config["arctic_shift"]["base_url"]),
        },
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output / "tbbt_qualified_control_manifest.json", result)
    return result
