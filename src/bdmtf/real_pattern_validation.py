from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from bdmtf.data.social_loader import load_comments, resolve_community_paths, strip_reddit_prefix
from bdmtf.revision.data_pipeline import _gini, _text_flags


OUTCOME_CONTROLS = {
    "log1p_late_comments": [
        "log1p_early_comments",
        "early_max_depth",
        "early_root_reply_share",
        "early_removed_rate",
    ],
    "late_mean_leaf_depth": [
        "log1p_early_comments",
        "early_max_depth",
        "early_root_reply_share",
        "early_removed_rate",
        "log1p_late_comments",
    ],
    "depth_growth": [
        "log1p_early_comments",
        "early_max_depth",
        "early_root_reply_share",
        "early_removed_rate",
        "log1p_late_comments",
    ],
    "late_root_reply_share": [
        "log1p_early_comments",
        "early_max_depth",
        "early_root_reply_share",
        "early_removed_rate",
        "log1p_late_comments",
    ],
}


def build_early_late_panel(
    social_root: str | Path,
    splits: pd.DataFrame,
    early_cutoff_minutes: float = 60.0,
    outcome_horizon_minutes: float = 720.0,
    min_early_comments: int = 3,
    min_late_comments: int = 3,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if outcome_horizon_minutes <= early_cutoff_minutes:
        raise ValueError("Outcome horizon must be later than the early cutoff")
    split_map = splits.copy()
    split_map["post_id"] = split_map["post_id"].astype(str)
    rows: list[dict[str, Any]] = []
    exclusions = {
        "missing_comments": 0,
        "insufficient_early_comments": 0,
        "insufficient_late_comments": 0,
    }
    for community, community_splits in split_map.groupby("community", sort=True):
        paths = resolve_community_paths(social_root, str(community))
        comments = load_comments(paths).copy()
        comments["post_id"] = comments["post_id"].astype(str)
        groups = {post_id: group for post_id, group in comments.groupby("post_id", sort=False)}
        for split_row in community_splits.itertuples(index=False):
            post_id = str(split_row.post_id)
            thread = groups.get(post_id)
            if thread is None or thread.empty:
                exclusions["missing_comments"] += 1
                continue
            row, reason = _thread_windows(
                thread,
                community=str(community),
                post_id=post_id,
                split=str(split_row.split),
                early_cutoff_minutes=early_cutoff_minutes,
                outcome_horizon_minutes=outcome_horizon_minutes,
                min_early_comments=min_early_comments,
                min_late_comments=min_late_comments,
            )
            if row is None:
                exclusions[reason] += 1
            else:
                rows.append(row)
    panel = pd.DataFrame(rows)
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "early_cutoff_minutes": float(early_cutoff_minutes),
        "outcome_horizon_minutes": float(outcome_horizon_minutes),
        "min_early_comments": int(min_early_comments),
        "min_late_comments": int(min_late_comments),
        "included_threads": int(len(panel)),
        "included_by_split": {
            str(key): int(value)
            for key, value in panel["split"].value_counts().sort_index().items()
        }
        if not panel.empty
        else {},
        "exclusions": exclusions,
        "temporal_contract": {
            "exposure": f"0 <= minutes_since_post <= {early_cutoff_minutes}",
            "outcomes": (
                f"{early_cutoff_minutes} < minutes_since_post <= "
                f"{outcome_horizon_minutes}"
            ),
        },
        "semantic_measure": "Local lexicon flags; no API calls.",
    }
    return panel, manifest


def analyze_heldout_patterns(
    panel: pd.DataFrame,
    exposure: str = "early_toxicity_density",
    bootstrap_samples: int = 2000,
    seed: int = 30371,
    expected_directions: Mapping[str, str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    directions = dict(
        expected_directions
        or {
            "log1p_late_comments": "positive",
            "late_mean_leaf_depth": "negative",
            "depth_growth": "negative",
            "late_root_reply_share": "positive",
        }
    )
    train = panel[panel["split"] == "train"].copy()
    test = panel[panel["split"] == "test"].copy()
    if train.empty or test.empty:
        raise ValueError("Real-pattern validation requires non-empty train and test splits")
    rows: list[dict[str, Any]] = []
    for outcome, direction in directions.items():
        controls = OUTCOME_CONTROLS[outcome]
        required = {outcome, exposure, "community", *controls}
        missing = required - set(panel.columns)
        if missing:
            raise ValueError(f"Missing analysis columns for {outcome}: {sorted(missing)}")

        baseline = _fit_linear(train, controls, outcome)
        full = _fit_linear(train, [exposure, *controls], outcome)
        baseline_prediction = _predict_linear(test, baseline)
        full_prediction = _predict_linear(test, full)
        observed = test[outcome].astype(float).to_numpy()
        baseline_rmse = _rmse(observed, baseline_prediction)
        full_rmse = _rmse(observed, full_prediction)

        exposure_model = _fit_linear(train, controls, exposure)
        exposure_residual = (
            test[exposure].astype(float).to_numpy()
            - _predict_linear(test, exposure_model)
        )
        outcome_residual = observed - baseline_prediction
        heldout_slope = _slope(exposure_residual, outcome_residual)
        ci_low, ci_high = _bootstrap_slope(
            exposure_residual,
            outcome_residual,
            bootstrap_samples,
            _stable_seed(seed, outcome),
        )
        coefficient = float(full["coefficients"][full["feature_names"].index(exposure)])
        expected_sign = 1.0 if direction == "positive" else -1.0
        supports = bool(
            heldout_slope * expected_sign > 0
            and ci_low * expected_sign > 0
            and ci_high * expected_sign > 0
        )
        rows.append(
            {
                "outcome": outcome,
                "expected_direction": direction,
                "train_exposure_coefficient_per_sd": coefficient,
                "heldout_partial_slope": heldout_slope,
                "heldout_ci_low": ci_low,
                "heldout_ci_high": ci_high,
                "baseline_test_rmse": baseline_rmse,
                "full_test_rmse": full_rmse,
                "relative_rmse_improvement": (
                    (baseline_rmse - full_rmse) / baseline_rmse
                    if baseline_rmse > 0
                    else 0.0
                ),
                "supports_expected_direction": supports,
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "controls": json.dumps(controls),
            }
        )
    results = pd.DataFrame(rows)
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "fit_split": "train",
        "evaluation_split": "test",
        "n_train": int(len(train)),
        "n_validation": int((panel["split"] == "validation").sum()),
        "n_test": int(len(test)),
        "exposure": exposure,
        "exposure_train_mean": float(train[exposure].astype(float).mean()),
        "exposure_train_std": float(train[exposure].astype(float).std(ddof=0)),
        "bootstrap_samples": int(bootstrap_samples),
        "supported_outcomes": int(results["supports_expected_direction"].sum()),
        "total_outcomes": int(len(results)),
        "causal_claim_allowed": False,
        "claim": "Held-out observational pattern consistency only.",
    }
    return results, manifest


def run_real_pattern_validation(
    root: str | Path,
    config: Mapping[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    root_path = Path(root)
    data = config["data"]
    window = config["window"]
    analysis = config["analysis"]
    splits = pd.read_csv(root_path / data["splits"])
    panel, panel_manifest = build_early_late_panel(
        root_path / data["social_root"],
        splits,
        early_cutoff_minutes=float(window["early_cutoff_minutes"]),
        outcome_horizon_minutes=float(window["outcome_horizon_minutes"]),
        min_early_comments=int(window["min_early_comments"]),
        min_late_comments=int(window["min_late_comments"]),
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    panel.to_csv(output / "early_late_panel.csv", index=False)
    panel.to_parquet(output / "early_late_panel.parquet", index=False)
    results, analysis_manifest = analyze_heldout_patterns(
        panel,
        exposure=str(analysis["exposure"]),
        bootstrap_samples=int(analysis["bootstrap_samples"]),
        seed=int(analysis["seed"]),
        expected_directions=analysis["primary_outcomes"],
    )
    results.to_csv(output / "heldout_pattern_results.csv", index=False)
    manifest = {
        "status": "complete",
        "panel": panel_manifest,
        "analysis": analysis_manifest,
        "evidence_scope": config["evidence_scope"],
        "inputs": {
            "splits": str(data["splits"]),
            "splits_sha256": _sha256(root_path / data["splits"]),
        },
    }
    (output / "real_pattern_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def _thread_windows(
    comments: pd.DataFrame,
    community: str,
    post_id: str,
    split: str,
    early_cutoff_minutes: float,
    outcome_horizon_minutes: float,
    min_early_comments: int,
    min_late_comments: int,
) -> tuple[dict[str, Any] | None, str]:
    frame = comments.copy()
    frame["minutes"] = pd.to_numeric(frame["minutes_since_post"], errors="coerce")
    frame = frame[
        frame["minutes"].notna()
        & (frame["minutes"] >= 0)
        & (frame["minutes"] <= outcome_horizon_minutes)
    ].copy()
    frame["comment_id_clean"] = frame["comment_id"].map(strip_reddit_prefix).astype(str)
    frame["parent_clean"] = frame["parent_id"].map(strip_reddit_prefix).astype(str)
    frame["depth_one_based"] = (
        pd.to_numeric(frame.get("depth", 0), errors="coerce")
        .fillna(0)
        .clip(lower=0)
        .astype(int)
        + 1
    )
    early = frame[frame["minutes"] <= early_cutoff_minutes].copy()
    late = frame[frame["minutes"] > early_cutoff_minutes].copy()
    if len(early) < min_early_comments:
        return None, "insufficient_early_comments"
    if len(late) < min_late_comments:
        return None, "insufficient_late_comments"

    early_text = early.get("comment_text", pd.Series("", index=early.index))
    early_toxic, early_counter, early_removed = _text_flags(early_text)
    authors = early.get("author", pd.Series("", index=early.index)).fillna("").astype(str)
    semantic = ~authors.str.casefold().isin(
        {"automoderator", "[deleted]", "", "none", "nan"}
    ) & ~early_removed
    semantic_count = int(semantic.sum())
    toxicity_density = float(early_toxic[semantic].mean()) if semantic_count else 0.0
    counterspeech_density = (
        float(early_counter[semantic].mean()) if semantic_count else 0.0
    )

    all_ids = set(frame["comment_id_clean"])
    early_ids = set(early["comment_id_clean"])
    late_root = ~late["parent_clean"].isin(all_ids)
    child_counts = frame["parent_clean"].value_counts()
    late_leaves = late[~late["comment_id_clean"].isin(child_counts.index)]
    late_width = late.groupby("depth_one_based").size().to_numpy(dtype=float)
    valid_authors = authors[~authors.str.casefold().isin({"[deleted]", "", "none", "nan"})]
    early_root = ~early["parent_clean"].isin(early_ids)

    return (
        {
            "community": community,
            "post_id": post_id,
            "split": split,
            "early_comment_count": int(len(early)),
            "log1p_early_comments": float(np.log1p(len(early))),
            "early_semantic_comment_count": semantic_count,
            "early_toxicity_density": toxicity_density,
            "early_counterspeech_density": counterspeech_density,
            "early_removed_rate": float(early_removed.mean()),
            "early_max_depth": float(early["depth_one_based"].max()),
            "early_mean_depth": float(early["depth_one_based"].mean()),
            "early_root_reply_share": float(early_root.mean()),
            "early_unique_authors": int(valid_authors.nunique()),
            "late_comment_count": int(len(late)),
            "log1p_late_comments": float(np.log1p(len(late))),
            "late_mean_leaf_depth": (
                float(late_leaves["depth_one_based"].mean())
                if not late_leaves.empty
                else 0.0
            ),
            "late_mean_depth": float(late["depth_one_based"].mean()),
            "late_root_reply_share": float(late_root.mean()),
            "late_width_gini": _gini(late_width),
            "depth_growth": float(
                frame["depth_one_based"].max() - early["depth_one_based"].max()
            ),
            "total_12h_comments": int(len(frame)),
        },
        "",
    )


def _fit_linear(frame: pd.DataFrame, features: Sequence[str], target: str) -> dict[str, Any]:
    matrix, feature_names, scales = _encode(frame, features, fit=True)
    values = frame[target].astype(float).to_numpy()
    coefficients, _, _, _ = np.linalg.lstsq(matrix, values, rcond=None)
    return {
        "features": list(features),
        "feature_names": feature_names,
        "scales": scales,
        "coefficients": coefficients,
    }


def _predict_linear(frame: pd.DataFrame, model: Mapping[str, Any]) -> np.ndarray:
    matrix, _, _ = _encode(
        frame,
        model["features"],
        fit=False,
        scales=model["scales"],
        feature_names=model["feature_names"],
    )
    return matrix @ np.asarray(model["coefficients"], dtype=float)


def _encode(
    frame: pd.DataFrame,
    features: Sequence[str],
    fit: bool,
    scales: Mapping[str, Any] | None = None,
    feature_names: Sequence[str] | None = None,
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    numeric_features = [name for name in features if name != "community"]
    if fit:
        communities = sorted(frame["community"].astype(str).unique())
        reference = communities[0]
        numeric_scales = {}
        for name in numeric_features:
            values = frame[name].astype(float)
            mean = float(values.mean())
            std = float(values.std(ddof=0))
            numeric_scales[name] = (mean, max(std, 1e-9))
        scales = {
            "numeric": numeric_scales,
            "communities": communities,
            "reference_community": reference,
        }
        feature_names = [
            "intercept",
            *numeric_features,
            *(f"community={name}" for name in communities if name != reference),
        ]
    if scales is None or feature_names is None:
        raise ValueError("Encoding scales and feature names are required")
    columns = [np.ones(len(frame), dtype=float)]
    for name in numeric_features:
        mean, std = scales["numeric"][name]
        columns.append((frame[name].astype(float).to_numpy() - mean) / std)
    community_values = frame["community"].astype(str)
    for community in scales["communities"]:
        if community == scales["reference_community"]:
            continue
        columns.append((community_values == community).to_numpy(dtype=float))
    return np.column_stack(columns), list(feature_names), dict(scales)


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    denominator = float(np.dot(x, x))
    return float(np.dot(x, y) / denominator) if denominator > 1e-12 else 0.0


def _bootstrap_slope(
    x: np.ndarray,
    y: np.ndarray,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    if len(x) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=float)
    for index in range(samples):
        sampled = rng.integers(0, len(x), size=len(x))
        estimates[index] = _slope(x[sampled], y[sampled])
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def _rmse(observed: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((observed - predicted) ** 2)))


def _stable_seed(seed: int, value: str) -> int:
    digest = hashlib.sha256(f"{seed}::{value}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
