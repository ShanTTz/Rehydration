from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


BENCHMARK_METRICS = {
    "comment_volume": "log1p",
    "mean_leaf_depth": "identity",
}


def benchmark_factorial_response(
    contrasts: pd.DataFrame,
    splits: pd.DataFrame,
    bootstrap_samples: int = 2000,
    seed: int = 30371,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    split_frame = splits[["community", "post_id", "split"]].copy()
    split_frame["post_id"] = split_frame["post_id"].astype(str)
    frame = contrasts.copy()
    frame["post_id"] = frame["post_id"].astype(str)
    frame = frame.merge(
        split_frame,
        on=["community", "post_id"],
        how="inner",
        validate="many_to_one",
    )
    train = frame[frame["split"] == "train"].copy()
    test = frame[frame["split"] == "test"].copy()
    if train.empty or test.empty:
        raise ValueError("Response benchmark requires non-empty train and test blocks")

    prediction_parts: list[pd.DataFrame] = []
    coefficient_rows: list[dict[str, Any]] = []
    for metric, transform in BENCHMARK_METRICS.items():
        train_effects = _effect_frame(train, metric, transform)
        test_effects = _effect_frame(test, metric, transform)
        global_effects = _mean_effects(train_effects)
        community_effects = {
            str(community): _mean_effects(group)
            for community, group in train_effects.groupby("community", sort=True)
        }
        coefficient_rows.append(
            {
                "metric": metric,
                "scope": "all",
                "transform": transform,
                **global_effects,
                "n_train_blocks": int(len(train_effects)),
            }
        )
        for community, effects in community_effects.items():
            coefficient_rows.append(
                {
                    "metric": metric,
                    "scope": community,
                    "transform": transform,
                    **effects,
                    "n_train_blocks": int(
                        (train_effects["community"] == community).sum()
                    ),
                }
            )
        prediction_parts.append(
            _predict_test_effects(
                test_effects,
                metric,
                transform,
                global_effects,
                community_effects,
            )
        )

    predictions = pd.concat(prediction_parts, ignore_index=True)
    summary = _summarize_predictions(predictions)
    improvements = _paired_improvements(
        predictions,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    coefficients = pd.DataFrame(coefficient_rows)
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "fit_split": "train",
        "evaluation_split": "test",
        "n_train_blocks": int(len(train)),
        "n_test_blocks": int(len(test)),
        "n_train_posts": int(
            train[["community", "post_id"]].drop_duplicates().shape[0]
        ),
        "n_test_posts": int(
            test[["community", "post_id"]].drop_duplicates().shape[0]
        ),
        "models": sorted(predictions["model"].unique()),
        "metrics": BENCHMARK_METRICS,
        "baseline_observed_on_test": (
            "Each prediction conditions on that test block's baseline-best outcome; "
            "the benchmark evaluates intervention-response transfer, not absolute "
            "cascade forecasting."
        ),
        "evidence_scope": (
            "Internal held-out mechanism benchmark over simulator-generated factorial "
            "outcomes; it is not real-world intervention validation."
        ),
    }
    return summary, improvements, predictions, manifest


def write_response_benchmark(
    contrasts_path: str | Path,
    splits_path: str | Path,
    output_dir: str | Path,
    bootstrap_samples: int = 2000,
    seed: int = 30371,
) -> dict[str, Any]:
    contrasts_file = Path(contrasts_path)
    splits_file = Path(splits_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary, improvements, predictions, manifest = benchmark_factorial_response(
        pd.read_csv(contrasts_file),
        pd.read_csv(splits_file),
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    summary_path = output / "response_benchmark_summary.csv"
    improvement_path = output / "response_benchmark_improvements.csv"
    prediction_path = output / "response_benchmark_predictions.parquet"
    summary.to_csv(summary_path, index=False)
    improvements.to_csv(improvement_path, index=False)
    predictions.to_parquet(prediction_path, index=False)
    manifest.update(
        {
            "bootstrap_samples": int(bootstrap_samples),
            "bootstrap_seed": int(seed),
            "inputs": {
                "contrasts": str(contrasts_file),
                "contrasts_sha256": _sha256(contrasts_file),
                "splits": str(splits_file),
                "splits_sha256": _sha256(splits_file),
            },
            "outputs": {
                "summary": str(summary_path),
                "summary_sha256": _sha256(summary_path),
                "improvements": str(improvement_path),
                "improvements_sha256": _sha256(improvement_path),
                "predictions": str(prediction_path),
                "predictions_sha256": _sha256(prediction_path),
            },
        }
    )
    (output / "response_benchmark_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def _effect_frame(
    frame: pd.DataFrame,
    metric: str,
    transform: str,
) -> pd.DataFrame:
    required = [
        f"{metric}__baseline_best",
        f"{metric}__baseline_controversial",
        f"{metric}__toxic_best",
        f"{metric}__toxic_controversial",
    ]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing factorial metric columns: {sorted(missing)}")
    values = frame[["community", "post_id", "seed", *required]].copy()
    cell_values = {
        role: _transform(
            values[f"{metric}__{role}"].to_numpy(dtype=float),
            transform,
        )
        for role in (
            "baseline_best",
            "baseline_controversial",
            "toxic_best",
            "toxic_controversial",
        )
    }
    result = values[["community", "post_id", "seed"]].copy()
    result["baseline"] = cell_values["baseline_best"]
    result["observed_joint"] = cell_values["toxic_controversial"]
    result["core_effect"] = (
        cell_values["toxic_best"] - cell_values["baseline_best"]
    )
    result["ranking_effect"] = (
        cell_values["baseline_controversial"] - cell_values["baseline_best"]
    )
    result["interaction_effect"] = (
        cell_values["toxic_controversial"]
        - cell_values["toxic_best"]
        - cell_values["baseline_controversial"]
        + cell_values["baseline_best"]
    )
    result["observed_change"] = result["observed_joint"] - result["baseline"]
    return result


def _mean_effects(frame: pd.DataFrame) -> dict[str, float]:
    return {
        "core_effect": float(frame["core_effect"].mean()),
        "ranking_effect": float(frame["ranking_effect"].mean()),
        "interaction_effect": float(frame["interaction_effect"].mean()),
    }


def _predict_test_effects(
    test: pd.DataFrame,
    metric: str,
    transform: str,
    global_effects: dict[str, float],
    community_effects: dict[str, dict[str, float]],
) -> pd.DataFrame:
    models = {
        "no_effect": lambda _: 0.0,
        "core_only_global": lambda _: global_effects["core_effect"],
        "ranking_only_global": lambda _: global_effects["ranking_effect"],
        "additive_global": lambda _: (
            global_effects["core_effect"] + global_effects["ranking_effect"]
        ),
        "interaction_global": lambda _: (
            global_effects["core_effect"]
            + global_effects["ranking_effect"]
            + global_effects["interaction_effect"]
        ),
        "additive_community": lambda community: (
            community_effects.get(community, global_effects)["core_effect"]
            + community_effects.get(community, global_effects)["ranking_effect"]
        ),
        "interaction_community": lambda community: (
            community_effects.get(community, global_effects)["core_effect"]
            + community_effects.get(community, global_effects)["ranking_effect"]
            + community_effects.get(community, global_effects)["interaction_effect"]
        ),
    }
    rows: list[dict[str, Any]] = []
    for row in test.itertuples(index=False):
        for model_name, effect_function in models.items():
            predicted_change = float(effect_function(str(row.community)))
            predicted_joint = float(row.baseline + predicted_change)
            rows.append(
                {
                    "community": str(row.community),
                    "post_id": str(row.post_id),
                    "seed": int(row.seed),
                    "metric": metric,
                    "transform": transform,
                    "model": model_name,
                    "baseline": float(row.baseline),
                    "observed_joint": float(row.observed_joint),
                    "observed_change": float(row.observed_change),
                    "predicted_joint": predicted_joint,
                    "predicted_change": predicted_change,
                    "error": predicted_joint - float(row.observed_joint),
                    "absolute_error": abs(
                        predicted_joint - float(row.observed_joint)
                    ),
                    "squared_error": (
                        predicted_joint - float(row.observed_joint)
                    )
                    ** 2,
                    "direction_correct": bool(
                        np.sign(predicted_change) == np.sign(row.observed_change)
                    ),
                }
            )
    return pd.DataFrame(rows)


def _summarize_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (metric, model), group in predictions.groupby(
        ["metric", "model"], sort=True
    ):
        rows.append(
            {
                "metric": metric,
                "transform": str(group["transform"].iloc[0]),
                "model": model,
                "mae": float(group["absolute_error"].mean()),
                "rmse": float(np.sqrt(group["squared_error"].mean())),
                "mean_error": float(group["error"].mean()),
                "direction_accuracy": float(group["direction_correct"].mean()),
                "n_test_blocks": int(len(group)),
                "n_test_posts": int(
                    group[["community", "post_id"]].drop_duplicates().shape[0]
                ),
            }
        )
    return pd.DataFrame(rows)


def _paired_improvements(
    predictions: pd.DataFrame,
    bootstrap_samples: int,
    seed: int,
) -> pd.DataFrame:
    comparisons = (
        ("interaction_global", "additive_global"),
        ("interaction_community", "additive_community"),
    )
    rows = []
    for metric, metric_frame in predictions.groupby("metric", sort=True):
        key = ["community", "post_id", "seed"]
        for full_name, reduced_name in comparisons:
            full = metric_frame[metric_frame["model"] == full_name][
                [*key, "absolute_error"]
            ].rename(columns={"absolute_error": "full_error"})
            reduced = metric_frame[metric_frame["model"] == reduced_name][
                [*key, "absolute_error"]
            ].rename(columns={"absolute_error": "reduced_error"})
            paired = full.merge(reduced, on=key, validate="one_to_one")
            paired["improvement"] = (
                paired["reduced_error"] - paired["full_error"]
            )
            low, high = _cluster_bootstrap_mean(
                paired,
                "improvement",
                bootstrap_samples,
                _stable_seed(seed, metric, full_name, reduced_name),
            )
            rows.append(
                {
                    "metric": metric,
                    "full_model": full_name,
                    "reduced_model": reduced_name,
                    "mean_absolute_error_improvement": float(
                        paired["improvement"].mean()
                    ),
                    "ci_low": low,
                    "ci_high": high,
                    "full_model_better_share": float(
                        (paired["improvement"] > 0).mean()
                    ),
                    "n_test_blocks": int(len(paired)),
                }
            )
    return pd.DataFrame(rows)


def _cluster_bootstrap_mean(
    frame: pd.DataFrame,
    column: str,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    grouped = [
        group[column].to_numpy(dtype=float)
        for _, group in frame.groupby(["community", "post_id"], sort=True)
    ]
    if not grouped:
        return float("nan"), float("nan")
    if samples <= 0:
        mean = float(frame[column].mean())
        return mean, mean
    rng = np.random.default_rng(seed)
    values = np.empty(samples, dtype=float)
    for index in range(samples):
        selected = rng.integers(0, len(grouped), size=len(grouped))
        values[index] = float(
            np.mean(np.concatenate([grouped[item] for item in selected]))
        )
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def _transform(values: np.ndarray, transform: str) -> np.ndarray:
    if transform == "log1p":
        return np.log1p(np.maximum(values, 0.0))
    if transform == "identity":
        return values
    raise ValueError(f"Unknown transform: {transform}")


def _stable_seed(seed: int, *parts: object) -> int:
    digest = hashlib.sha256(
        "|".join([str(seed), *map(str, parts)]).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
