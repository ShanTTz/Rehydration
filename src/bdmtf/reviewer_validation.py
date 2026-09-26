from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


DEFAULT_FACTORIAL_CELLS = {
    "baseline_best": "BASELINE",
    "baseline_controversial": "BASELINE_CONTROVERSIAL",
    "toxic_best": "CORE_TOXIC_BEST",
    "toxic_controversial": "CORE_TOXIC_CONTROVERSIAL",
}

FACTORIAL_METRICS = (
    "comment_volume",
    "mean_leaf_depth",
    "max_depth",
    "engagement_volume",
)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    return records


def factorial_contrasts(
    records: Sequence[Mapping[str, Any]],
    cells: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    cell_names = dict(cells or DEFAULT_FACTORIAL_CELLS)
    required_roles = set(DEFAULT_FACTORIAL_CELLS)
    missing_roles = required_roles - set(cell_names)
    if missing_roles:
        raise ValueError(f"Missing factorial cell role(s): {sorted(missing_roles)}")

    frame = pd.DataFrame(records)
    key_columns = ["community", "post_id", "seed"]
    required_columns = set(key_columns) | {"condition"} | set(FACTORIAL_METRICS)
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Missing run column(s): {sorted(missing_columns)}")

    selected_conditions = set(cell_names.values())
    selected = frame.loc[frame["condition"].isin(selected_conditions)].copy()
    duplicate_mask = selected.duplicated(key_columns + ["condition"], keep=False)
    if duplicate_mask.any():
        duplicate_keys = selected.loc[duplicate_mask, key_columns + ["condition"]]
        raise ValueError(
            "Duplicate factorial cells found: "
            + duplicate_keys.head(5).to_dict(orient="records").__repr__()
        )

    rows: list[dict[str, Any]] = []
    for key, group in selected.groupby(key_columns, sort=True):
        by_condition = group.set_index("condition")
        absent = selected_conditions - set(by_condition.index)
        if absent:
            continue
        row: dict[str, Any] = dict(zip(key_columns, key))
        for metric in FACTORIAL_METRICS:
            bb = float(by_condition.loc[cell_names["baseline_best"], metric])
            bc = float(by_condition.loc[cell_names["baseline_controversial"], metric])
            tb = float(by_condition.loc[cell_names["toxic_best"], metric])
            tc = float(by_condition.loc[cell_names["toxic_controversial"], metric])
            values = {
                "joint": tc - bb,
                "core_at_best": tb - bb,
                "ranking_at_baseline": bc - bb,
                "ranking_at_toxic": tc - tb,
                "core_ranking_interaction": tc - tb - bc + bb,
            }
            if metric == "comment_volume":
                log_values = {
                    "joint_log": np.log1p(tc) - np.log1p(bb),
                    "core_at_best_log": np.log1p(tb) - np.log1p(bb),
                    "ranking_at_baseline_log": np.log1p(bc) - np.log1p(bb),
                    "ranking_at_toxic_log": np.log1p(tc) - np.log1p(tb),
                    "core_ranking_interaction_log": (
                        np.log1p(tc)
                        - np.log1p(tb)
                        - np.log1p(bc)
                        + np.log1p(bb)
                    ),
                }
                values.update(log_values)
            for contrast, value in values.items():
                row[f"{metric}__{contrast}"] = float(value)
            row[f"{metric}__baseline_best"] = bb
            row[f"{metric}__baseline_controversial"] = bc
            row[f"{metric}__toxic_best"] = tb
            row[f"{metric}__toxic_controversial"] = tc
        row["shallow_swarm_joint"] = bool(
            row["comment_volume__joint"] > 0 and row["mean_leaf_depth__joint"] < 0
        )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_factorial(
    contrasts: pd.DataFrame,
    bootstrap_samples: int = 2000,
    seed: int = 30371,
) -> pd.DataFrame:
    if contrasts.empty:
        raise ValueError("No complete factorial blocks were found")
    value_columns = [
        column
        for column in contrasts.columns
        if "__" in column
        and not column.endswith(
            (
                "__baseline_best",
                "__baseline_controversial",
                "__toxic_best",
                "__toxic_controversial",
            )
        )
    ]
    summaries: list[dict[str, Any]] = []
    for scope, frame in _scopes(contrasts):
        for column in value_columns:
            values = frame[column].astype(float).to_numpy()
            low, high = _cluster_bootstrap_ci(
                frame,
                column,
                samples=bootstrap_samples,
                seed=_derived_seed(seed, scope, column),
            )
            metric, contrast = column.split("__", maxsplit=1)
            summaries.append(
                {
                    "scope": scope,
                    "metric": metric,
                    "contrast": contrast,
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "ci_low": low,
                    "ci_high": high,
                    "n_post_seed_blocks": int(len(frame)),
                    "n_posts": int(
                        frame[["community", "post_id"]].drop_duplicates().shape[0]
                    ),
                }
            )
    return pd.DataFrame(summaries)


def write_factorial_outputs(
    records_path: str | Path,
    output_dir: str | Path,
    cells: Mapping[str, str] | None = None,
    bootstrap_samples: int = 2000,
    seed: int = 30371,
) -> dict[str, Any]:
    records_file = Path(records_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    contrasts = factorial_contrasts(load_jsonl(records_file), cells=cells)
    summary = summarize_factorial(
        contrasts,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    contrasts.to_csv(output / "factorial_contrasts.csv", index=False)
    summary.to_csv(output / "factorial_summary.csv", index=False)

    joint = summary.loc[
        (summary["scope"] == "all")
        & (
            (
                (summary["metric"] == "comment_volume")
                & (summary["contrast"] == "joint_log")
            )
            | (
                (summary["metric"] == "mean_leaf_depth")
                & (summary["contrast"] == "joint")
            )
        )
    ]
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "records_path": str(records_file),
        "records_sha256": _sha256(records_file),
        "factorial_cells": dict(cells or DEFAULT_FACTORIAL_CELLS),
        "complete_post_seed_blocks": int(len(contrasts)),
        "complete_posts": int(
            contrasts[["community", "post_id"]].drop_duplicates().shape[0]
        ),
        "bootstrap_samples": int(bootstrap_samples),
        "bootstrap_seed": int(seed),
        "primary_joint_results": joint.to_dict(orient="records"),
    }
    (output / "factorial_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def _scopes(frame: pd.DataFrame):
    yield "all", frame
    for community, group in frame.groupby("community", sort=True):
        yield str(community), group


def _cluster_bootstrap_ci(
    frame: pd.DataFrame,
    column: str,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    if samples <= 0:
        mean = float(frame[column].astype(float).mean())
        return mean, mean
    grouped = [
        group[column].astype(float).to_numpy()
        for _, group in frame.groupby(["community", "post_id"], sort=True)
    ]
    if not grouped:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boot = np.empty(samples, dtype=float)
    for index in range(samples):
        sampled = rng.integers(0, len(grouped), size=len(grouped))
        boot[index] = float(
            np.mean(np.concatenate([grouped[group_index] for group_index in sampled]))
        )
    low, high = np.quantile(boot, [0.025, 0.975])
    return float(low), float(high)


def _derived_seed(seed: int, *parts: object) -> int:
    raw = "::".join([str(seed), *(str(part) for part in parts)])
    return int(hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8], 16)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
