from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from bdmtf.revision.provenance import sha256_file, write_json


ADAPTED = "platform_adapted_learned_bdmtf"
TARGET_DEFAULT = "target_default_learned_bdmtf"
ZERO_SHOT = "reddit_zero_shot_learned_bdmtf"
CLASSICAL = (
    "target_empirical_bootstrap",
    "target_branching_process",
    "target_hawkes",
)


def evaluate_adapter_gain(
    fidelity: pd.DataFrame,
    gate: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {"community", "metric", "model", "normalized_wasserstein"}
    missing = required - set(fidelity.columns)
    if missing:
        raise ValueError(f"Missing fidelity columns: {sorted(missing)}")
    pivot = fidelity.pivot_table(
        index=["community", "metric"],
        columns="model",
        values="normalized_wasserstein",
        aggfunc="mean",
    )
    required_models = {ADAPTED, TARGET_DEFAULT, ZERO_SHOT}
    missing_models = required_models - set(pivot.columns)
    if missing_models:
        raise ValueError(f"Missing adapter comparison models: {sorted(missing_models)}")
    complete = pivot.dropna(subset=sorted(required_models)).copy()
    if complete.empty:
        raise ValueError("No complete adapter comparison metric units")
    complete["gain_over_target_default"] = (
        complete[TARGET_DEFAULT] - complete[ADAPTED]
    )
    complete["gain_over_reddit_zero_shot"] = (
        complete[ZERO_SHOT] - complete[ADAPTED]
    )
    available_classical = [model for model in CLASSICAL if model in complete]
    if available_classical:
        complete["best_classical_distance"] = complete[available_classical].min(
            axis=1
        )
        complete["gain_over_best_classical"] = (
            complete["best_classical_distance"] - complete[ADAPTED]
        )
    else:
        complete["best_classical_distance"] = float("nan")
        complete["gain_over_best_classical"] = float("nan")
    units = complete.reset_index()

    default_share = float((units["gain_over_target_default"] > 0).mean())
    zero_share = float((units["gain_over_reddit_zero_shot"] > 0).mean())
    default_median = float(units["gain_over_target_default"].median())
    zero_median = float(units["gain_over_reddit_zero_shot"].median())
    checks = {
        "target_default_metric_share": bool(
            default_share
            >= float(gate["adapted_beats_target_default_metric_share_min"])
        ),
        "reddit_zero_shot_metric_share": bool(
            zero_share
            >= float(gate["adapted_beats_reddit_zero_shot_metric_share_min"])
        ),
        "target_default_median_gain": bool(
            default_median > 0
            or not bool(gate["require_positive_median_gain_over_target_default"])
        ),
        "reddit_zero_shot_median_gain": bool(
            zero_median > 0
            or not bool(gate["require_positive_median_gain_over_reddit_zero_shot"])
        ),
    }
    if bool(gate.get("require_beating_every_classical_baseline", False)):
        checks["every_classical_baseline"] = bool(
            (units["gain_over_best_classical"] > 0).all()
        )
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "supports_platform_adaptation": bool(all(checks.values())),
        "preregistered_gate": dict(gate),
        "gate_checks": checks,
        "n_metric_units": int(len(units)),
        "adapted_beats_target_default_metric_share": default_share,
        "adapted_beats_reddit_zero_shot_metric_share": zero_share,
        "median_gain_over_target_default": default_median,
        "median_gain_over_reddit_zero_shot": zero_median,
        "adapted_beats_best_classical_metric_share": (
            float((units["gain_over_best_classical"] > 0).mean())
            if available_classical
            else None
        ),
        "claim_boundary": (
            "Passing this gate supports bounded target-platform adaptation, "
            "not zero-shot universality or causal intervention validity."
        ),
    }
    return units, manifest


def write_adapter_gain(
    fidelity_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    fidelity_file = Path(fidelity_path)
    config_file = Path(config_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fidelity = pd.read_csv(fidelity_file)
    config = json.loads(config_file.read_text(encoding="utf-8"))
    units, manifest = evaluate_adapter_gain(fidelity, config["success_gate"])
    units_path = output / "adapter_gain_by_metric.csv"
    units.to_csv(units_path, index=False)
    manifest["inputs"] = {
        "fidelity": str(fidelity_file),
        "fidelity_sha256": sha256_file(fidelity_file),
        "config": str(config_file),
        "config_sha256": sha256_file(config_file),
    }
    manifest["adapter_gain_by_metric_sha256"] = sha256_file(units_path)
    write_json(output / "adapter_gain_manifest.json", manifest)
    return manifest
