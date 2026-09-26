from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


def latin_hypercube_draws(
    parameters: Sequence[Mapping[str, Any]],
    count: int,
    seed: int,
) -> list[dict[str, float | int]]:
    if count <= 0:
        raise ValueError("Latin hypercube draw count must be positive")
    rng = np.random.default_rng(seed)
    draws = [dict() for _ in range(count)]
    for parameter in parameters:
        name = str(parameter["name"])
        low = float(parameter["low"])
        high = float(parameter["high"])
        if high < low:
            raise ValueError(f"Invalid range for {name}: {low}..{high}")
        strata = (np.arange(count, dtype=float) + rng.random(count)) / count
        rng.shuffle(strata)
        values = low + strata * (high - low)
        if bool(parameter.get("integer", False)):
            values = np.rint(values).astype(int)
        for draw, value in zip(draws, values):
            draw[name] = int(value) if bool(parameter.get("integer", False)) else float(value)
    return draws


def apply_parameter_draw(
    source: Mapping[str, Any],
    parameters: Sequence[Mapping[str, Any]],
    draw: Mapping[str, float | int],
) -> dict[str, Any]:
    raw = copy.deepcopy(dict(source))
    simulation = raw.setdefault("simulation", {})
    calibrations = raw.setdefault("community_calibration", {})
    for parameter in parameters:
        name = str(parameter["name"])
        target = str(parameter["target"])
        mode = str(parameter.get("mode", "value"))
        draw_value = draw[name]
        if target not in simulation:
            raise KeyError(f"Source simulation config has no {target!r}")
        simulation[target] = _transform(simulation[target], draw_value, mode)
        if bool(parameter.get("integer", False)):
            simulation[target] = int(round(float(simulation[target])))
        community_field = parameter.get("community_field")
        if community_field:
            for calibration in calibrations.values():
                if community_field in calibration:
                    calibration[community_field] = _transform(
                        calibration[community_field],
                        draw_value,
                        mode,
                    )
    return raw


def summarize_sensitivity_records(
    records: Sequence[Mapping[str, Any]],
    baseline_condition: str = "BASELINE",
    treated_condition: str = "CORE_TOXIC_CONTROVERSIAL",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    frame = pd.DataFrame(records)
    keys = ["scenario", "community", "post_id", "seed"]
    required = set(keys) | {
        "condition",
        "comment_volume",
        "mean_leaf_depth",
        "max_depth",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing sensitivity column(s): {sorted(missing)}")
    selected = frame[frame["condition"].isin([baseline_condition, treated_condition])].copy()
    duplicate = selected.duplicated(keys + ["condition"], keep=False)
    if duplicate.any():
        raise ValueError("Sensitivity output contains duplicate condition cells")

    block_rows: list[dict[str, Any]] = []
    parameter_columns = [name for name in frame.columns if name.startswith("parameter__")]
    for key, group in selected.groupby(keys, sort=True):
        by_condition = group.set_index("condition")
        if baseline_condition not in by_condition.index or treated_condition not in by_condition.index:
            continue
        baseline = by_condition.loc[baseline_condition]
        treated = by_condition.loc[treated_condition]
        block = dict(zip(keys, key))
        for column in parameter_columns:
            block[column] = baseline[column]
        block["log_volume_effect"] = float(
            np.log1p(float(treated["comment_volume"]))
            - np.log1p(float(baseline["comment_volume"]))
        )
        block["volume_ratio"] = float(
            float(treated["comment_volume"]) / max(1.0, float(baseline["comment_volume"]))
        )
        block["leaf_depth_delta"] = float(
            treated["mean_leaf_depth"] - baseline["mean_leaf_depth"]
        )
        block["max_depth_delta"] = float(treated["max_depth"] - baseline["max_depth"])
        block["shallow_swarm"] = bool(
            block["log_volume_effect"] > 0 and block["leaf_depth_delta"] < 0
        )
        block_rows.append(block)
    blocks = pd.DataFrame(block_rows)
    if blocks.empty:
        raise ValueError("No complete sensitivity blocks were found")

    scenario_rows: list[dict[str, Any]] = []
    for scenario, group in blocks.groupby("scenario", sort=True):
        item: dict[str, Any] = {
            "scenario": scenario,
            "mean_log_volume_effect": float(group["log_volume_effect"].mean()),
            "geometric_volume_ratio": float(np.exp(group["log_volume_effect"].mean())),
            "median_volume_ratio": float(group["volume_ratio"].median()),
            "mean_leaf_depth_delta": float(group["leaf_depth_delta"].mean()),
            "mean_max_depth_delta": float(group["max_depth_delta"].mean()),
            "shallow_swarm_block_share": float(group["shallow_swarm"].mean()),
            "n_blocks": int(len(group)),
            "n_posts": int(group[["community", "post_id"]].drop_duplicates().shape[0]),
        }
        for column in parameter_columns:
            item[column] = group.iloc[0][column]
        item["scenario_supports_shallow_swarm"] = bool(
            item["mean_log_volume_effect"] > 0
            and item["mean_leaf_depth_delta"] < 0
        )
        scenario_rows.append(item)
    scenarios = pd.DataFrame(scenario_rows)

    supported = scenarios["scenario_supports_shallow_swarm"].astype(bool)
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "n_scenarios": int(len(scenarios)),
        "n_complete_blocks": int(len(blocks)),
        "supporting_scenarios": int(supported.sum()),
        "supporting_scenario_share": float(supported.mean()),
        "scenario_effect_ranges": {
            "geometric_volume_ratio": _quantiles(scenarios["geometric_volume_ratio"]),
            "mean_leaf_depth_delta": _quantiles(scenarios["mean_leaf_depth_delta"]),
            "mean_max_depth_delta": _quantiles(scenarios["mean_max_depth_delta"]),
            "shallow_swarm_block_share": _quantiles(
                scenarios["shallow_swarm_block_share"]
            ),
        },
    }
    return blocks, scenarios, manifest


def write_sensitivity_analysis(
    records_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    records_file = Path(records_path)
    records = []
    with records_file.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_number}") from exc
    blocks, scenarios, manifest = summarize_sensitivity_records(records)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    blocks.to_csv(output / "sensitivity_blocks.csv", index=False)
    scenarios.to_csv(output / "sensitivity_scenarios.csv", index=False)
    manifest["records_path"] = str(records_file)
    manifest["records_sha256"] = _sha256(records_file)
    (output / "sensitivity_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def _transform(base: Any, draw: float | int, mode: str) -> float:
    base_value = float(base)
    draw_value = float(draw)
    if mode == "scale":
        return base_value * draw_value
    if mode == "offset":
        return base_value + draw_value
    if mode == "value":
        return draw_value
    raise ValueError(f"Unknown parameter transformation mode: {mode}")


def _quantiles(values: pd.Series) -> dict[str, float]:
    numeric = values.astype(float)
    return {
        "min": float(numeric.min()),
        "q05": float(numeric.quantile(0.05)),
        "median": float(numeric.median()),
        "q95": float(numeric.quantile(0.95)),
        "max": float(numeric.max()),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
