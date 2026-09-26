from __future__ import annotations

import hashlib
import json
import random
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from bdmtf.schema import AgentProfile, Intervention


TRAIT_MODES = {
    "coupled_original",
    "independent_marginal",
    "shuffled_marginal",
    "aggressive_constructive",
}


def make_post_core_trait_transform(
    base_agents: Sequence[AgentProfile],
    mode: str,
    seed: int,
) -> Callable[[list[AgentProfile], Intervention], list[AgentProfile]]:
    if mode not in TRAIT_MODES:
        raise ValueError(f"Unknown trait mode: {mode}")
    if mode == "coupled_original":
        return lambda shifted, intervention: shifted

    rng = random.Random(seed)
    base_prosocial = [float(agent.prosocial) for agent in base_agents]
    if mode == "independent_marginal":
        assigned = [rng.choice(base_prosocial) for _ in base_agents]
    else:
        assigned = list(base_prosocial)
        rng.shuffle(assigned)
    if mode == "aggressive_constructive":
        count = max(1, len(base_agents) // 5)
        high_antagonism = sorted(
            range(len(base_agents)),
            key=lambda index: base_agents[index].antagonism,
            reverse=True,
        )[:count]
        for index in high_antagonism:
            assigned[index] = max(0.75, assigned[index])
    by_agent = {
        int(agent.agent_id): float(assigned[index])
        for index, agent in enumerate(base_agents)
    }

    def transform(
        shifted_agents: list[AgentProfile],
        intervention: Intervention,
    ) -> list[AgentProfile]:
        return [
            replace(
                agent,
                prosocial=by_agent[int(agent.agent_id)],
                metadata=dict(
                    agent.metadata,
                    trait_mode=mode,
                    prosocial_decoupled_after_core_shift=True,
                ),
            )
            for agent in shifted_agents
        ]

    return transform


def analyze_trait_coupling(
    records: Sequence[Mapping[str, Any]],
    bootstrap_samples: int = 2000,
    seed: int = 30371,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    frame = pd.DataFrame(records)
    required = {
        "community",
        "post_id",
        "seed",
        "trait_mode",
        "condition",
        "comment_volume",
        "mean_leaf_depth",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing trait-coupling columns: {sorted(missing)}")
    key = ["community", "post_id", "seed", "trait_mode"]
    if frame.duplicated([*key, "condition"]).any():
        raise ValueError("Duplicate trait-coupling condition records")
    pivot = frame.pivot(index=key, columns="condition")
    needed = {"BASELINE", "CORE_TOXIC_CONTROVERSIAL"}
    present = set(pivot["comment_volume"].columns)
    if not needed.issubset(present):
        raise ValueError(f"Missing trait-coupling conditions: {sorted(needed - present)}")
    complete = pd.Series(True, index=pivot.index)
    for metric in ("comment_volume", "mean_leaf_depth"):
        for condition in needed:
            complete &= pivot[(metric, condition)].notna()
    pivot = pivot.loc[complete]
    if pivot.empty:
        raise ValueError("No complete trait-coupling blocks were found")
    contrasts = pivot.index.to_frame(index=False)
    baseline_volume = pivot["comment_volume"]["BASELINE"].to_numpy(dtype=float)
    toxic_volume = pivot["comment_volume"][
        "CORE_TOXIC_CONTROVERSIAL"
    ].to_numpy(dtype=float)
    baseline_leaf = pivot["mean_leaf_depth"]["BASELINE"].to_numpy(dtype=float)
    toxic_leaf = pivot["mean_leaf_depth"][
        "CORE_TOXIC_CONTROVERSIAL"
    ].to_numpy(dtype=float)
    contrasts["volume_log_effect"] = np.log1p(toxic_volume) - np.log1p(
        baseline_volume
    )
    contrasts["volume_ratio"] = np.divide(
        toxic_volume,
        np.maximum(baseline_volume, 1.0),
    )
    contrasts["leaf_depth_delta"] = toxic_leaf - baseline_leaf
    contrasts["shallow_swarm"] = (
        (toxic_volume > baseline_volume) & (toxic_leaf < baseline_leaf)
    )

    rows: list[dict[str, Any]] = []
    for mode, group in contrasts.groupby("trait_mode", sort=True):
        for metric in ("volume_log_effect", "leaf_depth_delta"):
            low, high = _cluster_bootstrap(
                group,
                metric,
                bootstrap_samples,
                _stable_seed(seed, mode, metric),
            )
            rows.append(
                {
                    "trait_mode": str(mode),
                    "metric": metric,
                    "mean": float(group[metric].mean()),
                    "median": float(group[metric].median()),
                    "ci_low": low,
                    "ci_high": high,
                    "n_blocks": int(len(group)),
                    "n_posts": int(
                        group[["community", "post_id"]].drop_duplicates().shape[0]
                    ),
                    "shallow_swarm_share": float(group["shallow_swarm"].mean()),
                }
            )
    summary = pd.DataFrame(rows)
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "complete_blocks": int(len(contrasts)),
        "complete_posts": int(
            contrasts[["community", "post_id"]].drop_duplicates().shape[0]
        ),
        "trait_modes": sorted(contrasts["trait_mode"].unique()),
        "bootstrap_samples": int(bootstrap_samples),
        "bootstrap_seed": int(seed),
        "evidence_scope": (
            "Trait-coupling robustness inside the legacy simulator; no personality "
            "trait or causal effect is identified from real users."
        ),
    }
    return contrasts, summary, manifest


def write_trait_analysis(
    records_path: str | Path,
    output_dir: str | Path,
    bootstrap_samples: int = 2000,
    seed: int = 30371,
) -> dict[str, Any]:
    source = Path(records_path)
    records = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    contrasts, summary, manifest = analyze_trait_coupling(
        records,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    contrasts_path = output / "trait_coupling_contrasts.csv"
    summary_path = output / "trait_coupling_summary.csv"
    contrasts.to_csv(contrasts_path, index=False)
    summary.to_csv(summary_path, index=False)
    manifest["records_path"] = str(source)
    manifest["records_sha256"] = _sha256(source)
    manifest["contrasts_sha256"] = _sha256(contrasts_path)
    manifest["summary_sha256"] = _sha256(summary_path)
    (output / "trait_coupling_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def _cluster_bootstrap(
    frame: pd.DataFrame,
    column: str,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    grouped = [
        group[column].to_numpy(dtype=float)
        for _, group in frame.groupby(["community", "post_id"], sort=True)
    ]
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
