from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bdmtf.revision.data_pipeline import COMMUNITIES
from bdmtf.revision.provenance import write_json


@dataclass(frozen=True)
class CommunityModel:
    community: str
    n_train_cascades: int
    log_size_mean: float
    log_size_std: float
    mean_branching_factor: float
    root_reply_share: float
    depth_decay: float
    time_decay_minutes: float
    repeat_author_share: float
    toxicity_density: float
    toxicity_log_size_effect: float
    toxicity_leaf_depth_effect: float
    counterspeech_rate: float
    removed_rate: float


def _bounded_mean(frame: pd.DataFrame, column: str, default: float, low: float, high: float) -> float:
    if column not in frame or frame[column].dropna().empty:
        return default
    return float(np.clip(frame[column].astype(float).mean(), low, high))


def _fit_profile(train: pd.DataFrame, community: str) -> CommunityModel:
    if train.empty:
        raise ValueError(f"No training cascades for {community}")
    log_size = np.log1p(train["size"].clip(lower=0).to_numpy(dtype=float))
    mean_depth = max(1.0, float(train["mean_depth"].mean()))
    toxicity = train["toxicity_density"].to_numpy(dtype=float)
    toxicity_std = float(np.std(toxicity))
    log_sizes = np.log1p(train["size"].to_numpy(dtype=float))
    leaf_depths = train["mean_leaf_depth"].to_numpy(dtype=float)
    if toxicity_std > 1e-8:
        size_effect = float(np.corrcoef(toxicity, log_sizes)[0, 1] * np.std(log_sizes))
        leaf_effect = float(np.corrcoef(toxicity, leaf_depths)[0, 1] * np.std(leaf_depths))
    else:
        size_effect = 0.0
        leaf_effect = 0.0
    return CommunityModel(
        community=community,
        n_train_cascades=int(len(train)),
        log_size_mean=float(log_size.mean()),
        log_size_std=float(max(log_size.std(ddof=1), 0.15)),
        mean_branching_factor=_bounded_mean(train, "mean_branching_factor", 1.5, 0.05, 20.0),
        root_reply_share=_bounded_mean(train, "root_reply_share", 0.5, 0.01, 0.99),
        depth_decay=float(np.clip(1.0 / mean_depth, 0.03, 0.95)),
        time_decay_minutes=_bounded_mean(train, "time_to_90_minutes", 360.0, 10.0, 10080.0) / 2.3,
        repeat_author_share=_bounded_mean(train, "repeat_author_share", 0.2, 0.0, 1.0),
        toxicity_density=_bounded_mean(train, "toxicity_density", 0.02, 0.0, 1.0),
        toxicity_log_size_effect=float(np.clip(size_effect, -2.0, 2.0)),
        toxicity_leaf_depth_effect=float(np.clip(leaf_effect, -2.0, 2.0)),
        counterspeech_rate=_bounded_mean(train, "counterspeech_rate", 0.05, 0.0, 1.0),
        removed_rate=_bounded_mean(train, "removed_rate", 0.01, 0.0, 1.0),
    )


def fit_community_model(frame: pd.DataFrame, community: str) -> CommunityModel:
    train = frame[(frame["community"] == community) & (frame["split"] == "train")].copy()
    return _fit_profile(train, community)


def fit_pooled_model(
    frame: pd.DataFrame,
    excluded_communities: tuple[str, ...] = (),
    label: str = "pooled",
) -> CommunityModel:
    """Fit a zero-shot profile without observing target-community cascades."""
    excluded = {str(value) for value in excluded_communities}
    train = frame[frame["split"].eq("train") & ~frame["community"].astype(str).isin(excluded)].copy()
    return _fit_profile(train, label)


def fit_models(metrics_path: Path, output_path: Path) -> dict[str, Any]:
    frame = pd.read_parquet(metrics_path) if metrics_path.suffix == ".parquet" else pd.read_csv(metrics_path)
    models = {community: asdict(fit_community_model(frame, community)) for community in COMMUNITIES}
    payload = {
        "fit_scope": "training split only",
        "uses_paper_target_values": False,
        "identification_limit": "Aggregate event hazards and reply target distributions only; exposure non-actions are unavailable.",
        "communities": models,
    }
    write_json(output_path, payload)
    return payload


def load_models(path: Path) -> dict[str, CommunityModel]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {name: CommunityModel(**raw) for name, raw in payload["communities"].items()}
