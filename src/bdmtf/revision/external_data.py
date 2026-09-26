from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bdmtf.revision.data_pipeline import cascade_metrics, parse_timestamp
from bdmtf.revision.provenance import sha256_file, write_json


REQUIRED_EVENT_COLUMNS = ("content_id", "event_id", "parent_event_id", "created_at")
OPTIONAL_EVENT_DEFAULTS: dict[str, Any] = {
    "platform": "unknown",
    "community": "unknown",
    "author_id": "",
    "text": "",
    "score": 0.0,
    "event_type": "comment",
    "removed": False,
    "content_url": "",
}


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("events", payload.get("data", []))
        return pd.DataFrame(payload)
    return pd.read_csv(path, encoding="utf-8", encoding_errors="replace")


def normalize_events(
    frame: pd.DataFrame,
    column_map: dict[str, str] | None = None,
    platform: str = "unknown",
    community: str = "unknown",
) -> pd.DataFrame:
    """Normalize a platform export to the shared cascade-event contract."""
    column_map = column_map or {}
    renamed = frame.rename(columns={source: target for target, source in column_map.items() if source in frame}).copy()
    missing = [name for name in REQUIRED_EVENT_COLUMNS if name not in renamed]
    if missing:
        raise ValueError(f"Missing canonical event columns: {', '.join(missing)}")

    for name, default in OPTIONAL_EVENT_DEFAULTS.items():
        if name not in renamed:
            renamed[name] = default
    if platform != "unknown":
        renamed["platform"] = platform
    if community != "unknown":
        renamed["community"] = community

    renamed["content_id"] = renamed["content_id"].astype(str)
    renamed["event_id"] = renamed["event_id"].astype(str)
    renamed["parent_event_id"] = renamed["parent_event_id"].fillna("").astype(str)
    renamed["created_at"] = parse_timestamp(renamed["created_at"])
    if renamed["created_at"].isna().any():
        raise ValueError("Some external events have invalid timestamps")
    if renamed.duplicated(["platform", "event_id"]).any():
        raise ValueError("External event IDs must be unique within a platform")

    renamed["depth"] = _reconstruct_depths(renamed)
    _validate_parent_links(renamed)
    ordered = [
        "platform",
        "community",
        "content_id",
        "event_id",
        "parent_event_id",
        "created_at",
        "depth",
        "author_id",
        "text",
        "score",
        "event_type",
        "removed",
        "content_url",
    ]
    return renamed[ordered].sort_values(["platform", "community", "content_id", "created_at", "event_id"]).reset_index(drop=True)


def _validate_parent_links(events: pd.DataFrame) -> None:
    for _, group in events.groupby(["platform", "content_id"], sort=False):
        ids = set(group["event_id"].astype(str))
        explicit_root = group["event_type"].astype(str).str.lower().isin({"post", "root"}).any()
        parents = group["parent_event_id"].fillna("").astype(str)
        missing = parents[(parents != "") & ~parents.isin(ids)]
        if explicit_root and not missing.empty:
            raise ValueError("External cascade with an explicit root contains missing parent events")
        times = dict(zip(group["event_id"].astype(str), group["created_at"]))
        for row in group.itertuples(index=False):
            parent_time = times.get(str(row.parent_event_id))
            if parent_time is not None and parent_time > row.created_at:
                raise ValueError("External cascade parent event occurs after its child")


def _reconstruct_depths(events: pd.DataFrame) -> pd.Series:
    depths = pd.Series(index=events.index, dtype="int64")
    for _, group in events.groupby(["platform", "content_id"], sort=False):
        id_to_index = dict(zip(group["event_id"].astype(str), group.index))
        parents = group["parent_event_id"].fillna("").astype(str).to_dict()
        cache: dict[int, int] = {}

        def resolve(index: int, trail: set[int]) -> int:
            if index in cache:
                return cache[index]
            if index in trail:
                raise ValueError("Cycle detected in external cascade parent links")
            parent = parents[index]
            parent_index = id_to_index.get(parent)
            if parent_index is None:
                value = 1
            else:
                value = resolve(parent_index, trail | {index}) + 1
            cache[index] = value
            return value

        for index in group.index:
            depths.loc[index] = resolve(index, set())
    return depths.astype(int)


def external_cascade_metrics(events: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    keys = ["platform", "community", "content_id"]
    for (platform, community, content_id), group in events.groupby(keys, sort=False):
        root_mask = group["event_type"].astype(str).str.lower().isin({"post", "root"})
        comments = group[~root_mask].copy()
        if comments.empty:
            continue
        start = group["created_at"].min()
        depth_offset = 2 if root_mask.any() else 1
        comments_for_metric = pd.DataFrame(
            {
                "comment_id": comments["event_id"],
                "parent_id": comments["parent_event_id"],
                "depth": (comments["depth"] - depth_offset).clip(lower=0),
                "minutes_since_post": (comments["created_at"] - start).dt.total_seconds() / 60.0,
                "author": comments["author_id"],
                "comment_text": np.where(comments["removed"].astype(bool), "[removed]", comments["text"]),
            }
        )
        metrics = cascade_metrics(comments_for_metric, str(content_id), str(community))
        if metrics:
            metrics["platform"] = str(platform)
            records.append(metrics)
    return pd.DataFrame(records)


def audit_external_datasets(root: Path, config: dict[str, Any], output_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_metrics: list[pd.DataFrame] = []
    datasets: list[dict[str, Any]] = []
    for item in config.get("cross_platform", {}).get("datasets", []):
        configured_path = Path(str(item.get("path", "")))
        path = configured_path if configured_path.is_absolute() else root / configured_path
        record: dict[str, Any] = {"name": item.get("name", path.stem), "path": str(path)}
        if not path.is_file():
            record.update({"status": "missing", "reason": "configured file does not exist"})
            datasets.append(record)
            continue
        try:
            normalized = normalize_events(
                read_table(path),
                item.get("column_map", {}),
                str(item.get("platform", "unknown")),
                str(item.get("community", "unknown")),
            )
            metrics = external_cascade_metrics(normalized)
            dataset_name = str(record["name"])
            normalized.to_parquet(output_dir / f"{dataset_name}_events.parquet", index=False)
            metrics.to_parquet(output_dir / f"{dataset_name}_cascade_metrics.parquet", index=False)
            metrics.to_csv(output_dir / f"{dataset_name}_cascade_metrics.csv", index=False)
            if not metrics.empty:
                all_metrics.append(metrics.assign(dataset=dataset_name))
            record.update(
                {
                    "status": "ready" if not metrics.empty else "no_cascades",
                    "sha256": sha256_file(path),
                    "n_events": int(len(normalized)),
                    "n_cascades": int(len(metrics)),
                    "platforms": sorted(normalized["platform"].unique().astype(str).tolist()),
                    "communities": int(normalized["community"].nunique()),
                }
            )
            if metrics.empty:
                record["reason"] = "No complete reply cascade could be reconstructed from this dataset"
        except (ValueError, KeyError, TypeError) as exc:
            record.update({"status": "invalid", "reason": str(exc)})
        datasets.append(record)

    combined = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
    if not combined.empty:
        combined.to_parquet(output_dir / "external_cascade_metrics.parquet", index=False)
        combined.to_csv(output_dir / "external_cascade_metrics.csv", index=False)
    claim_allowed = bool(datasets) and all(item.get("status") == "ready" for item in datasets)
    manifest = {
        "status": "ready" if claim_allowed else "needs_data",
        "claim_allowed": claim_allowed,
        "n_configured_datasets": len(datasets),
        "datasets": datasets,
        "canonical_required_columns": list(REQUIRED_EVENT_COLUMNS),
        "note": (
            "Configured external datasets passed structural audit; claim scope remains limited to these datasets."
            if claim_allowed
            else "Cross-platform claims are disabled until at least one real external dataset passes this audit."
        ),
    }
    write_json(output_dir / "external_data_audit.json", manifest)
    return combined, manifest
