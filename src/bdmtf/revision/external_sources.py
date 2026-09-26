from __future__ import annotations

import hashlib
import ipaddress
import json
import re
import shutil
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Iterator
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

import numpy as np
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer

from bdmtf.revision.contracts import InterventionEvent, MatchedStory, records_frame, validate_platform_events
from bdmtf.revision.external_data import normalize_events, read_table
from bdmtf.revision.provenance import sha256_file, write_json


TBBT_RECORD_URL = "https://zenodo.org/records/18245670"
TBBT_API_ROOT = "https://zenodo.org/api/records/18245670/files"
TBBT_FILES: dict[str, dict[str, Any]] = {
    "ban.zip": {
        "md5": "fdc368b3ac87ae9206f29270a532034e",
        "bytes": 1_500_000_000,
        "category": "ban",
    },
    "post_removal.zip": {
        "md5": "39d365174b41446408b20e70b578eae9",
        "bytes": 1_900_000_000,
        "category": "post_removal",
    },
    "quarantine.zip": {
        "md5": "673cd48159e002b0a6f316518f2efc5c",
        "bytes": 685_600_000,
        "category": "quarantine",
    },
    "migration.zip": {
        "md5": "5076a745c24db96cb0c04629aef15f4e",
        "bytes": 23_100_000,
        "category": "migration",
    },
}

TBBT_INTERVENTIONS: tuple[dict[str, str], ...] = (
    {"date": "2022-06-01", "type": "post_removal", "community": "AskReddit"},
    {"date": "2019-06-26", "type": "quarantine", "community": "The_Donald"},
    {"date": "2019-08-06", "type": "quarantine", "community": "ChapoTrapHouse"},
    {"date": "2018-09-28", "type": "quarantine", "community": "Braincels"},
    {"date": "2020-06-29", "type": "ban", "community": "ChapoTrapHouse"},
    {"date": "2022-06-01", "type": "post_removal", "community": "science"},
    {"date": "2018-09-12", "type": "ban", "community": "greatawakening"},
    {"date": "2022-03-23", "type": "quarantine", "community": "GenZedong"},
    {"date": "2017-11-07", "type": "ban", "community": "Incels"},
    {"date": "2020-06-29", "type": "ban", "community": "The_Donald"},
    {"date": "2018-09-10", "type": "ban", "community": "MillionDollarExtreme"},
    {"date": "2018-09-28", "type": "quarantine", "community": "TheRedPill"},
    {"date": "2015-06-10", "type": "migration", "community": "fatpeoplehate"},
    {"date": "2022-03-24", "type": "ban", "community": "Chodi"},
    {"date": "2018-03-21", "type": "ban", "community": "DarkNetMarkets"},
    {"date": "2020-06-29", "type": "ban", "community": "ConsumeProduct"},
    {"date": "2020-06-29", "type": "ban", "community": "GenderCritical"},
    {"date": "2023-02-23", "type": "quarantine", "community": "goblin"},
    {"date": "2018-03-14", "type": "ban", "community": "SanctionedSuicide"},
    {"date": "2020-06-29", "type": "ban", "community": "DarkHumorAndMemes"},
    {"date": "2018-09-12", "type": "migration", "community": "greatawakening"},
    {"date": "2017-08-15", "type": "ban", "community": "Physical_Removal"},
    {"date": "2020-06-29", "type": "ban", "community": "DebateAltRight"},
    {"date": "2018-03-21", "type": "ban", "community": "GunsForSale"},
    {"date": "2020-06-29", "type": "ban", "community": "ShitNeoconsSay"},
)

TRACKING_QUERY_PREFIXES = ("utm_",)
TRACKING_QUERY_KEYS = {
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "ref",
    "ref_src",
    "source",
}


def _md5_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def fetch_tbbt(
    output_root: Path,
    execute: bool = False,
    categories: Iterable[str] | None = None,
    timeout: int = 60,
) -> dict[str, Any]:
    """Create a resumable TBBT acquisition manifest and optionally download archives."""
    output_root.mkdir(parents=True, exist_ok=True)
    requested = set(categories or (item["category"] for item in TBBT_FILES.values()))
    records: list[dict[str, Any]] = []
    for filename, metadata in TBBT_FILES.items():
        if metadata["category"] not in requested:
            continue
        path = output_root / filename
        url = f"{TBBT_API_ROOT}/{filename}/content"
        record = {
            "filename": filename,
            "category": metadata["category"],
            "url": url,
            "expected_md5": metadata["md5"],
            "expected_bytes_approx": metadata["bytes"],
            "path": str(path.resolve()),
        }
        if execute:
            _download_resumable(url, path, timeout)
        if path.is_file():
            record["bytes"] = path.stat().st_size
            record["complete"] = _md5_file(path) == metadata["md5"]
            record["observed_md5"] = _md5_file(path) if record["complete"] else ""
        else:
            record.update({"bytes": 0, "complete": False, "observed_md5": ""})
        records.append(record)
    complete = bool(records) and all(item["complete"] for item in records)
    manifest = {
        "status": "complete" if complete else ("partial" if any(item["bytes"] for item in records) else "planned"),
        "source": TBBT_RECORD_URL,
        "license": "CC BY-NC-ND 4.0",
        "redistribution_note": "Raw TBBT archives are not committed or redistributed.",
        "intervention_count": len(TBBT_INTERVENTIONS),
        "files": records,
    }
    write_json(output_root.parent / "fetch_manifest.json", manifest)
    return manifest


def _download_resumable(url: str, path: Path, timeout: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    offset = path.stat().st_size if path.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with requests.get(url, headers=headers, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        mode = "ab" if offset and response.status_code == 206 else "wb"
        with path.open(mode) as handle:
            for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                if chunk:
                    handle.write(chunk)


def _iter_zip_json(path: Path, max_records: int = 0) -> Iterator[tuple[str, dict[str, Any]]]:
    seen = 0
    with zipfile.ZipFile(path) as archive:
        for member in sorted(archive.namelist()):
            basename = member.rsplit("/", 1)[-1]
            if (
                member.endswith("/")
                or member.startswith("__MACOSX/")
                or basename.startswith(".")
                or basename.startswith("._")
            ):
                continue
            with archive.open(member) as handle:
                for raw in handle:
                    try:
                        item = json.loads(raw)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        continue
                    if isinstance(item, dict):
                        yield member, item
                        seen += 1
                    if max_records and seen >= max_records:
                        return


def _infer_tbbt_slice(member: str, archive_category: str) -> dict[str, Any]:
    lowered = member.lower().replace("\\", "/")
    period = "after" if "after" in lowered else "before" if "before" in lowered else "unknown"
    scope = "out" if re.search(r"(^|[/_\-])out([/_\-.]|$)", lowered) else "in"
    platform = "voat" if "voat" in lowered else "reddit"
    return {"period_label": period, "scope": scope, "platform": platform, "intervention_type": archive_category}


def _tbbt_intervention_slug(member: str) -> str:
    parts = [part for part in member.lower().replace("\\", "/").split("/") if part]
    return parts[1] if len(parts) > 2 else "unknown"


def _tbbt_catalog() -> pd.DataFrame:
    catalog = pd.DataFrame(TBBT_INTERVENTIONS)
    catalog.insert(0, "intervention_id", [f"tbbt-{index:02d}" for index in range(1, len(catalog) + 1)])
    catalog.insert(
        1,
        "intervention_slug",
        [
            f"{row['date'][:4]}-{row['community'].lower()}"
            for row in TBBT_INTERVENTIONS
        ],
    )
    return catalog


def _hll_add(registers: bytearray, author: str) -> None:
    precision = 6
    digest = int.from_bytes(
        hashlib.blake2b(author.encode("utf-8"), digest_size=8).digest(),
        "big",
    )
    index = digest & ((1 << precision) - 1)
    remainder = digest >> precision
    width = 64 - precision
    rank = width - remainder.bit_length() + 1 if remainder else width + 1
    registers[index] = max(registers[index], rank)


def _hll_count(registers: bytearray) -> float:
    buckets = len(registers)
    alpha = 0.709
    estimate = alpha * buckets * buckets / sum(2.0 ** (-value) for value in registers)
    zero_count = registers.count(0)
    if estimate <= 2.5 * buckets and zero_count:
        return float(buckets * np.log(buckets / zero_count))
    return float(estimate)


def build_tbbt_outcome_panel(
    daily_panel: pd.DataFrame,
    catalog: pd.DataFrame,
    output_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if daily_panel.empty:
        result = {
            "status": "needs_data",
            "claim_allowed": False,
            "reason": "The TBBT daily panel is empty.",
        }
        write_json(output_dir / "outcome_panel_manifest.json", result)
        return pd.DataFrame(), result
    cohort = catalog[["intervention_id", "date"]].rename(columns={"date": "intervention_date"})
    enriched = daily_panel.merge(cohort, on="intervention_id", how="left", validate="many_to_one")
    period = pd.to_datetime(enriched["period"], utc=True, errors="coerce")
    intervention_date = pd.to_datetime(enriched["intervention_date"], utc=True, errors="coerce")
    enriched["relative_period"] = (period.dt.floor("D") - intervention_date.dt.floor("D")).dt.days
    enriched["period"] = period.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    enriched["unit_id"] = (
        enriched["intervention_id"].astype(str)
        + ":"
        + enriched["platform"].astype(str)
        + ":"
        + enriched["scope"].astype(str)
    )
    enriched["treated"] = 1
    enriched["treatment_cohort"] = intervention_date.dt.strftime("%Y-%m-%d")
    enriched["source_id"] = TBBT_RECORD_URL
    series_columns = [
        "platform",
        "unit_id",
        "community_id",
        "period",
        "relative_period",
        "treated",
        "treatment_cohort",
        "intervention_id",
        "intervention_type",
        "intervention_slug",
        "scope",
        "source_id",
    ]
    enriched = (
        enriched.groupby(series_columns, as_index=False, dropna=False)
        .agg(
            period_label=(
                "period_label",
                lambda values: "+".join(sorted(set(values.astype(str)))),
            ),
            messages=("messages", "sum"),
            active_authors_estimate=("active_authors_estimate", "max"),
            score_sum=("score_sum", "sum"),
            removed_messages=("removed_messages", "sum"),
        )
    )
    value_columns = ["messages", "active_authors_estimate", "score_sum", "removed_messages"]
    long_panel = enriched.melt(
        id_vars=[*series_columns, "period_label"],
        value_vars=value_columns,
        var_name="outcome",
        value_name="value",
    )
    long_panel["control_type"] = "none_in_tbbt"
    long_panel["control_validated"] = False
    long_panel.to_parquet(output_dir / "tbbt_outcome_panel.parquet", index=False)
    long_panel.to_csv(output_dir / "tbbt_outcome_panel.csv", index=False)
    coverage = (
        enriched.groupby(["intervention_type", "scope", "period_label", "platform"])
        .agg(interventions=("intervention_id", "nunique"), rows=("period", "size"))
        .reset_index()
    )
    coverage.to_csv(output_dir / "series_coverage.csv", index=False)
    result = {
        "status": "descriptive_ready",
        "claim_allowed": False,
        "n_rows": int(len(long_panel)),
        "n_units": int(long_panel["unit_id"].nunique()),
        "n_interventions": int(long_panel["intervention_id"].nunique()),
        "outcomes": value_columns,
        "panel_path": str(output_dir / "tbbt_outcome_panel.parquet"),
        "control_audit": "TBBT contains activity slices for affected users, not matched untreated communities.",
        "allowed_use": (
            "Uncontrolled interrupted-series and migration descriptions with explicit identifying limits."
        ),
        "prohibited_use": "Do not label OUT activity as a never-treated community control.",
    }
    write_json(output_dir / "outcome_panel_manifest.json", result)
    return long_panel, result


def import_tbbt(
    raw_root: Path,
    output_dir: Path,
    max_records_per_archive: int = 0,
    categories: set[str] | None = None,
) -> dict[str, Any]:
    """Stream TBBT archives into privacy-preserving daily outcome panels."""
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_categories = set(categories or ())
    partition_dir = output_dir / "partitions"
    partition_dir.mkdir(parents=True, exist_ok=True)
    catalog = _tbbt_catalog()
    catalog_by_key = {
        (str(row.type), str(row.intervention_slug)): {
            "intervention_id": str(row.intervention_id),
            "community_id": str(row.community),
        }
        for row in catalog.itertuples(index=False)
    }
    importer_version = 3
    source_records: list[dict[str, Any]] = []
    partition_paths: list[Path] = []
    total = 0
    for filename, metadata in TBBT_FILES.items():
        if selected_categories and metadata["category"] not in selected_categories:
            continue
        path = raw_root / filename
        if not path.is_file():
            source_records.append({"file": filename, "status": "missing"})
            continue
        observed_md5 = _md5_file(path)
        complete = observed_md5 == metadata["md5"]
        if not complete and not max_records_per_archive:
            source_records.append({"file": filename, "status": "incomplete", "bytes": path.stat().st_size})
            continue
        category = str(metadata["category"])
        partition_path = partition_dir / f"{category}.parquet"
        partition_manifest_path = partition_dir / f"{category}.manifest.json"
        if partition_path.is_file() and partition_manifest_path.is_file() and not max_records_per_archive:
            cached = json.loads(partition_manifest_path.read_text(encoding="utf-8"))
            if (
                cached.get("importer_version") == importer_version
                and cached.get("source_md5") == metadata["md5"]
                and cached.get("status") == "complete"
            ):
                cached_record = dict(cached)
                cached_record.update({"file": filename, "partition_reused": True})
                source_records.append(cached_record)
                partition_paths.append(partition_path)
                total += int(cached.get("records_read", 0))
                continue

        daily: dict[tuple[Any, ...], list[int]] = {}
        author_registers: dict[tuple[Any, ...], bytearray] = {}
        member_cache: dict[str, tuple[dict[str, Any], str, dict[str, str]]] = {}
        count = 0
        for member, item in _iter_zip_json(path, max_records_per_archive):
            created_utc = item.get("created_utc")
            if isinstance(created_utc, (int, float)):
                day_index = int(created_utc) // 86_400
            else:
                created = pd.to_datetime(item.get("created_at"), utc=True, errors="coerce")
                if pd.isna(created):
                    continue
                day_index = int(created.timestamp()) // 86_400
            cached_member = member_cache.get(member)
            if cached_member is None:
                slice_info = _infer_tbbt_slice(member, metadata["category"])
                intervention_slug = _tbbt_intervention_slug(member)
                intervention = catalog_by_key.get(
                    (category, intervention_slug),
                    {
                        "intervention_id": f"tbbt-unknown-{intervention_slug}",
                        "community_id": intervention_slug.split("-", 1)[-1],
                    },
                )
                cached_member = (slice_info, intervention_slug, intervention)
                member_cache[member] = cached_member
            slice_info, intervention_slug, intervention = cached_member
            platform = "voat" if item.get("subverse") else slice_info["platform"]
            key = (
                platform,
                category,
                intervention["intervention_id"],
                intervention_slug,
                intervention["community_id"],
                slice_info["scope"],
                slice_info["period_label"],
                day_index,
            )
            aggregates = daily.setdefault(key, [0, 0, 0])
            aggregates[0] += 1
            aggregates[1] += int(item.get("score") or 0)
            if item.get("removal_reason"):
                aggregates[2] += 1
            author = str(item.get("author") or "")
            if author:
                _hll_add(author_registers.setdefault(key, bytearray(64)), author)
            count += 1
            total += 1
        rows: list[dict[str, Any]] = []
        for key, aggregates in daily.items():
            rows.append(
            {
                "platform": key[0],
                "intervention_type": key[1],
                "intervention_id": key[2],
                "intervention_slug": key[3],
                "community_id": key[4],
                "scope": key[5],
                "period_label": key[6],
                "period": datetime.fromtimestamp(key[7] * 86_400, tz=timezone.utc).isoformat(),
                "messages": aggregates[0],
                "score_sum": aggregates[1],
                "removed_messages": aggregates[2],
                "active_authors_estimate": _hll_count(author_registers.get(key, bytearray(64))),
            }
        )
        partition = pd.DataFrame(rows)
        partition.to_parquet(partition_path, index=False)
        source_record = {
            "file": filename,
            "status": "complete" if complete else "partial_sample",
            "sha256": sha256_file(path),
            "source_md5": observed_md5,
            "records_read": count,
            "panel_rows": int(len(partition)),
            "importer_version": importer_version,
            "partition_reused": False,
        }
        write_json(partition_manifest_path, source_record)
        source_records.append(source_record)
        partition_paths.append(partition_path)

    partition_frames = [pd.read_parquet(path) for path in partition_paths if path.is_file()]
    panel = pd.concat(partition_frames, ignore_index=True) if partition_frames else pd.DataFrame()
    if not panel.empty:
        panel = panel.sort_values(
            ["intervention_id", "platform", "scope", "period"],
            kind="stable",
        ).reset_index(drop=True)
        panel.to_parquet(output_dir / "tbbt_daily_panel.parquet", index=False)
        panel.to_csv(output_dir / "tbbt_daily_panel.csv", index=False)
    catalog.to_csv(output_dir / "tbbt_intervention_catalog.csv", index=False)
    _, outcome_manifest = build_tbbt_outcome_panel(panel, catalog, output_dir)
    data_complete = bool(total) and all(item["status"] == "complete" for item in source_records)
    result = {
        "status": "complete" if data_complete else "partial",
        "data_complete": data_complete,
        "descriptive_use_allowed": data_complete,
        "claim_allowed": False,
        "selected_categories": sorted(selected_categories) if selected_categories else sorted(
            {item["category"] for item in TBBT_FILES.values()}
        ),
        "records_read": total,
        "panel_rows": int(len(panel)),
        "interventions": len(TBBT_INTERVENTIONS),
        "outcome_panel": outcome_manifest,
        "sources": source_records,
        "text_limit": "Raw TBBT text, URLs, and author identifiers are discarded after aggregation.",
        "causal_limit": "TBBT alone is observational. Causal claims require defensible controls and diagnostics.",
        "aggregation_unit": "intervention x platform x IN/OUT scope x day",
        "author_count_method": "Per-day HyperLogLog with 64 registers; raw author identifiers are discarded.",
        "importer_version": importer_version,
    }
    write_json(output_dir / "import_manifest.json", result)
    return result


LEMMY_ACTIONS: dict[str, tuple[str, str]] = {
    "removed_posts": ("remove_post", "mod_remove_post"),
    "removed_comments": ("remove_comment", "mod_remove_comment"),
    "locked_posts": ("lock_post", "mod_lock_post"),
    "featured_posts": ("feature_post", "mod_feature_post"),
}


def parse_lemmy_modlog(payload: dict[str, Any], instance: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    interventions: list[InterventionEvent] = []
    excluded: list[dict[str, Any]] = []
    for collection, (action_type, action_key) in LEMMY_ACTIONS.items():
        for item in payload.get(collection, []) or []:
            action = item.get(action_key, {}) or {}
            target = item.get("post") or item.get("comment") or {}
            community = item.get("community", {}) or {}
            active_field = "locked" if action_type == "lock_post" else "removed"
            active = bool(action.get(active_field, True))
            is_restore = not active
            intervention_id = f"lemmy:{instance}:{action_type}:{action.get('id', '')}"
            content_id = str(target.get("post_id") or target.get("id") or "")
            record = InterventionEvent(
                platform="lemmy",
                intervention_id=intervention_id,
                unit_id=f"{instance}:{content_id}",
                community_id=str(community.get("id") or target.get("community_id") or "unknown"),
                intervention_type=action_type,
                occurred_at=action.get("when_") or action.get("published"),
                active=active,
                moderator_id=str((item.get("moderator") or {}).get("id") or ""),
                reason=str(action.get("reason") or ""),
                content_id=content_id,
                event_id=str(target.get("id") or ""),
                source_id=str(target.get("ap_id") or ""),
                excluded_reason="restore_event" if is_restore else "",
            )
            if is_restore:
                excluded.append(record.to_record())
            else:
                interventions.append(record)
    for key in (
        "purged_posts",
        "purged_comments",
        "purged_communities",
        "admin_purged_persons",
        "admin_purged_posts",
        "admin_purged_comments",
        "admin_purged_communities",
    ):
        for item in payload.get(key, []) or []:
            excluded.append(
                {
                    "platform": "lemmy",
                    "intervention_type": key,
                    "excluded_reason": "purge_event_not_comparable",
                    "source": json.dumps(item, ensure_ascii=True, default=str),
                }
            )
    return records_frame(interventions), pd.DataFrame(excluded)


def extract_lemmy_post_events(payload: dict[str, Any], instance: str) -> pd.DataFrame:
    records: dict[str, dict[str, Any]] = {}
    for collection in LEMMY_ACTIONS:
        for item in payload.get(collection, []) or []:
            post = item.get("post") or {}
            if not post.get("id"):
                continue
            community = item.get("community", {}) or {}
            event_id = f"{instance}:post:{post['id']}"
            records[event_id] = {
                "platform": "lemmy",
                "community": str(community.get("name") or community.get("id") or post.get("community_id") or "unknown"),
                "content_id": str(post["id"]),
                "event_id": event_id,
                "parent_event_id": "",
                "created_at": post.get("published"),
                "author_id": "",
                "text": str(post.get("body") or ""),
                "title": str(post.get("name") or ""),
                "score": 0.0,
                "event_type": "post",
                "removed": bool(post.get("removed")),
                "content_url": str(post.get("url") or post.get("ap_id") or ""),
            }
    return pd.DataFrame(records.values())


def collect_lemmy(
    output_dir: Path,
    instances: Iterable[str],
    pages: int = 1,
    page_size: int = 50,
    timeout: int = 60,
    sleep_seconds: float = 0.2,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    interventions: list[pd.DataFrame] = []
    excluded: list[pd.DataFrame] = []
    post_events: list[pd.DataFrame] = []
    sources: list[dict[str, Any]] = []
    for instance in instances:
        base = instance.rstrip("/")
        for page in range(1, pages + 1):
            response = requests.get(
                f"{base}/api/v3/modlog",
                params={"page": page, "limit": page_size},
                timeout=timeout,
            )
            response.raise_for_status()
            payload = response.json()
            slug = re.sub(r"[^a-z0-9]+", "-", urlsplit(base).netloc.lower()).strip("-")
            raw_path = raw_dir / f"{slug}-page-{page:04d}.json"
            raw_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            parsed, removed = parse_lemmy_modlog(payload, urlsplit(base).netloc)
            post_events.append(extract_lemmy_post_events(payload, urlsplit(base).netloc))
            interventions.append(parsed)
            excluded.append(removed)
            sources.append(
                {
                    "instance": base,
                    "page": page,
                    "sha256": sha256_file(raw_path),
                    "interventions": int(len(parsed)),
                    "excluded": int(len(removed)),
                }
            )
            time.sleep(sleep_seconds)
    intervention_frame = pd.concat(interventions, ignore_index=True) if interventions else pd.DataFrame()
    excluded_frame = pd.concat(excluded, ignore_index=True) if excluded else pd.DataFrame()
    post_frame = pd.concat(post_events, ignore_index=True) if post_events else pd.DataFrame()
    if not intervention_frame.empty:
        intervention_frame = intervention_frame.drop_duplicates("intervention_id")
        intervention_frame.to_parquet(output_dir / "lemmy_interventions.parquet", index=False)
        intervention_frame.to_csv(output_dir / "lemmy_interventions.csv", index=False)
    if not excluded_frame.empty:
        excluded_frame.to_csv(output_dir / "lemmy_excluded_events.csv", index=False)
    if not post_frame.empty:
        post_frame = post_frame.drop_duplicates(["platform", "event_id"])
        post_frame.to_parquet(output_dir / "lemmy_post_events.parquet", index=False)
        post_frame.to_csv(output_dir / "lemmy_post_events.csv", index=False)
    manifest = {
        "status": "complete" if sources else "needs_data",
        "claim_allowed": bool(sources and len(intervention_frame)),
        "instances": list(instances),
        "n_interventions": int(len(intervention_frame)),
        "n_excluded": int(len(excluded_frame)),
        "n_observed_posts": int(len(post_frame)),
        "sources": sources,
        "note": "Modlog events establish treatment times; outcome threads must be collected separately.",
    }
    write_json(output_dir / "collection_manifest.json", manifest)
    return manifest


def import_reddit_expansion(
    inputs: Iterable[Path],
    output_dir: Path,
    column_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Normalize user-supplied Reddit exports without pretending missing communities were collected."""
    output_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    sources = []
    for path in inputs:
        if not path.is_file():
            sources.append({"path": str(path), "status": "missing"})
            continue
        frame = normalize_events(read_table(path), column_map or {}, platform="reddit")
        frames.append(frame)
        sources.append(
            {
                "path": str(path.resolve()),
                "status": "ready",
                "sha256": sha256_file(path),
                "events": int(len(frame)),
                "communities": int(frame["community"].nunique()),
            }
        )
    combined = pd.concat(frames, ignore_index=True).drop_duplicates(["platform", "event_id"]) if frames else pd.DataFrame()
    audit = validate_platform_events(
        combined.rename(
            columns={
                "community": "community_id",
                "content_url": "url",
            }
        )
    ) if not combined.empty else {"status": "needs_data", "claim_allowed": False, "n_events": 0, "issues": ["no inputs"]}
    if not combined.empty:
        combined.to_parquet(output_dir / "reddit_expanded_events.parquet", index=False)
        combined.to_csv(output_dir / "reddit_expanded_events.csv", index=False)
    communities = int(combined["community"].nunique()) if not combined.empty else 0
    cascades = int(combined["content_id"].nunique()) if not combined.empty else 0
    result = {
        **audit,
        "sources": sources,
        "n_communities": communities,
        "n_cascades": cascades,
        "target_met": communities >= 20 and cascades >= 10_000,
        "claim_allowed": bool(audit.get("claim_allowed")) and communities >= 20 and cascades >= 10_000,
    }
    write_json(output_dir / "expansion_manifest.json", result)
    return result


def canonicalize_url(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if "://" not in text:
        text = "https://" + text
    try:
        parsed = urlsplit(text)
    except ValueError:
        return ""
    host = (parsed.hostname or "").lower().removeprefix("www.")
    if not host:
        return ""
    try:
        port = parsed.port
    except ValueError:
        return ""
    netloc = host if port in (None, 80, 443) else f"{host}:{port}"
    path = re.sub(r"/+", "/", parsed.path or "/")
    if path != "/":
        path = path.rstrip("/")
    query = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key.lower() not in TRACKING_QUERY_KEYS
        and not any(key.lower().startswith(prefix) for prefix in TRACKING_QUERY_PREFIXES)
    ]
    return urlunsplit(("https", netloc, path, urlencode(sorted(query)), ""))


class _CanonicalLinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.href = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self.href or tag.lower() != "link":
            return
        values = {str(key).lower(): str(value or "") for key, value in attrs}
        if "canonical" in values.get("rel", "").lower().split() and values.get("href"):
            self.href = values["href"].strip()


def _network_url(value: Any) -> str:
    text = str(value or "").strip()
    if text and "://" not in text:
        return "https://" + text
    return text


def _is_public_http_url(value: str) -> bool:
    try:
        parsed = urlsplit(value)
        host = (parsed.hostname or "").lower()
    except ValueError:
        return False
    if parsed.scheme not in {"http", "https"} or not host:
        return False
    if host == "localhost" or host.endswith((".localhost", ".local")):
        return False
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return True
    return not (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
    )


def _resolve_one_story_url(source_url: str, timeout: float) -> dict[str, Any]:
    requested_url = _network_url(source_url)
    base = {
        "source_url": source_url,
        "requested_url": requested_url,
        "resolved_at": datetime.now(timezone.utc).isoformat(),
    }
    if not _is_public_http_url(requested_url):
        return {**base, "status": "skipped_unsafe", "resolved_url": "", "error": "non-public HTTP(S) URL"}
    response = None
    try:
        response = requests.get(
            requested_url,
            allow_redirects=True,
            timeout=timeout,
            headers={"User-Agent": "BDMTF-reproducibility-audit/1.0"},
            stream=True,
        )
        response.raise_for_status()
        final_url = str(response.url)
        canonical_href = ""
        content_type = str(response.headers.get("Content-Type", "")).lower()
        if "html" in content_type:
            payload = bytearray()
            for chunk in response.iter_content(chunk_size=16_384):
                payload.extend(chunk)
                if len(payload) >= 65_536:
                    break
            parser = _CanonicalLinkParser()
            parser.feed(bytes(payload[:65_536]).decode(response.encoding or "utf-8", errors="replace"))
            canonical_href = urljoin(final_url, parser.href) if parser.href else ""
        selected_url = canonical_href or final_url
        if not _is_public_http_url(selected_url):
            raise ValueError("response canonical URL is not public HTTP(S)")
        resolved = canonicalize_url(selected_url)
        if not resolved:
            raise ValueError("response did not yield a canonical HTTP(S) URL")
        return {
            **base,
            "status": "resolved",
            "resolved_url": resolved,
            "redirected_url": canonicalize_url(final_url),
            "html_canonical_url": canonicalize_url(canonical_href),
            "redirect_count": len(response.history),
            "http_status": int(response.status_code),
            "error": "",
        }
    except (requests.RequestException, ValueError, UnicodeError) as exc:
        return {**base, "status": "failed", "resolved_url": "", "error": f"{type(exc).__name__}: {exc}"}
    finally:
        if response is not None:
            response.close()


def resolve_story_urls(
    roots: pd.DataFrame,
    cache_path: Path,
    execute: bool = False,
    workers: int = 8,
    timeout: float = 10.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Resolve redirects and HTML canonical links, with deterministic offline cache replay."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache: dict[str, Any] = {"version": 1, "records": {}}
    if cache_path.is_file():
        try:
            loaded = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict) and isinstance(loaded.get("records"), dict):
                cache = loaded
        except (OSError, json.JSONDecodeError):
            cache = {"version": 1, "records": {}}
    records: dict[str, dict[str, Any]] = cache["records"]
    source_by_key = {
        canonicalize_url(row.source_url): str(row.source_url)
        for row in roots[["source_url"]].drop_duplicates().itertuples(index=False)
        if canonicalize_url(row.source_url)
    }
    cache_hits = sum(
        1 for key in source_by_key if records.get(key, {}).get("status") == "resolved"
    )
    pending = [key for key in source_by_key if key not in records or records[key].get("status") != "resolved"]
    attempted = 0
    if execute and pending:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {
                pool.submit(_resolve_one_story_url, source_by_key[key], timeout): key
                for key in pending
            }
            for future in as_completed(futures):
                key = futures[future]
                records[key] = future.result()
                attempted += 1
        temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps({"version": 1, "records": records}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary.replace(cache_path)

    resolved = roots.copy()
    resolved["normalized_url"] = resolved["source_url"].map(canonicalize_url)
    resolved["canonical_url"] = resolved["normalized_url"].map(
        lambda key: records.get(key, {}).get("resolved_url", "") or key
    )
    statuses = {
        str(record.get("status", "unknown"))
        for key, record in records.items()
        if key in source_by_key
    }
    resolved_count = sum(
        1 for key in source_by_key if records.get(key, {}).get("status") == "resolved"
    )
    failure_count = sum(
        1
        for key in source_by_key
        if records.get(key, {}).get("status") in {"failed", "skipped_unsafe"}
    )
    if execute:
        status = "complete" if resolved_count == len(source_by_key) else "partial"
    elif resolved_count:
        status = "cache_replay"
    else:
        status = "normalized_only"
    audit = {
        "status": status,
        "network_requested": execute,
        "unique_urls": len(source_by_key),
        "cache_hits_before_run": cache_hits,
        "attempted": attempted,
        "resolved": resolved_count,
        "failed_or_unsafe": failure_count,
        "record_statuses": sorted(statuses),
        "cache_path": str(cache_path.resolve()),
        "note": "Network resolution is opt-in; offline runs replay successful cached redirects and canonical links.",
    }
    return resolved, audit


def _root_stories(frame: pd.DataFrame) -> pd.DataFrame:
    roots = frame[frame["event_type"].astype(str).str.lower().isin({"post", "root", "story"})].copy()
    if roots.empty:
        roots = frame.sort_values("created_at").groupby(["platform", "content_id"], as_index=False).head(1).copy()
    url_source = roots["content_url"] if "content_url" in roots else roots["url"] if "url" in roots else pd.Series("", index=roots.index)
    title_source = roots["title"] if "title" in roots else roots["text"] if "text" in roots else pd.Series("", index=roots.index)
    roots["source_url"] = url_source.fillna("").astype(str)
    roots["canonical_url"] = roots["source_url"].map(canonicalize_url)
    roots["title"] = title_source.fillna("").astype(str)
    roots["created_at"] = roots["created_at"].map(
        lambda value: pd.to_datetime(value, utc=True, errors="coerce")
    )
    roots = roots[roots["created_at"].notna()].copy()
    return roots


def build_story_matches(
    events: pd.DataFrame,
    output_dir: Path,
    semantic_threshold: float = 0.72,
    max_hours: float = 72.0,
    resolve_urls: bool = False,
    resolution_workers: int = 8,
    resolution_timeout: float = 10.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    roots = _root_stories(events)
    roots, resolution_audit = resolve_story_urls(
        roots,
        output_dir / "url_resolution_cache.json",
        execute=resolve_urls,
        workers=resolution_workers,
        timeout=resolution_timeout,
    )
    roots.to_csv(output_dir / "resolved_story_roots.csv", index=False)
    exact_records: list[dict[str, Any]] = []
    for url, group in roots[roots["canonical_url"] != ""].groupby("canonical_url"):
        rows = list(group.itertuples(index=False))
        for left_index, left in enumerate(rows):
            for right in rows[left_index + 1 :]:
                if left.platform == right.platform:
                    continue
                exact_records.append(
                    MatchedStory(
                        match_id=hashlib.sha256(
                            f"exact|{left.platform}|{left.content_id}|{right.platform}|{right.content_id}|{url}".encode()
                        ).hexdigest()[:20],
                        left_platform=str(left.platform),
                        left_content_id=str(left.content_id),
                        right_platform=str(right.platform),
                        right_content_id=str(right.content_id),
                        match_type="exact_url",
                        canonical_url=url,
                        title_similarity=1.0,
                        time_distance_hours=abs((left.created_at - right.created_at).total_seconds()) / 3600.0,
                        created_at=min(left.created_at, right.created_at),
                    ).to_record()
                )

    exact_pairs = {
        frozenset(
            (
                (item["left_platform"], item["left_content_id"]),
                (item["right_platform"], item["right_content_id"]),
            )
        )
        for item in exact_records
    }
    semantic_records: list[dict[str, Any]] = []
    eligible = roots[roots["title"].str.strip() != ""].reset_index(drop=True)
    if len(eligible) >= 2:
        matrix = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit_transform(eligible["title"])
        platform_indices = {
            platform: np.flatnonzero(eligible["platform"].astype(str).to_numpy() == platform)
            for platform in sorted(eligible["platform"].astype(str).unique())
        }
        for left_platform, right_platform in combinations(platform_indices, 2):
            left_indices = platform_indices[left_platform]
            right_indices = platform_indices[right_platform]
            similarities = (matrix[left_indices] @ matrix[right_indices].T).tocoo()
            keep = similarities.data >= semantic_threshold
            for left_local, right_local, similarity in zip(
                similarities.row[keep],
                similarities.col[keep],
                similarities.data[keep],
            ):
                left = eligible.iloc[left_indices[left_local]]
                right = eligible.iloc[right_indices[right_local]]
                pair = frozenset(
                    (
                        (str(left["platform"]), str(left["content_id"])),
                        (str(right["platform"]), str(right["content_id"])),
                    )
                )
                if pair in exact_pairs:
                    continue
                distance = abs((left["created_at"] - right["created_at"]).total_seconds()) / 3600.0
                if distance > max_hours:
                    continue
                semantic_records.append(
                    MatchedStory(
                        match_id=hashlib.sha256(
                            f"semantic|{left['platform']}|{left['content_id']}|{right['platform']}|{right['content_id']}".encode()
                        ).hexdigest()[:20],
                        left_platform=str(left["platform"]),
                        left_content_id=str(left["content_id"]),
                        right_platform=str(right["platform"]),
                        right_content_id=str(right["content_id"]),
                        match_type="semantic_event",
                        canonical_url="",
                        title_similarity=float(similarity),
                        time_distance_hours=distance,
                        created_at=min(left["created_at"], right["created_at"]),
                    ).to_record()
                )
    matches = pd.DataFrame(exact_records + semantic_records)
    if not matches.empty:
        matches.to_parquet(output_dir / "story_matches.parquet", index=False)
        matches.to_csv(output_dir / "story_matches.csv", index=False)
    manifest = {
        "status": "complete",
        "n_roots": int(len(roots)),
        "n_platforms": int(roots["platform"].nunique()),
        "n_exact_url": len(exact_records),
        "n_semantic_event": len(semantic_records),
        "semantic_threshold": semantic_threshold,
        "max_hours": max_hours,
        "claim_allowed": bool(exact_records or semantic_records),
        "url_resolution": resolution_audit,
        "note": "Exact URL and semantic event matches are stored and reported separately.",
    }
    write_json(output_dir / "match_manifest.json", manifest)
    return matches, manifest


def make_prospective_split(
    events: pd.DataFrame,
    output_dir: Path,
    freeze_at: str | datetime,
    weeks: int = 8,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    freeze = pd.to_datetime(freeze_at, utc=True, errors="raise")
    end = freeze + pd.Timedelta(weeks=weeks)
    timestamps = events["created_at"].map(
        lambda value: pd.to_datetime(value, utc=True, errors="coerce")
    )
    if timestamps.isna().any():
        raise ValueError("Prospective input contains invalid created_at values")
    assigned = events.copy()
    assigned["prospective_split"] = np.select(
        [timestamps < freeze, (timestamps >= freeze) & (timestamps < end)],
        ["development", "prospective_test"],
        default="post_window",
    )
    assigned.to_parquet(output_dir / "prospective_events.parquet", index=False)
    assigned[["platform", "content_id", "created_at", "prospective_split"]].drop_duplicates().to_csv(
        output_dir / "prospective_assignments.csv",
        index=False,
    )
    source_hash = hashlib.sha256(
        pd.util.hash_pandas_object(
            assigned[["platform", "content_id", "event_id", "created_at"]],
            index=False,
        ).values.tobytes()
    ).hexdigest()
    counts = assigned["prospective_split"].value_counts().to_dict()
    manifest = {
        "status": "frozen",
        "freeze_at": freeze.isoformat(),
        "test_end": end.isoformat(),
        "weeks": weeks,
        "event_counts": {str(key): int(value) for key, value in counts.items()},
        "assignment_sha256": source_hash,
        "claim_allowed": int(counts.get("prospective_test", 0)) > 0,
        "rule": "No model fitting may read prospective_test or post_window rows.",
    }
    write_json(output_dir / "prospective_manifest.json", manifest)
    return assigned, manifest
