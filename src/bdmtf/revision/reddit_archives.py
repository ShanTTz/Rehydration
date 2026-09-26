from __future__ import annotations

import json
import shutil
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from bdmtf.revision.contracts import validate_platform_events
from bdmtf.revision.provenance import sha256_file, write_json


EVENT_COLUMNS = [
    "platform",
    "community",
    "content_id",
    "event_id",
    "parent_event_id",
    "created_at",
    "depth",
    "author_id",
    "text",
    "title",
    "score",
    "event_type",
    "removed",
    "content_url",
    "source_id",
    "is_viral",
]
NORMALIZER_VERSION = 1


def _series(frame: pd.DataFrame, name: str, default: Any = "") -> pd.Series:
    if name in frame:
        return frame[name]
    return pd.Series(default, index=frame.index)


def _clean_id(value: Any, prefix: str = "") -> str:
    text = "" if pd.isna(value) else str(value).strip()
    if prefix and text.lower().startswith(prefix.lower()):
        return text[len(prefix) :]
    return text


def _timestamps(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric.notna() & np.isfinite(numeric)
    magnitude = numeric.abs()
    result = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns, UTC]")
    unit_ranges = (
        ("s", 1e8, 5e9),
        ("ms", 1e11, 5e12),
        ("us", 1e14, 5e15),
        ("ns", 1e17, 5e18),
    )
    for unit, low, high in unit_ranges:
        mask = finite & magnitude.between(low, high, inclusive="both")
        if mask.any():
            result.loc[mask] = pd.to_datetime(
                numeric.loc[mask],
                unit=unit,
                utc=True,
                errors="coerce",
            )
    textual = numeric.isna() & values.notna()
    if textual.any():
        result.loc[textual] = pd.to_datetime(
            values.loc[textual],
            utc=True,
            errors="coerce",
            format="mixed",
        )
    return result


def _removed(values: pd.Series) -> pd.Series:
    lowered = values.fillna("").astype(str).str.strip().str.lower()
    return lowered.isin({"[removed]", "[deleted]", "removed", "deleted"})


def _booleans(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    lowered = values.fillna("").astype(str).str.strip().str.lower()
    return lowered.isin({"1", "true", "yes", "y"})


def _community_from_member(member: str) -> str:
    stem = Path(PurePosixPath(member).name).stem
    for suffix in ("_data", "-data"):
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


def _nested_archive_index(archives: Iterable[Path]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for archive in archives:
        if not archive.is_file():
            continue
        with zipfile.ZipFile(archive) as outer:
            for info in outer.infolist():
                if info.is_dir() or not info.filename.lower().endswith(".zip"):
                    continue
                community = _community_from_member(info.filename)
                key = community.casefold()
                candidate = {
                    "community": community,
                    "outer_archive": str(archive.resolve()),
                    "outer_size": int(archive.stat().st_size),
                    "outer_mtime_ns": int(archive.stat().st_mtime_ns),
                    "member": info.filename,
                    "member_crc": int(info.CRC),
                    "member_size": int(info.file_size),
                    "member_compressed_size": int(info.compress_size),
                }
                if key in index:
                    raise ValueError(f"Duplicate nested Reddit archive for {community}")
                index[key] = candidate
    return index


def _source_archive_records(archives: list[Path], output_dir: Path) -> list[dict[str, Any]]:
    cache_path = output_dir / "source_archives.json"
    cached: dict[str, dict[str, Any]] = {}
    if cache_path.is_file():
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            cached = {str(item["path"]): item for item in payload.get("archives", [])}
        except (OSError, ValueError, KeyError):
            cached = {}

    records: list[dict[str, Any]] = []
    for path in archives:
        resolved = str(path.resolve())
        if not path.is_file():
            records.append({"path": resolved, "status": "missing"})
            continue
        size = int(path.stat().st_size)
        mtime_ns = int(path.stat().st_mtime_ns)
        previous = cached.get(resolved, {})
        digest = previous.get("sha256")
        if previous.get("size") != size or previous.get("mtime_ns") != mtime_ns or not digest:
            digest = sha256_file(path)
        records.append(
            {
                "path": resolved,
                "status": "ready",
                "size": size,
                "mtime_ns": mtime_ns,
                "sha256": digest,
            }
        )
        write_json(cache_path, {"archives": records})
    write_json(cache_path, {"archives": records})
    return records


def _materialize_nested_archive(source: dict[str, Any], cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f"{source['community']}.zip"
    identity_path = target.with_suffix(".source.json")
    expected = {
        key: source[key]
        for key in (
            "outer_archive",
            "outer_size",
            "outer_mtime_ns",
            "member",
            "member_crc",
            "member_size",
        )
    }
    if target.is_file() and identity_path.is_file():
        try:
            identity = json.loads(identity_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            identity = {}
        if identity == expected and target.stat().st_size == source["member_size"]:
            return target

    partial = target.with_suffix(".zip.part")
    partial.unlink(missing_ok=True)
    with zipfile.ZipFile(source["outer_archive"]) as outer:
        with outer.open(source["member"]) as source_handle, partial.open("wb") as target_handle:
            shutil.copyfileobj(source_handle, target_handle, length=8 * 1024 * 1024)
    if partial.stat().st_size != source["member_size"]:
        partial.unlink(missing_ok=True)
        raise ValueError(f"Nested archive size mismatch for {source['community']}")
    partial.replace(target)
    write_json(identity_path, expected)
    return target


def _csv_member(inner: zipfile.ZipFile, prefix: str) -> str:
    matches = [
        name
        for name in inner.namelist()
        if not name.endswith("/")
        and PurePosixPath(name).name.lower().startswith(prefix.lower())
        and name.lower().endswith(".csv")
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one {prefix}*.csv, found {len(matches)}")
    return matches[0]


def _read_community_tables(archive: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    with zipfile.ZipFile(archive) as inner:
        posts_member = _csv_member(inner, "posts_features_")
        comments_member = _csv_member(inner, "comments_data_")
        with inner.open(posts_member) as handle:
            posts = pd.read_csv(handle, encoding="utf-8", encoding_errors="replace", low_memory=False)
        with inner.open(comments_member) as handle:
            comments = pd.read_csv(handle, encoding="utf-8", encoding_errors="replace", low_memory=False)
    return posts, comments, {"posts": posts_member, "comments": comments_member}


def _comment_depths(parent_ids: dict[str, str], post_id: str) -> tuple[dict[str, int], bool]:
    depths: dict[str, int] = {}
    visiting: set[str] = set()

    def visit(comment_id: str) -> int:
        if comment_id in depths:
            return depths[comment_id]
        if comment_id in visiting:
            raise ValueError("cycle")
        visiting.add(comment_id)
        parent = parent_ids[comment_id]
        if parent == f"post:{post_id}":
            depth = 2
        else:
            parent_comment = parent.removeprefix("comment:")
            depth = visit(parent_comment) + 1
        visiting.remove(comment_id)
        depths[comment_id] = depth
        return depth

    try:
        for comment_id in parent_ids:
            visit(comment_id)
    except (KeyError, ValueError, RecursionError):
        return {}, False
    return depths, True


def _normalize_community(
    community: str,
    archive: Path,
    source: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    posts, comments, members = _read_community_tables(archive)
    raw_post_rows = int(len(posts))
    raw_comment_rows = int(len(comments))

    posts = posts.copy()
    posts["_post_id"] = _series(posts, "post_id").map(lambda value: _clean_id(value, "t3_"))
    posts["_created_at"] = _timestamps(_series(posts, "created_utc"))
    duplicate_posts = set(
        posts.loc[posts["_post_id"].duplicated(keep=False), "_post_id"].astype(str)
    )
    invalid_post_ids = set(
        posts.loc[(posts["_post_id"] == "") | posts["_created_at"].isna(), "_post_id"].astype(str)
    )
    invalid_post_ids.update(duplicate_posts)
    posts = posts.loc[~posts["_post_id"].isin(invalid_post_ids)].copy()

    comments = comments.copy()
    comments["_post_id"] = _series(comments, "post_id").map(lambda value: _clean_id(value, "t3_"))
    comments["_comment_id"] = _series(comments, "comment_id").map(lambda value: _clean_id(value, "t1_"))
    comments["_created_at"] = _timestamps(_series(comments, "created_utc"))
    comments["_parent_raw"] = _series(comments, "parent_id").fillna("").astype(str).str.strip()

    known_posts = set(posts["_post_id"])
    comments_for_unknown_posts = int((~comments["_post_id"].isin(known_posts)).sum())
    comments = comments.loc[comments["_post_id"].isin(known_posts)].copy()

    bad_cascades: dict[str, set[str]] = {}

    def reject(post_id: str, reason: str) -> None:
        bad_cascades.setdefault(post_id, set()).add(reason)

    for post_id in invalid_post_ids:
        if post_id:
            reject(post_id, "invalid_or_duplicate_post")

    duplicate_comment_mask = comments["_comment_id"].duplicated(keep=False)
    for post_id in comments.loc[duplicate_comment_mask, "_post_id"].astype(str).unique():
        reject(post_id, "duplicate_comment_id")
    for post_id in comments.loc[
        (comments["_comment_id"] == "") | comments["_created_at"].isna(), "_post_id"
    ].astype(str).unique():
        reject(post_id, "invalid_comment")

    post_times = posts.set_index("_post_id")["_created_at"].to_dict()
    accepted_depths: dict[str, int] = {}
    accepted_parents: dict[str, str] = {}
    original_depth_mismatches = 0

    for post_id, group in comments.groupby("_post_id", sort=False):
        post_id = str(post_id)
        if post_id in bad_cascades:
            continue
        comment_ids = set(group["_comment_id"].astype(str))
        times = dict(zip(group["_comment_id"].astype(str), group["_created_at"], strict=False))
        parents: dict[str, str] = {}
        structural_error = False
        for comment_id_value, raw_parent_value in group[
            ["_comment_id", "_parent_raw"]
        ].itertuples(index=False, name=None):
            comment_id = str(comment_id_value)
            raw_parent = str(raw_parent_value)
            if raw_parent.lower().startswith("t3_"):
                parent_post = _clean_id(raw_parent, "t3_")
                if parent_post != post_id:
                    structural_error = True
                    reject(post_id, "cross_post_root_parent")
                    break
                parent = f"post:{post_id}"
                parent_time = post_times[post_id]
            elif raw_parent.lower().startswith("t1_"):
                parent_comment = _clean_id(raw_parent, "t1_")
                if parent_comment not in comment_ids:
                    structural_error = True
                    reject(post_id, "broken_parent_link")
                    break
                parent = f"comment:{parent_comment}"
                parent_time = times[parent_comment]
            else:
                structural_error = True
                reject(post_id, "invalid_parent_type")
                break
            if times[comment_id] < parent_time:
                structural_error = True
                reject(post_id, "temporal_parent_after_child")
                break
            parents[comment_id] = parent
        if structural_error:
            continue
        depths, acyclic = _comment_depths(parents, post_id)
        if not acyclic:
            reject(post_id, "cycle")
            continue
        accepted_depths.update(depths)
        accepted_parents.update(parents)
        if "depth" in group:
            expected = pd.to_numeric(group["depth"], errors="coerce") + 2
            computed = group["_comment_id"].astype(str).map(depths)
            original_depth_mismatches += int((expected.notna() & (expected != computed)).sum())

    rejected_post_ids = set(bad_cascades)
    valid_posts = posts.loc[~posts["_post_id"].isin(rejected_post_ids)].copy()
    valid_post_ids = set(valid_posts["_post_id"])
    valid_comments = comments.loc[comments["_post_id"].isin(valid_post_ids)].copy()

    post_text = _series(valid_posts, "full_text").fillna("").astype(str)
    post_title = _series(valid_posts, "title").fillna("").astype(str)
    post_events = pd.DataFrame(
        {
            "platform": "reddit",
            "community": community,
            "content_id": valid_posts["_post_id"].astype(str),
            "event_id": "reddit:post:" + valid_posts["_post_id"].astype(str),
            "parent_event_id": "",
            "created_at": valid_posts["_created_at"],
            "depth": 1,
            "author_id": _series(valid_posts, "author").fillna("").astype(str),
            "text": post_text.where(post_text.str.strip() != "", post_title),
            "title": post_title,
            "score": pd.to_numeric(_series(valid_posts, "final_score", 0), errors="coerce").fillna(0.0),
            "event_type": "post",
            "removed": False,
            "content_url": _series(valid_posts, "url").fillna("").astype(str),
            "source_id": source["member"],
            "is_viral": _booleans(_series(valid_posts, "is_viral", False)),
        }
    )

    comment_text = _series(valid_comments, "comment_text").fillna("").astype(str)
    comment_ids = valid_comments["_comment_id"].astype(str)
    parent_ids = comment_ids.map(accepted_parents)
    comment_events = pd.DataFrame(
        {
            "platform": "reddit",
            "community": community,
            "content_id": valid_comments["_post_id"].astype(str),
            "event_id": "reddit:comment:" + comment_ids,
            "parent_event_id": parent_ids.map(
                lambda value: (
                    f"reddit:post:{value.removeprefix('post:')}"
                    if str(value).startswith("post:")
                    else f"reddit:comment:{str(value).removeprefix('comment:')}"
                )
            ),
            "created_at": valid_comments["_created_at"],
            "depth": comment_ids.map(accepted_depths).astype(int),
            "author_id": _series(valid_comments, "author").fillna("").astype(str),
            "text": comment_text,
            "title": "",
            "score": pd.to_numeric(_series(valid_comments, "score", 0), errors="coerce").fillna(0.0),
            "event_type": "comment",
            "removed": _removed(comment_text),
            "content_url": "",
            "source_id": source["member"],
            "is_viral": False,
        }
    )
    viral_by_post = post_events.set_index("content_id")["is_viral"].to_dict()
    comment_events["is_viral"] = comment_events["content_id"].map(viral_by_post).fillna(False).astype(bool)

    events = pd.concat([post_events, comment_events], ignore_index=True)
    events = events[EVENT_COLUMNS].sort_values(
        ["created_at", "event_type", "event_id"],
        kind="stable",
    )
    event_audit = validate_platform_events(
        events.rename(columns={"community": "community_id", "content_url": "url"})
    )
    reason_counts = Counter(
        reason
        for reasons in bad_cascades.values()
        for reason in reasons
    )
    manifest = {
        **event_audit,
        "community": community,
        "normalizer_version": NORMALIZER_VERSION,
        "status": "ready" if event_audit["claim_allowed"] else "invalid",
        "source": source,
        "inner_members": members,
        "raw_posts": raw_post_rows,
        "raw_comments": raw_comment_rows,
        "complete_cascades": int(len(valid_posts)),
        "comments_retained": int(len(valid_comments)),
        "cascades_excluded": int(len(rejected_post_ids)),
        "comments_for_unknown_posts": comments_for_unknown_posts,
        "original_depth_mismatches": original_depth_mismatches,
        "exclusion_reasons": dict(sorted(reason_counts.items())),
    }
    return events, manifest


def _cached_community(
    output_path: Path,
    manifest_path: Path,
    source: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    if not output_path.is_file() or not manifest_path.is_file():
        return False, {}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False, {}
    source_keys = (
        "outer_archive",
        "outer_size",
        "outer_mtime_ns",
        "member",
        "member_crc",
        "member_size",
    )
    identity_matches = all(
        manifest.get("source", {}).get(key) == source.get(key)
        for key in source_keys
    )
    return bool(
        identity_matches
        and manifest.get("status") == "ready"
        and manifest.get("normalizer_version") == NORMALIZER_VERSION
    ), manifest


def _combine_parquets(paths: list[Path], target: Path) -> None:
    partial = target.with_suffix(".parquet.part")
    partial.unlink(missing_ok=True)
    writer: pq.ParquetWriter | None = None
    try:
        for path in paths:
            table = pq.read_table(path)
            if writer is None:
                writer = pq.ParquetWriter(partial, table.schema, compression="snappy")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise ValueError("No community Parquet files were available")
    partial.replace(target)


def import_reddit_archive_bundles(
    archives: Iterable[Path],
    output_dir: Path,
    communities: Iterable[str],
    minimum_communities: int = 20,
    minimum_cascades: int = 10_000,
    community_groups: dict[str, Iterable[str]] | None = None,
) -> dict[str, Any]:
    """Import nested community archives with cascade-level structural auditing."""
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_paths = [Path(path) for path in archives]
    selected = list(dict.fromkeys(str(value).strip() for value in communities if str(value).strip()))
    groups = {
        group: list(values)
        for group, values in (community_groups or {}).items()
    }
    selection_manifest = {
        "selection_protocol": "outcome_blind_stratified_fixed_list",
        "selected_communities": selected,
        "community_groups": groups,
        "minimum_communities": int(minimum_communities),
        "minimum_cascades": int(minimum_cascades),
        "uses_images": False,
        "uses_author_history": False,
    }
    write_json(output_dir / "selection_manifest.json", selection_manifest)

    archive_records = _source_archive_records(archive_paths, output_dir)
    index = _nested_archive_index(archive_paths)
    missing = [community for community in selected if community.casefold() not in index]
    if missing:
        result = {
            "status": "needs_data",
            "claim_allowed": False,
            "target_met": False,
            "missing_communities": missing,
            "selected_communities": selected,
            "sources": archive_records,
            "n_communities": 0,
            "n_cascades": 0,
            "n_events": 0,
        }
        write_json(output_dir / "expansion_manifest.json", result)
        return result

    cache_dir = output_dir / "archive_cache"
    community_dir = output_dir / "communities"
    community_dir.mkdir(parents=True, exist_ok=True)
    community_manifests: list[dict[str, Any]] = []
    community_paths: list[Path] = []

    for community in selected:
        source = dict(index[community.casefold()])
        source["community"] = community
        output_path = community_dir / f"{community}.parquet"
        manifest_path = community_dir / f"{community}.manifest.json"
        cached, manifest = _cached_community(output_path, manifest_path, source)
        if not cached:
            inner_archive = _materialize_nested_archive(source, cache_dir)
            events, manifest = _normalize_community(community, inner_archive, source)
            partial = output_path.with_suffix(".parquet.part")
            events.to_parquet(partial, index=False)
            partial.replace(output_path)
            manifest["parquet_sha256"] = sha256_file(output_path)
            write_json(manifest_path, manifest)
        community_paths.append(output_path)
        community_manifests.append(manifest)
        write_json(
            output_dir / "progress.json",
            {
                "status": "running",
                "completed": len(community_manifests),
                "total": len(selected),
                "last_community": community,
                "complete_cascades_so_far": sum(
                    int(item.get("complete_cascades", 0))
                    for item in community_manifests
                ),
            },
        )

    combined_path = output_dir / "reddit_expanded_events.parquet"
    _combine_parquets(community_paths, combined_path)
    ready = [item for item in community_manifests if item.get("status") == "ready"]
    n_communities = len(ready)
    n_cascades = sum(int(item.get("complete_cascades", 0)) for item in ready)
    n_events = sum(int(item.get("n_events", 0)) for item in ready)
    target_met = (
        n_communities >= int(minimum_communities)
        and n_cascades >= int(minimum_cascades)
        and len(ready) == len(selected)
    )
    result = {
        "status": "complete" if target_met else "insufficient",
        "claim_allowed": target_met,
        "target_met": target_met,
        "selection_protocol": selection_manifest["selection_protocol"],
        "selected_communities": selected,
        "community_groups": groups,
        "n_communities": n_communities,
        "n_cascades": n_cascades,
        "n_events": n_events,
        "cascades_excluded": sum(
            int(item.get("cascades_excluded", 0))
            for item in community_manifests
        ),
        "minimum_communities": int(minimum_communities),
        "minimum_cascades": int(minimum_cascades),
        "sources": archive_records,
        "community_manifests": community_manifests,
        "output": {
            "path": str(combined_path.resolve()),
            "sha256": sha256_file(combined_path),
        },
    }
    write_json(output_dir / "expansion_manifest.json", result)
    write_json(
        output_dir / "progress.json",
        {
            "status": result["status"],
            "completed": len(selected),
            "total": len(selected),
            "complete_cascades_so_far": n_cascades,
        },
    )
    return result
