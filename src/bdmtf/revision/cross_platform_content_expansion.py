from __future__ import annotations

import hashlib
import json
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from bdmtf.revision.external_sources import canonicalize_url
from bdmtf.revision.hackernews import (
    DEFAULT_BASE_URL,
    DEFAULT_SEARCH_URL,
    _fetch_item,
    _iso_timestamp,
    _plain_text,
)
from bdmtf.revision.lemmy_content_matched_validation import (
    _load_resolution_records,
    build_exact_url_pairs,
)
from bdmtf.revision.lemmy_outcomes import collect_post_thread
from bdmtf.revision.provenance import sha256_file, write_json


def _request_json(url: str, params: Mapping[str, Any], retries: int = 5) -> Any:
    last_error: Exception | None = None
    headers = {"User-Agent": "bdmtf-cross-platform-expansion/1.0"}
    for attempt in range(retries):
        try:
            response = requests.get(
                url,
                params=dict(params),
                headers=headers,
                timeout=60,
            )
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            time.sleep(0.5 * (2**attempt))
    raise RuntimeError(f"Request failed after {retries} attempts: {url}") from last_error


def _fetch_hn_tree(
    story_id: str,
    cache_dir: Path,
    tree_url: str,
) -> Mapping[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{story_id}.json"
    if cache_path.is_file():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    payload = _request_json(f"{tree_url.rstrip('/')}/{story_id}", {})
    temporary = cache_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(cache_path)
    return payload


def _flatten_hn_comments(root: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    comments: list[Mapping[str, Any]] = []
    frontier = list(root.get("children") or [])
    while frontier:
        item = frontier.pop()
        if not isinstance(item, Mapping):
            continue
        comments.append(item)
        frontier.extend(item.get("children") or [])
    return comments


def discover_hackernews_roots(
    target: int,
    max_comments: int,
    cache_dir: Path,
    search_url: str = DEFAULT_SEARCH_URL,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Discover HN roots without downloading every reply tree."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    records: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    before_timestamp: int | None = None
    page = 0
    while len(records) < int(target):
        cache_path = cache_dir / f"discovery-{page:04d}.json"
        numeric_filters = [
            "num_comments>0",
            f"num_comments<={int(max_comments)}",
        ]
        if before_timestamp is not None:
            numeric_filters.append(f"created_at_i<{before_timestamp}")
        params = {
            "tags": "story",
            "hitsPerPage": 1000,
            "page": 0,
            "numericFilters": ",".join(numeric_filters),
        }
        if cache_path.is_file():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            payload = _request_json(search_url, params)
            temporary = cache_path.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False),
                encoding="utf-8",
            )
            temporary.replace(cache_path)
        hits = payload.get("hits", []) if isinstance(payload, dict) else []
        timestamps: list[int] = []
        for hit in hits:
            try:
                content_id = str(int(hit["objectID"]))
                created_at = int(hit["created_at_i"])
            except (KeyError, TypeError, ValueError):
                continue
            timestamps.append(created_at)
            if content_id in seen:
                continue
            seen.add(content_id)
            records.append(
                {
                    "platform": "HackerNews",
                    "community": "historical",
                    "content_id": content_id,
                    "event_id": content_id,
                    "parent_event_id": "",
                    "created_at": _iso_timestamp(created_at),
                    "author_id": str(hit.get("author") or ""),
                    "text": _plain_text(
                        f"{hit.get('title') or ''} {hit.get('story_text') or ''}"
                    ),
                    "score": float(hit.get("points") or 0),
                    "event_type": "post",
                    "removed": False,
                    "content_url": str(
                        hit.get("url")
                        or f"https://news.ycombinator.com/item?id={content_id}"
                    ),
                    "reported_comments": int(hit.get("num_comments") or 0),
                }
            )
            if len(records) >= int(target):
                break
        oldest = min(timestamps) if timestamps else None
        audit.append(
            {
                "page": page,
                "before_timestamp": before_timestamp,
                "hits": len(hits),
                "oldest_timestamp": oldest,
                "cumulative_unique_roots": len(records),
            }
        )
        if oldest is None or (
            before_timestamp is not None and oldest >= before_timestamp
        ):
            break
        before_timestamp = oldest
        page += 1
    frame = pd.DataFrame(records).drop_duplicates("content_id")
    return frame.head(int(target)).reset_index(drop=True), audit


def _collect_hn_story(
    story_id: str,
    item_cache_dir: Path,
    tree_cache_dir: Path,
    max_comments: int,
    base_url: str,
    tree_url: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = _fetch_item(int(story_id), item_cache_dir, base_url)
    if not root or root.get("type") != "story" or "time" not in root:
        return [], {"content_id": story_id, "complete": False, "reason": "missing_root"}
    reported = int(root.get("descendants") or 0)
    if reported > int(max_comments):
        return [], {
            "content_id": story_id,
            "complete": False,
            "reason": "reported_comment_limit",
            "reported_comments": reported,
        }
    try:
        tree = _fetch_hn_tree(story_id, tree_cache_dir, tree_url)
    except RuntimeError:
        return [], {
            "content_id": story_id,
            "complete": False,
            "reason": "tree_request_failed",
            "reported_comments": reported,
        }
    if str(tree.get("id") or "") != story_id or tree.get("type") != "story":
        return [], {
            "content_id": story_id,
            "complete": False,
            "reason": "invalid_tree_root",
            "reported_comments": reported,
        }
    content_url = str(
        root.get("url")
        or f"https://news.ycombinator.com/item?id={story_id}"
    )
    events = [
        {
            "platform": "HackerNews",
            "community": "historical_matched",
            "content_id": story_id,
            "event_id": story_id,
            "parent_event_id": "",
            "created_at": _iso_timestamp(root["time"]),
            "author_id": str(root.get("by") or ""),
            "text": _plain_text(
                f"{tree.get('title') or root.get('title') or ''} "
                f"{tree.get('text') or root.get('text') or ''}"
            ),
            "score": float(root.get("score") or 0),
            "event_type": "post",
            "removed": bool(root.get("deleted") or root.get("dead")),
            "content_url": content_url,
        }
    ]
    seen: set[str] = set()
    for item in _flatten_hn_comments(tree):
        item_id = str(item.get("id") or "")
        if not item_id or item_id in seen or item.get("created_at_i") is None:
            continue
        seen.add(item_id)
        events.append(
            {
                "platform": "HackerNews",
                "community": "historical_matched",
                "content_id": story_id,
                "event_id": item_id,
                "parent_event_id": str(item.get("parent_id") or story_id),
                "created_at": _iso_timestamp(item["created_at_i"]),
                "author_id": str(item.get("author") or ""),
                "text": _plain_text(str(item.get("text") or "")),
                "score": 0.0,
                "event_type": "comment",
                "removed": bool(
                    not item.get("author") or item.get("text") is None
                ),
                "content_url": content_url,
            }
        )
    observed = len(events) - 1
    timestamps = {
        str(event["event_id"]): pd.Timestamp(event["created_at"])
        for event in events
    }
    temporal_violations = sum(
        1
        for event in events[1:]
        if str(event["parent_event_id"]) in timestamps
        and timestamps[str(event["parent_event_id"])]
        > timestamps[str(event["event_id"])]
    )
    complete = observed == reported and temporal_violations == 0
    return (
        events if complete else [],
        {
            "content_id": story_id,
            "complete": complete,
            "reported_comments": reported,
            "observed_comments": observed,
            "temporal_order_violations": temporal_violations,
            "reason": "" if complete else "incomplete_tree",
        },
    )


def collect_hackernews_threads(
    story_ids: Iterable[str],
    item_cache_dir: Path,
    tree_cache_dir: Path,
    workers: int,
    max_comments: int,
    base_url: str = DEFAULT_BASE_URL,
    tree_url: str = "https://hn.algolia.com/api/v1/items",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    item_cache_dir.mkdir(parents=True, exist_ok=True)
    tree_cache_dir.mkdir(parents=True, exist_ok=True)
    events: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    ids = sorted(set(map(str, story_ids)), key=int)
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
        futures = {
            executor.submit(
                _collect_hn_story,
                story_id,
                item_cache_dir,
                tree_cache_dir,
                max_comments,
                base_url,
                tree_url,
            ): story_id
            for story_id in ids
        }
        for future in as_completed(futures):
            story_events, audit = future.result()
            events.extend(story_events)
            audits.append(audit)
    return pd.DataFrame(events), pd.DataFrame(audits)


def _lemmy_root_event(post: Mapping[str, Any]) -> dict[str, Any]:
    post_id = str(post["content_id"])
    return {
        "platform": "lemmy",
        "community": str(post.get("community") or "unknown"),
        "community_id": str(post.get("community_id") or "unknown"),
        "content_id": post_id,
        "event_id": post_id,
        "parent_event_id": "",
        "created_at": str(post["published"]),
        "depth": 0,
        "author_id": str(post.get("author_id") or ""),
        "text": str(post.get("title") or ""),
        "event_type": "post",
        "removed": bool(post.get("removed")),
        "content_url": str(post.get("url") or ""),
    }


def collect_lemmy_threads(
    posts: pd.DataFrame,
    post_ids: Iterable[str],
    base_url: str,
    cache_dir: Path,
    workers: int,
    pages: int,
    page_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    post_index = posts.copy()
    post_index["content_id"] = post_index["content_id"].astype(str)
    post_index = post_index.set_index("content_id", drop=False)
    ids = sorted(set(map(str, post_ids)), key=int)

    def collect_one(post_id: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        row = post_index.loc[post_id]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        comments, observed_post = collect_post_thread(
            base_url,
            post_id,
            cache_dir,
            pages=pages,
            page_size=page_size,
        )
        post = observed_post or row.to_dict()
        root = _lemmy_root_event(post)
        event_ids = {str(item["event_id"]) for item in comments}
        broken = sum(
            1
            for item in comments
            if str(item["parent_event_id"]) != post_id
            and str(item["parent_event_id"]) not in event_ids
        )
        reported = int(row.get("reported_comments") or 0)
        observed = len(comments)
        complete = broken == 0 and observed >= max(0, reported - 2)
        return (
            [root, *comments] if complete else [],
            {
                "content_id": post_id,
                "complete": complete,
                "reported_comments": reported,
                "observed_comments": observed,
                "broken_parent_links": broken,
                "reason": "" if complete else "incomplete_tree",
            },
        )

    events: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
        futures = {executor.submit(collect_one, post_id): post_id for post_id in ids}
        for future in as_completed(futures):
            post_events, audit = future.result()
            events.extend(post_events)
            audits.append(audit)
    return pd.DataFrame(events), pd.DataFrame(audits)


def _merge_events(*frames: pd.DataFrame) -> pd.DataFrame:
    available = [frame for frame in frames if not frame.empty]
    if not available:
        return pd.DataFrame()
    result = pd.concat(available, ignore_index=True, sort=False)
    result["content_id"] = result["content_id"].astype(str)
    result["event_id"] = result["event_id"].astype(str)
    return result.drop_duplicates(["platform", "content_id", "event_id"])


def _bootstrap_interval(
    values: np.ndarray,
    samples: int,
    seed: int,
) -> tuple[float, float, float]:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    rng = np.random.default_rng(seed)
    draws = [
        float(rng.choice(clean, size=len(clean), replace=True).mean())
        for _ in range(int(samples))
    ]
    return (
        float(clean.mean()),
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    )


def analyze_voat_author_content_transfer(
    archive_path: Path,
    output_dir: Path,
    minimum_messages: int = 2,
    permutations: int = 1000,
    bootstrap_samples: int = 2000,
    seed: int = 30371,
) -> dict[str, Any]:
    """Compare Reddit-before and Voat-after content for the same pseudonymous users."""
    documents: dict[tuple[str, str, str], list[str]] = {}
    counts: dict[tuple[str, str, str], int] = {}
    with zipfile.ZipFile(archive_path) as archive:
        members = [
            name
            for name in archive.namelist()
            if not name.endswith("/")
            and "__MACOSX" not in name
            and ".DS_Store" not in name
        ]
        for member in members:
            intervention = (
                "fatpeoplehate"
                if "fatpeoplehate" in member.lower()
                else "greatawakening"
            )
            platform = "reddit_before" if "in-before" in member.lower() else "voat_after"
            with archive.open(member) as handle:
                for raw in handle:
                    try:
                        record = json.loads(raw)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        continue
                    author = str(record.get("author") or "")
                    text = str(record.get("selftext") or "").strip()
                    if not author or not text:
                        continue
                    key = (intervention, platform, author)
                    documents.setdefault(key, []).append(text)
                    counts[key] = counts.get(key, 0) + 1

    pair_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    for intervention in ("fatpeoplehate", "greatawakening"):
        reddit_authors = {
            author
            for (name, platform, author), count in counts.items()
            if name == intervention
            and platform == "reddit_before"
            and count >= int(minimum_messages)
        }
        voat_authors = {
            author
            for (name, platform, author), count in counts.items()
            if name == intervention
            and platform == "voat_after"
            and count >= int(minimum_messages)
        }
        shared = sorted(reddit_authors & voat_authors)
        if len(shared) < 10:
            continue
        reddit_docs = [
            " ".join(documents[(intervention, "reddit_before", author)])
            for author in shared
        ]
        voat_docs = [
            " ".join(documents[(intervention, "voat_after", author)])
            for author in shared
        ]
        matrix = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_features=30000,
        ).fit_transform([*reddit_docs, *voat_docs])
        matrix = normalize(matrix)
        reddit_matrix = matrix[: len(shared)]
        voat_matrix = matrix[len(shared) :]
        matched = np.asarray(
            reddit_matrix.multiply(voat_matrix).sum(axis=1)
        ).reshape(-1)
        permutation = rng.permutation(len(shared))
        random_similarity = np.asarray(
            reddit_matrix.multiply(voat_matrix[permutation]).sum(axis=1)
        ).reshape(-1)
        for index, author in enumerate(shared):
            pair_rows.append(
                {
                    "intervention": intervention,
                    "author_hash": hashlib.sha256(
                        f"bdmtf-voat|{author}".encode("utf-8")
                    ).hexdigest()[:20],
                    "reddit_messages": counts[
                        (intervention, "reddit_before", author)
                    ],
                    "voat_messages": counts[
                        (intervention, "voat_after", author)
                    ],
                    "matched_similarity": float(matched[index]),
                    "random_similarity": float(random_similarity[index]),
                    "similarity_gain": float(
                        matched[index] - random_similarity[index]
                    ),
                }
            )
    pairs = pd.DataFrame(pair_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(output_dir / "voat_author_content_pairs.csv", index=False)
    summaries: list[dict[str, Any]] = []
    for intervention, group in pairs.groupby("intervention", sort=False):
        gain = _bootstrap_interval(
            group["similarity_gain"].to_numpy(float),
            bootstrap_samples,
            seed + len(summaries),
        )
        observed = float(group["matched_similarity"].mean())
        combined = group[["matched_similarity", "random_similarity"]].to_numpy()
        permutation_means: list[float] = []
        for _ in range(int(permutations)):
            swap = rng.random(len(combined)) < 0.5
            signed = combined[:, 0] - combined[:, 1]
            signed[swap] *= -1
            permutation_means.append(float(signed.mean()))
        p_value = (1 + sum(value >= gain[0] for value in permutation_means)) / (
            1 + len(permutation_means)
        )
        summaries.append(
            {
                "intervention": intervention,
                "matched_authors": int(len(group)),
                "matched_similarity_mean": observed,
                "random_similarity_mean": float(
                    group["random_similarity"].mean()
                ),
                "similarity_gain": gain[0],
                "ci_low": gain[1],
                "ci_high": gain[2],
                "permutation_p_value": float(p_value),
            }
        )
    summary = pd.DataFrame(summaries)
    summary.to_csv(output_dir / "voat_content_transfer_summary.csv", index=False)
    result = {
        "status": "complete",
        "design": "same-pseudonymous-author Reddit-before to Voat-after content-profile comparison",
        "matched_authors": int(len(pairs)),
        "interventions": summaries,
        "claim_boundary": (
            "This tests author-level semantic continuity after migration; it is not "
            "an exact-story match and does not identify a platform causal effect."
        ),
        "source": {
            "path": archive_path.as_posix(),
            "sha256": sha256_file(archive_path),
        },
    }
    write_json(output_dir / "voat_content_transfer_manifest.json", result)
    return result


def run_cross_platform_content_expansion(
    root: str | Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    project = Path(root)
    output = project / str(config["output_dir"])
    data_output = project / str(config["data_output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    data_output.mkdir(parents=True, exist_ok=True)

    inputs = {
        name: project / str(relative)
        for name, relative in config["inputs"].items()
    }
    existing_hn = pd.read_parquet(inputs["hackernews_events"])
    lemmy_posts = pd.read_parquet(inputs["lemmy_posts"])
    existing_lemmy = pd.read_parquet(inputs["lemmy_events"])
    records = _load_resolution_records(inputs["url_resolution_cache"])

    discovery = config["hackernews_discovery"]
    roots, discovery_audit = discover_hackernews_roots(
        int(discovery["candidate_stories"]),
        int(discovery["max_comments_per_story"]),
        data_output / "hn_discovery_cache",
        str(discovery.get("search_url", DEFAULT_SEARCH_URL)),
    )
    existing_roots = existing_hn[
        existing_hn["event_type"].astype(str).str.lower().isin({"post", "root"})
    ].copy()
    candidate_roots = _merge_events(existing_roots, roots)
    pairs = build_exact_url_pairs(
        candidate_roots,
        lemmy_posts,
        records,
        float(config["matching"]["max_hours"]),
    )
    pairs = pairs[
        pd.to_numeric(pairs["reported_comments"], errors="coerce")
        <= int(config["lemmy_collection"]["max_reported_comments"])
    ].copy()
    pairs = pairs.sort_values(
        ["time_distance_hours", "created_at", "match_id"]
    ).head(int(config["matching"]["target_exact_pairs"]))
    pairs.to_csv(output / "target_exact_url_pairs.csv", index=False)

    existing_hn_ids = set(existing_roots["content_id"].astype(str))
    new_hn_ids = set(pairs["hn_content_id"].astype(str)) - existing_hn_ids
    hn_events, hn_audit = collect_hackernews_threads(
        new_hn_ids,
        data_output / "hn_item_cache",
        data_output / "hn_tree_cache",
        int(config["hackernews_collection"]["workers"]),
        int(discovery["max_comments_per_story"]),
        str(config["hackernews_collection"].get("base_url", DEFAULT_BASE_URL)),
        str(
            config["hackernews_collection"].get(
                "tree_url",
                "https://hn.algolia.com/api/v1/items",
            )
        ),
    )
    hn_audit.to_csv(output / "hackernews_thread_audit.csv", index=False)
    complete_hn_ids = set(
        hn_audit.loc[hn_audit["complete"].astype(bool), "content_id"].astype(str)
    ) | existing_hn_ids

    existing_lemmy_ids = set(
        existing_lemmy.loc[
            existing_lemmy["event_type"].astype(str).str.lower().isin(
                {"post", "root"}
            ),
            "content_id",
        ].astype(str)
    )
    requested_lemmy_ids = set(pairs["lemmy_content_id"].astype(str))
    new_lemmy_ids = requested_lemmy_ids - existing_lemmy_ids
    lemmy_events, lemmy_audit = collect_lemmy_threads(
        lemmy_posts,
        new_lemmy_ids,
        str(config["lemmy_collection"]["base_url"]),
        data_output / "lemmy_thread_cache",
        int(config["lemmy_collection"]["workers"]),
        int(config["lemmy_collection"]["pages"]),
        int(config["lemmy_collection"]["page_size"]),
    )
    lemmy_audit.to_csv(output / "lemmy_thread_audit.csv", index=False)
    complete_lemmy_ids = set(
        lemmy_audit.loc[
            lemmy_audit["complete"].astype(bool), "content_id"
        ].astype(str)
    ) | existing_lemmy_ids

    usable_pairs = pairs[
        pairs["hn_content_id"].astype(str).isin(complete_hn_ids)
        & pairs["lemmy_content_id"].astype(str).isin(complete_lemmy_ids)
    ].copy()
    usable_pairs.to_csv(output / "complete_exact_url_pairs.csv", index=False)

    merged_hn = _merge_events(existing_hn, hn_events)
    merged_lemmy = _merge_events(existing_lemmy, lemmy_events)
    hn_path = data_output / "hackernews_events.parquet"
    lemmy_path = data_output / "lemmy_events.parquet"
    posts_path = data_output / "lemmy_posts.parquet"
    merged_hn.to_parquet(hn_path, index=False)
    merged_lemmy.to_parquet(lemmy_path, index=False)
    lemmy_posts.to_parquet(posts_path, index=False)

    voat_result = analyze_voat_author_content_transfer(
        inputs["tbbt_migration_archive"],
        output,
        int(config["voat"]["minimum_messages_per_platform"]),
        int(config["voat"]["permutations"]),
        int(config["bootstrap_samples"]),
        int(config["seed"]),
    )
    result = {
        "status": "complete",
        "protocol": config.get("protocol", {}),
        "hackernews": {
            "candidate_roots": int(len(candidate_roots)),
            "new_roots_discovered": int(len(roots)),
            "targeted_threads": int(len(new_hn_ids)),
            "complete_new_threads": int(
                hn_audit["complete"].astype(bool).sum()
            )
            if not hn_audit.empty
            else 0,
            "merged_complete_roots": int(
                merged_hn["content_id"].astype(str).nunique()
            ),
            "discovery_queries": discovery_audit,
        },
        "lemmy": {
            "candidate_posts": int(len(lemmy_posts)),
            "targeted_threads": int(len(new_lemmy_ids)),
            "complete_new_threads": int(
                lemmy_audit["complete"].astype(bool).sum()
            )
            if not lemmy_audit.empty
            else 0,
            "merged_complete_roots": int(
                merged_lemmy["content_id"].astype(str).nunique()
            ),
        },
        "matching": {
            "target_exact_pairs": int(len(pairs)),
            "complete_exact_pairs": int(len(usable_pairs)),
            "unique_canonical_urls": int(
                usable_pairs["canonical_url"].nunique()
            ),
            "minimum_complete_pairs": int(
                config["matching"]["minimum_complete_pairs"]
            ),
        },
        "voat": voat_result,
        "claim_allowed": bool(
            len(usable_pairs)
            >= int(config["matching"]["minimum_complete_pairs"])
        ),
        "outputs": {
            "hackernews_events": {
                "path": hn_path.as_posix(),
                "sha256": sha256_file(hn_path),
            },
            "lemmy_posts": {
                "path": posts_path.as_posix(),
                "sha256": sha256_file(posts_path),
            },
            "lemmy_events": {
                "path": lemmy_path.as_posix(),
                "sha256": sha256_file(lemmy_path),
            },
        },
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output / "expansion_manifest.json", result)
    return result
