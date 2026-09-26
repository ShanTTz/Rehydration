from __future__ import annotations

import html
import json
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from bdmtf.revision.provenance import sha256_file, write_json


DEFAULT_BASE_URL = "https://hacker-news.firebaseio.com/v0"
DEFAULT_SEARCH_URL = "https://hn.algolia.com/api/v1/search_by_date"


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def _plain_text(value: str) -> str:
    parser = _TextExtractor()
    parser.feed(value or "")
    return " ".join(html.unescape(" ".join(parser.parts)).split())


def _request_json(url: str, retries: int = 4) -> Any:
    headers = {"User-Agent": "bdmtf-external-validity-research/0.3"}
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            time.sleep(0.5 * (2**attempt))
    raise RuntimeError(f"Hacker News API request failed after {retries} attempts: {url}") from last_error


def _fetch_item(item_id: int, cache_dir: Path, base_url: str) -> dict[str, Any] | None:
    cache_path = cache_dir / f"{item_id}.json"
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            cache_path.unlink(missing_ok=True)
    item = _request_json(f"{base_url}/item/{item_id}.json")
    if item is not None:
        temporary_path = cache_path.with_suffix(".json.tmp")
        temporary_path.write_text(json.dumps(item, ensure_ascii=False), encoding="utf-8")
        temporary_path.replace(cache_path)
    return item


def _iso_timestamp(value: Any) -> str:
    return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat()


def _discover_story_ids_algolia(
    target: int,
    max_comments_per_story: int,
    search_url: str = DEFAULT_SEARCH_URL,
) -> tuple[list[int], dict[str, Any]]:
    """Use the public HN Search index only to discover historical story IDs."""
    ids: list[int] = []
    seen: set[int] = set()
    before_timestamp: int | None = None
    queries: list[dict[str, Any]] = []
    while len(ids) < target:
        numeric_filters = [
            "num_comments>0",
            f"num_comments<={int(max_comments_per_story)}",
        ]
        if before_timestamp is not None:
            numeric_filters.append(f"created_at_i<{before_timestamp}")
        response = _request_json(
            search_url
            + "?tags=story"
            + "&hitsPerPage=1000"
            + "&page=0"
            + "&numericFilters="
            + requests.utils.quote(",".join(numeric_filters), safe=",<>=")
        )
        hits = response.get("hits", []) if isinstance(response, dict) else []
        timestamps = [
            int(hit["created_at_i"])
            for hit in hits
            if str(hit.get("created_at_i", "")).isdigit()
        ]
        queries.append(
            {
                "before_timestamp": before_timestamp,
                "hits": len(hits),
                "oldest_timestamp": min(timestamps) if timestamps else None,
            }
        )
        for hit in hits:
            try:
                story_id = int(hit["objectID"])
            except (KeyError, TypeError, ValueError):
                continue
            if story_id not in seen:
                seen.add(story_id)
                ids.append(story_id)
                if len(ids) >= target:
                    break
        if len(ids) >= target or not timestamps:
            break
        next_before = min(timestamps)
        if before_timestamp is not None and next_before >= before_timestamp:
            break
        before_timestamp = next_before
    return ids, {
        "provider": "HN Search powered by Algolia",
        "endpoint": search_url,
        "queries": queries,
        "query_count": len(queries),
        "ids_discovered": len(ids),
        "filter": f"story with 1..{int(max_comments_per_story)} indexed comments",
        "content_source": "Official Hacker News Firebase item API",
    }


def collect_hackernews(
    output_dir: Path,
    story_count: int = 50,
    max_comments_per_story: int = 300,
    feed: str = "topstories",
    workers: int = 8,
    base_url: str = DEFAULT_BASE_URL,
    historical_scan_multiplier: int = 8,
    discovery: str = "auto",
    search_url: str = DEFAULT_SEARCH_URL,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Collect a bounded, cached snapshot from the official read-only HN API."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    request_failures: list[dict[str, Any]] = []

    def fetch_item(item_id: int) -> dict[str, Any] | None:
        try:
            return _fetch_item(item_id, cache_dir, base_url)
        except RuntimeError as exc:
            request_failures.append({"item_id": int(item_id), "reason": str(exc)})
            return None

    feed_ids = _request_json(f"{base_url}/{feed}.json")
    if not isinstance(feed_ids, list):
        raise RuntimeError(f"Unexpected Hacker News {feed} response")

    candidate_ids = [int(value) for value in feed_ids]
    selection_mode = "bounded_feed"
    discovery_audit: dict[str, Any] = {
        "provider": "Official Hacker News Firebase feed",
        "content_source": "Official Hacker News Firebase item API",
    }
    if story_count > len(candidate_ids):
        active_discovery = "algolia" if discovery == "auto" else discovery
        if active_discovery == "algolia":
            selection_mode = "algolia_historical_discovery"
            reserve_target = max(story_count, (story_count * 11 + 9) // 10)
            candidate_ids, discovery_audit = _discover_story_ids_algolia(
                reserve_target,
                max_comments_per_story,
                search_url,
            )
        elif active_discovery == "official_maxitem":
            selection_mode = "official_maxitem_reverse_scan"
            max_item = int(_request_json(f"{base_url}/maxitem.json"))
            feed_set = set(candidate_ids)
            scan_limit = max(story_count * historical_scan_multiplier, story_count)
            candidate_ids.extend(
                item_id
                for item_id in range(max_item, max(0, max_item - scan_limit), -1)
                if item_id not in feed_set
            )
            discovery_audit = {
                "provider": "Official Hacker News Firebase maxitem reverse scan",
                "max_item": max_item,
                "scan_limit": scan_limit,
                "content_source": "Official Hacker News Firebase item API",
            }
        else:
            raise ValueError("discovery must be auto, algolia, or official_maxitem")

    reserve_target = max(story_count, (story_count * 11 + 9) // 10)
    stories: list[dict[str, Any]] = []
    batch_size = max(workers * 20, 100)
    for offset in range(0, len(candidate_ids), batch_size):
        batch_ids = candidate_ids[offset : offset + batch_size]
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            candidates = list(executor.map(fetch_item, batch_ids))
        stories.extend(
            item
            for item in candidates
            if item
            and item.get("type") == "story"
            and 0 < int(item.get("descendants", 0)) <= max_comments_per_story
        )
        if len(stories) >= reserve_target:
            break
    stories = stories[:reserve_target]
    records: list[dict[str, Any]] = []
    frontier: deque[tuple[int, int]] = deque()
    stories_by_id: dict[int, dict[str, Any]] = {}
    seen_by_story: dict[int, set[int]] = {}
    collected_by_story: dict[int, int] = {}
    for story in stories:
        story_id = int(story["id"])
        stories_by_id[story_id] = story
        seen_by_story[story_id] = set()
        collected_by_story[story_id] = 0
        community = str(feed)
        content_url = str(story.get("url") or f"https://news.ycombinator.com/item?id={story_id}")
        records.append(
            {
                "platform": "HackerNews",
                "community": community,
                "content_id": str(story_id),
                "event_id": str(story_id),
                "parent_event_id": "",
                "created_at": _iso_timestamp(story["time"]),
                "author_id": str(story.get("by", "")),
                "text": _plain_text(f"{story.get('title', '')} {story.get('text', '')}"),
                "score": float(story.get("score", 0)),
                "event_type": "post",
                "removed": bool(story.get("deleted") or story.get("dead")),
                "content_url": content_url,
            }
        )
        frontier.extend((story_id, int(value)) for value in story.get("kids", []))

    batch_size = max(workers * 20, 100)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        while frontier:
            batch: list[tuple[int, int]] = []
            while frontier and len(batch) < batch_size:
                story_id, item_id = frontier.popleft()
                seen = seen_by_story[story_id]
                if item_id in seen or len(seen) >= max_comments_per_story:
                    continue
                seen.add(item_id)
                batch.append((story_id, item_id))
            if not batch:
                continue
            items = list(
                executor.map(
                    lambda pair: fetch_item(pair[1]),
                    batch,
                )
            )
            for (story_id, _item_id), item in zip(batch, items):
                if not item or item.get("type") != "comment" or "time" not in item:
                    continue
                story = stories_by_id[story_id]
                content_url = str(
                    story.get("url")
                    or f"https://news.ycombinator.com/item?id={story_id}"
                )
                records.append(
                    {
                        "platform": "HackerNews",
                        "community": str(feed),
                        "content_id": str(story_id),
                        "event_id": str(item["id"]),
                        "parent_event_id": str(item.get("parent", story_id)),
                        "created_at": _iso_timestamp(item["time"]),
                        "author_id": str(item.get("by", "")),
                        "text": _plain_text(str(item.get("text", ""))),
                        "score": 0.0,
                        "event_type": "comment",
                        "removed": bool(item.get("deleted") or item.get("dead")),
                        "content_url": content_url,
                    }
                )
                collected_by_story[story_id] += 1
                if len(seen_by_story[story_id]) < max_comments_per_story:
                    frontier.extend(
                        (story_id, int(value)) for value in item.get("kids", [])
                    )

    truncations = []
    records_by_story: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        records_by_story.setdefault(int(record["content_id"]), []).append(record)
    for story_id, story in stories_by_id.items():
        reported = int(story.get("descendants", 0))
        observed = int(collected_by_story[story_id])
        story_records = records_by_story.get(story_id, [])
        timestamps = {
            str(record["event_id"]): pd.to_datetime(record["created_at"], utc=True)
            for record in story_records
        }
        temporal_violations = sum(
            1
            for record in story_records
            if timestamps.get(str(record["parent_event_id"])) is not None
            and timestamps[str(record["parent_event_id"])]
            > timestamps[str(record["event_id"])]
        )
        truncations.append(
            {
                "content_id": str(story_id),
                "reported_comments": reported,
                "collected_comments": observed,
                "visited_comment_items": len(seen_by_story[story_id]),
                "truncated": observed < reported,
                "temporal_order_violations": temporal_violations,
            }
        )

    complete_story_ids = [
        item["content_id"]
        for item in truncations
        if not bool(item["truncated"]) and int(item["temporal_order_violations"]) == 0
    ]
    retained_story_ids = set(complete_story_ids[:story_count])
    frame = pd.DataFrame(records)
    frame = frame[frame["content_id"].astype(str).isin(retained_story_ids)].copy()
    if frame.empty:
        raise RuntimeError("Hacker News collection returned no usable events")
    events_path = output_dir / "hackernews_events.parquet"
    frame.to_parquet(events_path, index=False)
    frame.to_csv(output_dir / "hackernews_events.csv", index=False)
    collected_story_count = int(frame["content_id"].nunique())
    target_met = collected_story_count >= int(story_count)
    data_complete = bool(target_met and not request_failures)
    manifest = {
        "status": "complete" if data_complete else "partial",
        "data_complete": data_complete,
        "source": discovery_audit["content_source"],
        "discovery": discovery_audit,
        "base_url": base_url,
        "feed": feed,
        "selection_mode": selection_mode,
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "story_count_requested": int(story_count),
        "story_count_collected": collected_story_count,
        "target_met": target_met,
        "eligible_stories_traversed": int(len(stories)),
        "structurally_complete_stories": int(len(complete_story_ids)),
        "incomplete_stories_excluded": int(
            sum(bool(item["truncated"]) for item in truncations)
        ),
        "temporally_reparented_stories_excluded": int(
            sum(int(item["temporal_order_violations"]) > 0 for item in truncations)
        ),
        "candidate_item_limit": int(len(candidate_ids)),
        "historical_scan_multiplier": int(historical_scan_multiplier),
        "request_failure_count": int(len(request_failures)),
        "request_failures": request_failures,
        "event_count": int(len(frame)),
        "max_comments_per_story": int(max_comments_per_story),
        "workers": int(workers),
        "events_sha256": sha256_file(events_path),
        "selection_limit": (
            "A bounded feed snapshot is not representative of all Hacker News discussions."
            if selection_mode == "bounded_feed"
            else (
                "Reverse maxitem scanning is a consecutive-item sample, not a popularity-ranked sample."
                if selection_mode == "official_maxitem_reverse_scan"
                else "Algolia is used only as a historical story index; index coverage may differ from Firebase."
            )
        ),
        "completeness_rule": (
            "Stories reporting more comments than max_comments_per_story are excluded before collection; "
            "stories with fewer retrievable comment items than the API-reported descendant count or "
            "parent-after-child moderation reparenting are excluded after traversal."
        ),
        "truncations": truncations,
    }
    write_json(output_dir / "collection_manifest.json", manifest)
    return frame, manifest
