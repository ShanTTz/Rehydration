from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
import requests

from bdmtf.revision.provenance import sha256_file, write_json


def _get_json(
    base_url: str,
    endpoint: str,
    params: dict[str, Any],
    timeout: int = 60,
    retries: int = 4,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.get(
                base_url.rstrip("/") + endpoint,
                params=params,
                headers={"User-Agent": "bdmtf-intervention-validation/0.3"},
                timeout=timeout,
            )
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            time.sleep(0.5 * (2**attempt))
    raise RuntimeError(f"Lemmy API request failed: {endpoint} {params}") from last_error


def _post_record(post_view: dict[str, Any], instance: str) -> dict[str, Any] | None:
    post = post_view.get("post", {}) or {}
    community = post_view.get("community", {}) or {}
    counts = post_view.get("counts", {}) or {}
    if not post.get("id") or not post.get("published"):
        return None
    return {
        "platform": "lemmy",
        "instance": instance,
        "content_id": str(post["id"]),
        "author_id": str((post_view.get("creator") or {}).get("id") or post.get("creator_id") or ""),
        "community_id": str(community.get("id") or post.get("community_id") or "unknown"),
        "community": str(community.get("name") or community.get("id") or "unknown"),
        "published": post["published"],
        "title": str(post.get("name") or ""),
        "url": str(post.get("url") or post.get("ap_id") or ""),
        "removed": bool(post.get("removed")),
        "locked": bool(post.get("locked")),
        "score": float(counts.get("score") or 0),
        "reported_comments": int(counts.get("comments") or 0),
    }


def collect_community_candidates(
    base_url: str,
    community_ids: Iterable[str],
    pages: int,
    page_size: int,
    workers: int = 1,
    cache_dir: Path | None = None,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    instance = urlsplit(base_url).netloc
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    communities = sorted(
        set(
            str(value)
            for value in community_ids
            if str(value) != "unknown"
        )
    )

    def collect_one(
        community_id: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
        records: list[dict[str, Any]] = []
        failures: list[dict[str, str]] = []
        for page in range(1, pages + 1):
            cache_path = (
                cache_dir
                / f"community-{community_id}-page-{page:04d}.json"
                if cache_dir is not None
                else None
            )
            if cache_path is not None and cache_path.is_file():
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
            else:
                try:
                    payload = _get_json(
                        base_url,
                        "/api/v3/post/list",
                        {
                            "community_id": int(community_id),
                            "sort": "New",
                            "page": page,
                            "limit": page_size,
                        },
                    )
                except RuntimeError as exc:
                    failures.append(
                        {
                            "stage": "candidate_posts",
                            "community_id": community_id,
                            "page": str(page),
                            "reason": str(exc),
                        }
                    )
                    break
                if cache_path is not None:
                    temporary = cache_path.with_suffix(".json.tmp")
                    temporary.write_text(
                        json.dumps(payload, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    temporary.replace(cache_path)
            posts = payload.get("posts", []) or []
            for post_view in posts:
                record = _post_record(post_view, instance)
                if record:
                    records.append(record)
            if len(posts) < page_size:
                break
        return records, failures

    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=max(1, min(workers, 4))) as executor:
        for community_records, community_failures in executor.map(
            collect_one,
            communities,
        ):
            records.extend(community_records)
            failures.extend(community_failures)
    frame = (
        pd.DataFrame(records).drop_duplicates("content_id")
        if records
        else pd.DataFrame()
    )
    return frame, failures


def _comment_parent(path: str, post_id: str) -> tuple[str, int]:
    nodes = [value for value in str(path).split(".") if value and value != "0"]
    if len(nodes) <= 1:
        return post_id, 1
    return nodes[-2], len(nodes)


def collect_post_thread(
    base_url: str,
    post_id: str,
    cache_dir: Path,
    pages: int = 20,
    page_size: int = 50,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{post_id}.json"
    if cache_path.is_file():
        pages_payload = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        pages_payload = []
        for page in range(1, pages + 1):
            payload = _get_json(
                base_url,
                "/api/v3/comment/list",
                {
                    "post_id": int(post_id),
                    "sort": "Old",
                    "page": page,
                    "limit": page_size,
                },
            )
            comments = payload.get("comments", []) or []
            pages_payload.append(payload)
            if len(comments) < page_size:
                break
        cache_path.write_text(json.dumps(pages_payload, ensure_ascii=False), encoding="utf-8")

    events: list[dict[str, Any]] = []
    observed_post: dict[str, Any] | None = None
    instance = urlsplit(base_url).netloc
    for payload in pages_payload:
        for view in payload.get("comments", []) or []:
            comment = view.get("comment", {}) or {}
            post = view.get("post", {}) or {}
            if observed_post is None:
                observed_post = _post_record(view, instance)
            if not comment.get("id") or not comment.get("published"):
                continue
            parent_id, depth = _comment_parent(str(comment.get("path") or ""), str(post_id))
            events.append(
                {
                    "platform": "lemmy",
                    "community": str((view.get("community") or {}).get("name") or post.get("community_id") or "unknown"),
                    "community_id": str((view.get("community") or {}).get("id") or post.get("community_id") or "unknown"),
                    "content_id": str(post_id),
                    "event_id": str(comment["id"]),
                    "parent_event_id": parent_id,
                    "created_at": comment["published"],
                    "depth": depth,
                    "author_id": str((view.get("creator") or {}).get("id") or comment.get("creator_id") or ""),
                    "text": str(comment.get("content") or ""),
                    "event_type": "comment",
                    "removed": bool(comment.get("removed") or comment.get("deleted")),
                    "content_url": str(post.get("url") or post.get("ap_id") or ""),
                }
            )
    if observed_post is None:
        observed_post = collect_post_snapshot(base_url, post_id, cache_dir)
    return events, observed_post


def collect_post_snapshot(
    base_url: str,
    post_id: str,
    cache_dir: Path,
) -> dict[str, Any] | None:
    """Fetch post metadata even when a treated thread has no visible comments."""
    cache_path = cache_dir / f"post-{post_id}.json"
    unavailable_path = cache_dir / f"post-{post_id}.unavailable.json"
    if unavailable_path.is_file():
        return None
    if cache_path.is_file():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        try:
            payload = _get_json(base_url, "/api/v3/post", {"id": int(post_id)})
        except RuntimeError as exc:
            unavailable_path.write_text(
                json.dumps(
                    {
                        "content_id": str(post_id),
                        "status": "unavailable_after_fixed_retries",
                        "reason": str(exc),
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            return None
        cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    view = payload.get("post_view", {}) or {}
    return _post_record(view, urlsplit(base_url).netloc) if view else None


def collect_lemmy_outcomes(
    base_url: str,
    interventions_path: Path,
    output_dir: Path,
    community_pages: int = 5,
    max_treated_posts: int = 0,
    comment_pages: int = 20,
    page_size: int = 50,
    workers: int = 8,
    intervention_types: Iterable[str] = ("remove_post", "lock_post"),
    deduplicate_content: bool = False,
    matching: dict[str, Any] | None = None,
    pre_days: int = 7,
    post_days: int = 7,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    interventions = pd.read_parquet(interventions_path)
    interventions = interventions[
        interventions["intervention_type"].isin(list(intervention_types))
        & interventions["active"].astype(bool)
    ].copy()
    interventions = interventions.sort_values("occurred_at")
    dedupe_columns = ["content_id"] if deduplicate_content else [
        "content_id",
        "intervention_type",
    ]
    interventions = interventions.drop_duplicates(dedupe_columns, keep="first")
    if max_treated_posts:
        interventions = interventions.head(max_treated_posts)
    candidates, candidate_failures = collect_community_candidates(
        base_url,
        interventions["community_id"].astype(str),
        community_pages,
        page_size,
        workers,
        output_dir / "candidate_cache",
    )
    target_ids = interventions["content_id"].astype(str).tolist()
    cache_dir = output_dir / "cache"

    def collect(post_id: str) -> tuple[str, list[dict[str, Any]], dict[str, Any] | None]:
        events, post = collect_post_thread(
            base_url,
            post_id,
            cache_dir,
            comment_pages,
            page_size,
        )
        return post_id, events, post

    def collect_many(
        post_ids: Iterable[str],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, str]]]:
        collected_events: list[dict[str, Any]] = []
        collected_posts: list[dict[str, Any]] = []
        collected_failures: list[dict[str, str]] = []
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = {executor.submit(collect, post_id): post_id for post_id in post_ids}
            for future, post_id in [(future, futures[future]) for future in futures]:
                try:
                    _, post_events, post = future.result()
                    collected_events.extend(post_events)
                    if post:
                        collected_posts.append(post)
                except RuntimeError as exc:
                    collected_failures.append({"content_id": post_id, "reason": str(exc)})
        return collected_events, collected_posts, collected_failures

    all_events, observed_posts, failures = collect_many(target_ids)
    failures = [*candidate_failures, *failures]
    preliminary_posts = pd.concat(
        [frame for frame in (candidates, pd.DataFrame(observed_posts)) if not frame.empty],
        ignore_index=True,
    ).drop_duplicates("content_id")
    preliminary_matches = _lemmy_risk_set_candidates(
        interventions,
        preliminary_posts,
        matching,
    )
    if not preliminary_matches.empty:
        preliminary_matches.to_csv(
            output_dir / "preliminary_control_candidates.csv",
            index=False,
        )
    control_ids = (
        preliminary_matches["control_content_id"].astype(str).drop_duplicates().tolist()
        if not preliminary_matches.empty
        else []
    )
    control_events, control_posts, control_failures = collect_many(control_ids)
    all_events.extend(control_events)
    observed_posts.extend(control_posts)
    failures.extend(control_failures)

    posts = pd.concat(
        [frame for frame in (candidates, pd.DataFrame(observed_posts)) if not frame.empty],
        ignore_index=True,
    ).drop_duplicates("content_id")
    events = pd.DataFrame(all_events)
    if not posts.empty:
        posts.to_parquet(output_dir / "lemmy_posts.parquet", index=False)
        posts.to_csv(output_dir / "lemmy_posts.csv", index=False)
    if not events.empty:
        events = events.drop_duplicates(["platform", "event_id"])
        events.to_parquet(output_dir / "lemmy_thread_events.parquet", index=False)
        events.to_csv(output_dir / "lemmy_thread_events.csv", index=False)
    cascade_events, cascade_manifest = build_lemmy_cascade_events(posts, events)
    if not cascade_events.empty:
        cascade_path = output_dir / "lemmy_cascade_events.parquet"
        cascade_events.to_parquet(cascade_path, index=False)
        cascade_events.to_csv(output_dir / "lemmy_cascade_events.csv", index=False)
        cascade_manifest["events_sha256"] = sha256_file(cascade_path)
    write_json(output_dir / "lemmy_cascade_manifest.json", cascade_manifest)
    matches, panel = build_lemmy_risk_set_panel(
        interventions,
        posts,
        events,
        pre_days,
        post_days,
        matching,
    )
    if not matches.empty:
        matches.to_csv(output_dir / "risk_set_matches.csv", index=False)
    if not panel.empty:
        panel.to_parquet(output_dir / "lemmy_outcome_panel.parquet", index=False)
        panel.to_csv(output_dir / "lemmy_outcome_panel.csv", index=False)
    manifest = {
        "status": (
            "pilot_complete"
            if max_treated_posts and not failures and len(matches)
            else "complete"
            if not failures and len(matches)
            else "partial"
        ),
        "claim_allowed": bool(not max_treated_posts and not failures and len(matches) and not panel.empty),
        "analysis_ready": bool(len(matches) and not panel.empty),
        "max_treated_posts": int(max_treated_posts),
        "n_treated_interventions": int(len(interventions)),
        "n_treated_posts_observed": int(
            posts["content_id"].astype(str).isin(target_ids).sum()
        ),
        "n_treated_posts_unavailable": int(
            len(set(target_ids).difference(set(posts["content_id"].astype(str))))
        ),
        "n_candidate_posts": int(len(candidates)),
        "n_candidate_control_threads": int(len(control_ids)),
        "n_collected_posts": int(len(posts)),
        "n_comment_events": int(len(events)),
        "n_complete_cascades": int(cascade_manifest["n_complete_cascades"]),
        "n_risk_set_pairs": int(len(matches)),
        "n_panel_rows": int(len(panel)),
        "failures": failures,
        "interventions_sha256": sha256_file(interventions_path),
        "selection_rule": {
            "intervention_types": sorted(set(intervention_types)),
            "active_only": True,
            "deduplicate_content": bool(deduplicate_content),
            "max_treated_posts": int(max_treated_posts),
        },
        "matching_rule": {
            "same_community": bool((matching or {}).get("same_community", True)),
            "without_replacement": bool(
                (matching or {}).get("without_replacement", False)
            ),
            "candidate_pool_size": int(
                (matching or {}).get("candidate_pool_size", 1)
            ),
            "distance": "absolute log1p post-age difference at intervention",
            "max_log_age_distance": (matching or {}).get(
                "max_log_age_distance"
            ),
            "pre_event_matching": bool(
                (matching or {}).get("pre_event_matching", False)
            ),
            "max_pretrajectory_distance": (matching or {}).get(
                "max_pretrajectory_distance"
            ),
            "outcome_blind": True,
        },
        "window": {"pre_days": int(pre_days), "post_days": int(post_days)},
        "causal_limit": "Risk-set matching still requires balance, overlap, pretrend, and sensitivity diagnostics.",
    }
    write_json(output_dir / "outcome_collection_manifest.json", manifest)
    return manifest


def build_lemmy_cascade_events(
    posts: pd.DataFrame,
    comments: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Join Lemmy post roots and comments while excluding broken parent chains."""
    required_post_columns = {"content_id", "published", "community"}
    required_comment_columns = {
        "content_id",
        "event_id",
        "parent_event_id",
        "created_at",
    }
    if (
        posts.empty
        or comments.empty
        or not required_post_columns.issubset(posts.columns)
        or not required_comment_columns.issubset(comments.columns)
    ):
        return pd.DataFrame(), {
            "status": "no_cascades",
            "n_candidate_threads": 0,
            "n_complete_cascades": 0,
            "excluded_broken_parent_chain": 0,
        }

    unique_posts = posts.drop_duplicates("content_id").copy()
    unique_posts["content_id"] = unique_posts["content_id"].astype(str)
    post_lookup = unique_posts.set_index("content_id")
    records: list[dict[str, Any]] = []
    excluded = 0
    candidate_threads = 0
    for content_id, group in comments.groupby(comments["content_id"].astype(str), sort=False):
        if content_id not in post_lookup.index:
            excluded += 1
            continue
        candidate_threads += 1
        raw_ids = set(group["event_id"].astype(str))
        valid_parents = raw_ids | {content_id}
        if not set(group["parent_event_id"].fillna("").astype(str)).issubset(valid_parents):
            excluded += 1
            continue
        post = post_lookup.loc[content_id]
        if isinstance(post, pd.DataFrame):
            post = post.iloc[0]
        root_id = f"post:{content_id}"
        records.append(
            {
                "platform": "Lemmy",
                "community": str(post.get("community", "unknown")),
                "content_id": content_id,
                "event_id": root_id,
                "parent_event_id": "",
                "created_at": post["published"],
                "author_id": str(post.get("author_id", "")),
                "text": "",
                "title": str(post.get("title", "")),
                "score": float(post.get("score", 0) or 0),
                "event_type": "post",
                "removed": bool(post.get("removed", False)),
                "content_url": str(post.get("url", "")),
            }
        )
        for comment in group.itertuples(index=False):
            raw_parent = str(comment.parent_event_id)
            parent_id = root_id if raw_parent == content_id else f"comment:{raw_parent}"
            records.append(
                {
                    "platform": "Lemmy",
                    "community": str(getattr(comment, "community", post.get("community", "unknown"))),
                    "content_id": content_id,
                    "event_id": f"comment:{comment.event_id}",
                    "parent_event_id": parent_id,
                    "created_at": comment.created_at,
                    "author_id": str(getattr(comment, "author_id", "")),
                    "text": str(getattr(comment, "text", "")),
                    "title": "",
                    "score": 0.0,
                    "event_type": "comment",
                    "removed": bool(getattr(comment, "removed", False)),
                    "content_url": str(getattr(comment, "content_url", post.get("url", ""))),
                }
            )
    frame = pd.DataFrame(records)
    complete = int(frame.loc[frame.get("event_type", pd.Series(dtype=str)).eq("post"), "content_id"].nunique()) if not frame.empty else 0
    return frame, {
        "status": "ready" if complete else "no_cascades",
        "n_candidate_threads": candidate_threads,
        "n_complete_cascades": complete,
        "excluded_broken_parent_chain": excluded,
        "event_count": int(len(frame)),
        "id_rule": "Post and comment event IDs are namespace-prefixed before platform normalization.",
    }


def build_lemmy_risk_set_panel(
    interventions: pd.DataFrame,
    posts: pd.DataFrame,
    events: pd.DataFrame,
    pre_days: int = 7,
    post_days: int = 7,
    matching: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if interventions.empty or posts.empty:
        return pd.DataFrame(), pd.DataFrame()
    match_frame = _match_lemmy_risk_sets(
        interventions,
        posts,
        matching,
        events,
        pre_days,
    )
    if match_frame.empty:
        return match_frame, pd.DataFrame()
    events = events.copy()
    events["created_at"] = events["created_at"].map(
        lambda value: pd.to_datetime(value, utc=True, errors="coerce")
    )
    panel_records: list[dict[str, Any]] = []
    for match in match_frame.itertuples(index=False):
        for treated, content_id in ((1, match.treated_content_id), (0, match.control_content_id)):
            group = events[events["content_id"].astype(str) == str(content_id)].copy()
            group["relative_period"] = np.floor(
                (group["created_at"] - match.event_time).dt.total_seconds() / 86400.0
            ).astype("Int64")
            for relative in range(-pre_days, post_days + 1):
                period_group = group[group["relative_period"] == relative]
                values = {
                    "reply_count": float(len(period_group)),
                    "active_authors": float(period_group["author_id"].astype(str).nunique()),
                    "max_depth": float(period_group["depth"].max()) if len(period_group) else 0.0,
                    "removed_replies": float(period_group["removed"].astype(bool).sum()),
                }
                for outcome, value in values.items():
                    panel_records.append(
                        {
                            "unit_id": f"{match.intervention_id}|{treated}|{content_id}",
                            "period": match.event_time.floor("D") + pd.Timedelta(days=relative),
                            "treated": treated,
                            "relative_period": relative,
                            "outcome": outcome,
                            "value": value,
                            "community_id": match.community_id,
                            "intervention_id": match.intervention_id,
                            "intervention_type": match.intervention_type,
                            "content_id": content_id,
                        }
                    )
    return match_frame, pd.DataFrame(panel_records)


def _lemmy_risk_set_candidates(
    interventions: pd.DataFrame,
    posts: pd.DataFrame,
    matching: dict[str, Any] | None = None,
) -> pd.DataFrame:
    matching = matching or {}
    same_community = bool(matching.get("same_community", True))
    configured_pool_size = matching.get("candidate_pool_size")
    candidate_pool_size = (
        int(configured_pool_size)
        if configured_pool_size is not None
        else 100
        if bool(matching.get("without_replacement", False))
        else 1
    )
    configured_caliper = matching.get("max_log_age_distance")
    max_log_age_distance = (
        float(configured_caliper)
        if configured_caliper is not None
        else float("inf")
    )
    posts = posts.copy()
    posts["published"] = posts["published"].map(lambda value: pd.to_datetime(value, utc=True, errors="coerce"))
    interventions = interventions.copy()
    interventions["occurred_at"] = interventions["occurred_at"].map(
        lambda value: pd.to_datetime(value, utc=True, errors="coerce")
    )
    treated_ids = set(interventions["content_id"].astype(str))
    controls = posts[~posts["content_id"].astype(str).isin(treated_ids)].copy()
    candidates: list[dict[str, Any]] = []
    for intervention in interventions.itertuples(index=False):
        treated_post = posts[posts["content_id"].astype(str) == str(intervention.content_id)]
        if treated_post.empty:
            continue
        treated_post = treated_post.iloc[0]
        event_time = intervention.occurred_at
        treated_age = (event_time - treated_post["published"]).total_seconds() / 3600.0
        eligible = controls["published"] <= event_time
        if same_community:
            eligible &= (
                controls["community_id"].astype(str)
                == str(intervention.community_id)
            )
        risk_set = controls[eligible].copy()
        if risk_set.empty or treated_age < 0:
            continue
        risk_set["age_hours"] = (event_time - risk_set["published"]).dt.total_seconds() / 3600.0
        risk_set["distance"] = (
            np.log1p(risk_set["age_hours"].clip(lower=0))
            - np.log1p(max(0.0, treated_age))
        )
        risk_set["distance"] = risk_set["distance"].abs()
        risk_set = risk_set[
            risk_set["distance"] <= max_log_age_distance
        ]
        if risk_set.empty:
            continue
        risk_set = risk_set.sort_values(
            ["distance", "content_id"]
        ).head(max(1, candidate_pool_size))
        for rank, (_, control) in enumerate(risk_set.iterrows(), start=1):
            candidates.append(
                {
                    "intervention_id": intervention.intervention_id,
                    "intervention_type": intervention.intervention_type,
                    "community_id": str(intervention.community_id),
                    "event_time": event_time,
                    "treated_content_id": str(intervention.content_id),
                    "control_content_id": str(control["content_id"]),
                    "treated_age_hours": treated_age,
                    "control_age_hours": float(control["age_hours"]),
                    "distance": float(control["distance"]),
                    "candidate_rank": rank,
                    "distance_definition": "absolute_log1p_age_hours",
                    "same_community": bool(
                        str(intervention.community_id)
                        == str(control["community_id"])
                    ),
                }
            )
    return pd.DataFrame(candidates)


def _pre_event_vector(
    events_by_content: dict[str, pd.DataFrame],
    content_id: str,
    event_time: pd.Timestamp,
    pre_days: int,
) -> np.ndarray:
    group = events_by_content.get(str(content_id))
    if group is None or group.empty:
        return np.zeros(pre_days * 3, dtype=float)
    relative = np.floor(
        (group["created_at"] - event_time).dt.total_seconds() / 86400.0
    )
    prepared = group.assign(relative_period=relative)
    values: list[float] = []
    for outcome in ("reply_count", "active_authors", "max_depth"):
        for period in range(-pre_days, 0):
            window = prepared[prepared["relative_period"].eq(period)]
            if outcome == "reply_count":
                value = float(len(window))
            elif outcome == "active_authors":
                value = float(window["author_id"].astype(str).nunique())
            else:
                value = (
                    float(window["depth"].max())
                    if len(window)
                    else 0.0
                )
            values.append(float(np.log1p(value)))
    return np.asarray(values, dtype=float)


def _match_lemmy_risk_sets(
    interventions: pd.DataFrame,
    posts: pd.DataFrame,
    matching: dict[str, Any] | None = None,
    events: pd.DataFrame | None = None,
    pre_days: int = 7,
) -> pd.DataFrame:
    matching = matching or {}
    candidates = _lemmy_risk_set_candidates(
        interventions,
        posts,
        matching,
    )
    if candidates.empty:
        return candidates
    candidates = candidates.copy()
    candidates["pretrajectory_distance"] = 0.0
    if (
        bool(matching.get("pre_event_matching", False))
        and events is not None
        and not events.empty
    ):
        prepared_events = events.copy()
        prepared_events["content_id"] = prepared_events[
            "content_id"
        ].astype(str)
        prepared_events["created_at"] = pd.to_datetime(
            prepared_events["created_at"],
            utc=True,
            errors="coerce",
        )
        prepared_events = prepared_events.dropna(subset=["created_at"])
        events_by_content = {
            content_id: group
            for content_id, group in prepared_events.groupby(
                "content_id",
                sort=False,
            )
        }
        vector_cache: dict[tuple[str, int], np.ndarray] = {}

        def vector(content_id: str, event_time: pd.Timestamp) -> np.ndarray:
            key = (str(content_id), int(event_time.value))
            if key not in vector_cache:
                vector_cache[key] = _pre_event_vector(
                    events_by_content,
                    str(content_id),
                    event_time,
                    pre_days,
                )
            return vector_cache[key]

        trajectory_distances: list[float] = []
        for candidate in candidates.itertuples(index=False):
            treated_vector = vector(
                candidate.treated_content_id,
                candidate.event_time,
            )
            control_vector = vector(
                candidate.control_content_id,
                candidate.event_time,
            )
            trajectory_distances.append(
                float(np.mean(np.abs(treated_vector - control_vector)))
            )
        candidates["pretrajectory_distance"] = trajectory_distances
        configured_caliper = matching.get(
            "max_pretrajectory_distance"
        )
        if configured_caliper is not None:
            candidates = candidates[
                candidates["pretrajectory_distance"]
                <= float(configured_caliper)
            ]
    if candidates.empty:
        return candidates
    candidates["match_score"] = (
        float(matching.get("age_distance_weight", 1.0))
        * candidates["distance"]
        + float(matching.get("pretrajectory_distance_weight", 1.0))
        * candidates["pretrajectory_distance"]
    )
    without_replacement = bool(
        matching.get("without_replacement", False)
    )
    candidate_counts = candidates.groupby(
        "intervention_id"
    ).size().to_dict()
    candidates["_candidate_count"] = candidates[
        "intervention_id"
    ].map(candidate_counts)
    intervention_order = (
        candidates[
            ["intervention_id", "_candidate_count", "event_time"]
        ]
        .drop_duplicates("intervention_id")
        .sort_values(
            ["_candidate_count", "event_time", "intervention_id"]
        )["intervention_id"]
        .tolist()
    )
    used_controls: set[str] = set()
    selected: list[pd.Series] = []
    for intervention_id in intervention_order:
        group = candidates[
            candidates["intervention_id"].eq(intervention_id)
        ]
        if without_replacement and used_controls:
            group = group[
                ~group["control_content_id"].astype(str).isin(
                    used_controls
                )
            ]
        if group.empty:
            continue
        choice = group.sort_values(
            ["match_score", "distance", "control_content_id"]
        ).iloc[0]
        selected.append(choice)
        if without_replacement:
            used_controls.add(str(choice["control_content_id"]))
    if not selected:
        return pd.DataFrame()
    result = pd.DataFrame(selected).drop(
        columns=["_candidate_count"],
        errors="ignore",
    )
    result["pre_event_feature_definition"] = (
        "mean absolute difference across log1p daily reply-count, "
        "active-author, and max-depth vectors for frozen pre days"
    )
    return result.reset_index(drop=True)
