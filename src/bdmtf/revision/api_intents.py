from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from bdmtf.data.social_loader import load_comments, load_posts, resolve_community_paths
from bdmtf.revision.data_pipeline import COMMUNITIES
from bdmtf.revision.provenance import write_json


def _headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


def list_model_ids(base_url: str, api_key: str) -> list[str]:
    response = requests.get(f"{base_url.rstrip('/')}/models", headers=_headers(api_key), timeout=60)
    response.raise_for_status()
    return sorted(str(item["id"]) for item in response.json().get("data", []) if item.get("id"))


def select_models(available: list[str], config: dict[str, Any]) -> list[dict[str, str]]:
    selected = []
    for spec in config.get("preferred_models", []) + config.get("fallback_models", []):
        match = next((model for pattern in spec["patterns"] for model in available if pattern.lower() in model.lower()), None)
        if match and all(item["family"] != spec["family"] for item in selected):
            selected.append({"family": spec["family"], "model": match})
        if len(selected) == int(config.get("require_distinct_families", 3)):
            break
    required = int(config.get("require_distinct_families", 3))
    if len(selected) < required:
        raise RuntimeError(f"Only {len(selected)} distinct requested model families are available; {required} are required")
    return selected


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


def _post_context(post: Any) -> str:
    title = _clean_text(getattr(post, "title", ""))
    body = _clean_text(getattr(post, "full_text", ""))
    if title and body and body != title:
        return f"{title}\n\n{body}"
    return title or body


def _participant_context(profile: dict[str, Any], leader: int) -> str:
    persona = _clean_text(profile.get("persona"))
    bio = _clean_text(profile.get("bio"))
    topics = profile.get("interested_topics", [])
    topic_text = ", ".join(
        _clean_text(item) for item in topics if _clean_text(item)
    )
    parts = [
        f"Participant index: {leader}",
        f"Persona: {persona[:1800]}" if persona else "",
        f"Bio: {bio[:400]}" if bio else "",
        f"Interests: {topic_text[:400]}" if topic_text else "",
    ]
    return "\n".join(part for part in parts if part)


def build_intent_tasks(social_root: Path, splits_path: Path, config: dict[str, Any]) -> pd.DataFrame:
    splits = pd.read_csv(splits_path)
    rows = []
    posts_per_community = int(config.get("posts_per_community", 10))
    leaders = int(config.get("leaders_per_post", 10))
    for community in COMMUNITIES:
        paths = resolve_community_paths(social_root, community)
        posts = load_posts(paths, enriched=False)
        population = []
        if paths.population_json and paths.population_json.is_file():
            population = json.loads(
                paths.population_json.read_text(encoding="utf-8")
            )
        candidates = splits[(splits["community"] == community) & (splits["split"] == "test")].sort_values(["is_viral", "created_at", "post_id"])
        selected_ids = candidates.groupby("is_viral", group_keys=False).head(max(1, posts_per_community // 2))["post_id"].astype(str).tolist()
        selected_ids = selected_ids[:posts_per_community]
        selected_posts = posts[posts["post_id"].astype(str).isin(selected_ids)]
        for post in selected_posts.itertuples(index=False):
            text = _post_context(post)[:3000]
            if not text:
                raise ValueError(
                    f"No textual context for {community}/{post.post_id}"
                )
            for leader in range(leaders):
                profile = (
                    population[leader % len(population)]
                    if population
                    else {}
                )
                participant = _participant_context(profile, leader)
                prompt = (
                    "You are a participant deciding how to respond to a Reddit post. "
                    "No popularity, ranking, vote, or later feedback is visible. Return JSON with keys "
                    "action (reply or abstain), polarity (constructive, neutral, antagonistic), and content.\n"
                    f"Community: {community}\nPost: {text}\n{participant}"
                )
                rows.append(
                    {
                        "task_id": hashlib.sha256(f"{community}|{post.post_id}|{leader}".encode()).hexdigest()[:20],
                        "community": community,
                        "post_id": str(post.post_id),
                        "leader": leader,
                        "post_context_sha256": hashlib.sha256(
                            text.encode("utf-8")
                        ).hexdigest(),
                        "participant_context_sha256": hashlib.sha256(
                            participant.encode("utf-8")
                        ).hexdigest(),
                        "prompt": prompt,
                        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    }
                )
    return pd.DataFrame(rows)


def build_annotation_tasks(social_root: Path, config: dict[str, Any]) -> pd.DataFrame:
    annotation = config.get("semantic_annotation", {})
    per_community = int(annotation.get("comments_per_community", 100))
    batch_size = int(annotation.get("batch_size", 20))
    seed = int(config.get("seed", 30371))
    rows = []
    for community_index, community in enumerate(COMMUNITIES):
        comments = load_comments(resolve_community_paths(social_root, community)).copy()
        comments["depth_bucket"] = pd.cut(
            pd.to_numeric(comments.get("depth", 0), errors="coerce").fillna(0),
            bins=[-1, 0, 2, 5, float("inf")],
            labels=False,
        )
        samples = []
        for _, group in comments.groupby("depth_bucket", dropna=False):
            take = min(len(group), max(1, per_community // 4))
            samples.append(group.sample(n=take, random_state=seed + community_index))
        selected = pd.concat(samples).drop_duplicates("comment_id").head(per_community)
        for batch_index in range(0, len(selected), batch_size):
            batch = selected.iloc[batch_index : batch_index + batch_size]
            items = [
                {"comment_id": str(item.comment_id), "text": str(item.comment_text)[:1200]}
                for item in batch.itertuples(index=False)
            ]
            prompt = (
                "Label each Reddit comment. Return JSON with an items array preserving comment_id. "
                "For each item include toxicity (0..1), emotion (one short label), topic (one short label), "
                "and counterspeech (boolean). Do not infer user identity.\nComments:\n"
                + json.dumps(items, ensure_ascii=False)
            )
            rows.append(
                {
                    "task_id": hashlib.sha256(f"annotation|{community}|{batch_index}".encode()).hexdigest()[:20],
                    "community": community,
                    "batch_index": batch_index // batch_size,
                    "comment_count": len(items),
                    "item_ids": [item["comment_id"] for item in items],
                    "prompt": prompt,
                    "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                }
            )
    return pd.DataFrame(rows)


def generate_intents(social_root: Path, splits_path: Path, config_path: Path, output_dir: Path, execute: bool = False) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    tasks = build_intent_tasks(social_root, splits_path, config)
    annotation_tasks = build_annotation_tasks(social_root, config)
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks.drop(columns=["prompt"]).to_csv(output_dir / "intent_tasks.csv", index=False)
    annotation_tasks.drop(columns=["prompt"]).to_csv(output_dir / "semantic_annotation_tasks.csv", index=False)
    base_url = (
        os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        .strip()
        .rstrip("/")
    )
    key = os.environ.get("OPENAI_API_KEY", "")
    manifest: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "execute": execute,
        "task_count_before_models": int(len(tasks)),
        "semantic_annotation_batches_before_models": int(len(annotation_tasks)),
        "expected_intent_calls": int(len(tasks) * int(config.get("require_distinct_families", 3))),
        "expected_annotation_calls": int(len(annotation_tasks) * int(config.get("require_distinct_families", 3))),
        "expected_calls": int((len(tasks) + len(annotation_tasks)) * int(config.get("require_distinct_families", 3))),
        "base_host": base_url.split("//", 1)[-1].split("/", 1)[0],
        "key_stored": False,
    }
    if not execute:
        manifest["status"] = "dry_run_ready"
        write_json(output_dir / "api_manifest.json", manifest)
        return manifest
    if not key:
        raise RuntimeError("OPENAI_API_KEY is required for --execute; no key is read from repository files")
    selected = select_models(list_model_ids(base_url, key), config)
    manifest["models"] = selected
    intent_execution = _execute_tasks(
        tasks,
        selected,
        config,
        base_url,
        key,
        output_dir / "frozen_intents_multimodel.jsonl",
        "intent",
    )
    annotation_execution = _execute_tasks(
        annotation_tasks,
        selected,
        config,
        base_url,
        key,
        output_dir / "semantic_annotations_multimodel.jsonl",
        "annotation",
    )
    summarize_multimodel_cache(output_dir / "frozen_intents_multimodel.jsonl", output_dir / "model_semantic_summary.csv")
    manifest["execution"] = {
        "intent": intent_execution,
        "annotation": annotation_execution,
    }
    failures = (
        int(intent_execution["failed"])
        + int(annotation_execution["failed"])
    )
    manifest["status"] = "complete" if failures == 0 else "partial"
    write_json(output_dir / "api_manifest.json", manifest)
    return manifest


def _execute_tasks(
    tasks: pd.DataFrame,
    selected: list[dict[str, str]],
    config: dict[str, Any],
    base_url: str,
    key: str,
    cache_path: Path,
    task_kind: str,
) -> dict[str, int]:
    existing = set()
    completed = 0
    skipped = 0
    failed = 0
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                existing.add(
                    (
                        item["task_id"],
                        item["model"],
                        item.get("prompt_sha256", ""),
                    )
                )
    with cache_path.open("a", encoding="utf-8") as handle:
        for task in tasks.to_dict(orient="records"):
            for model in selected:
                cache_key = (
                    task["task_id"],
                    model["model"],
                    task.get("prompt_sha256", ""),
                )
                if cache_key in existing:
                    skipped += 1
                    continue
                max_tokens = int(
                    config.get(
                        f"{task_kind}_max_tokens",
                        config.get("max_tokens", 180),
                    )
                )
                payload = {
                    "model": model["model"],
                    "temperature": float(config.get("temperature", 0.2)),
                    "max_tokens": max_tokens,
                    "messages": [{"role": "user", "content": task["prompt"]}],
                    "response_format": {"type": "json_object"},
                }
                overrides = dict(
                    config.get("family_request_overrides", {}).get(
                        model["family"], {}
                    )
                )
                if overrides.pop("omit_max_tokens", False):
                    payload.pop("max_tokens", None)
                payload.update(overrides)
                max_retries = int(config.get("max_retries", 0))
                retry_delay = float(config.get("retry_delay_seconds", 1.0))
                started = time.time()
                raw: dict[str, Any] | None = None
                content = ""
                attempt = 0
                last_error: Exception | None = None
                for attempt in range(1, max_retries + 2):
                    try:
                        response = requests.post(
                            f"{base_url}/chat/completions",
                            headers=_headers(key),
                            json=payload,
                            timeout=120,
                        )
                        response.raise_for_status()
                        raw = response.json()
                        content = raw["choices"][0]["message"]["content"]
                        if task_kind == "intent":
                            _validate_intent_response(content)
                        elif task_kind == "annotation":
                            _validate_annotation_response(content, task)
                        elif task_kind == "reviewer_semantic_annotation":
                            _validate_reviewer_semantic_response(
                                content,
                                task,
                            )
                        last_error = None
                        break
                    except (KeyError, TypeError, ValueError, json.JSONDecodeError, requests.RequestException) as error:
                        last_error = error
                        if attempt > max_retries:
                            failure_path = cache_path.with_name(
                                f"{cache_path.stem}_failures.jsonl"
                            )
                            failure = {
                                "task_id": task["task_id"],
                                "task_kind": task_kind,
                                "family": model["family"],
                                "model": model["model"],
                                "returned_model": (
                                    raw.get("model")
                                    if isinstance(raw, dict)
                                    else None
                                ),
                                "finish_reason": (
                                    raw.get("choices", [{}])[0].get("finish_reason")
                                    if isinstance(raw, dict)
                                    else None
                                ),
                                "attempts": attempt,
                                "error_type": type(error).__name__,
                                "error": str(error),
                                "response_length": len(content),
                                "response_sha256": (
                                    hashlib.sha256(
                                        content.encode("utf-8")
                                    ).hexdigest()
                                    if content
                                    else None
                                ),
                                "created_at": datetime.now(
                                    timezone.utc
                                ).isoformat(),
                            }
                            with failure_path.open("a", encoding="utf-8") as failure_handle:
                                failure_handle.write(
                                    json.dumps(failure, ensure_ascii=False) + "\n"
                                )
                            failed += 1
                            break
                        time.sleep(retry_delay * attempt)
                if last_error is not None:
                    continue
                assert raw is not None
                record = {
                    **{key_name: task[key_name] for key_name in task if key_name != "prompt"},
                    "task_kind": task_kind,
                    "family": model["family"],
                    "model": model["model"],
                    "returned_model": raw.get("model"),
                    "system_fingerprint": raw.get("system_fingerprint"),
                    "temperature": payload["temperature"],
                    "request_options": {
                        key_name: payload[key_name]
                        for key_name in (
                            "temperature",
                            "max_tokens",
                            "response_format",
                            "thinking",
                            "enable_thinking",
                        )
                        if key_name in payload
                    },
                    "seed": int(config.get("seed", 30371)),
                    "attempts": attempt,
                    "response": content,
                    "response_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                    "usage": raw.get("usage", {}),
                    "latency_seconds": round(time.time() - started, 4),
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                completed += 1
    return {
        "completed": completed,
        "skipped": skipped,
        "failed": failed,
    }


def _parse_json_object(content: str) -> dict[str, Any]:
    cleaned = str(content).strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("response must be a JSON object")
    return parsed


def _validate_intent_response(content: str) -> None:
    parsed = _parse_json_object(content)
    action = str(parsed.get("action", "")).lower()
    polarity = str(parsed.get("polarity", "")).lower()
    if action not in {"reply", "abstain"}:
        raise ValueError("intent action must be reply or abstain")
    if polarity not in {"constructive", "neutral", "antagonistic"}:
        raise ValueError(
            "intent polarity must be constructive, neutral, or antagonistic"
        )
    if not isinstance(parsed.get("content"), str):
        raise ValueError("intent content must be a string")


def _validate_annotation_response(
    content: str,
    task: dict[str, Any],
) -> None:
    parsed = _parse_json_object(content)
    items = parsed.get("items")
    if not isinstance(items, list):
        raise ValueError("annotation response must contain an items array")
    expected_ids = [str(value) for value in task.get("item_ids", [])]
    actual_ids = [str(item.get("comment_id", "")) for item in items]
    if actual_ids != expected_ids:
        raise ValueError(
            "annotation response must preserve every comment_id in order"
        )
    for item in items:
        toxicity = float(item["toxicity"])
        if not 0.0 <= toxicity <= 1.0:
            raise ValueError("toxicity must be in [0, 1]")
        if not isinstance(item["emotion"], str):
            raise ValueError("emotion must be a string")
        if not isinstance(item["topic"], str):
            raise ValueError("topic must be a string")
        if not isinstance(item["counterspeech"], bool):
            raise ValueError("counterspeech must be boolean")


def _validate_reviewer_semantic_response(
    content: str, task: dict[str, Any]
) -> None:
    parsed = _parse_json_object(content)
    if not isinstance(parsed.get("items"), list):
        raise ValueError("semantic response must contain an items array")
    items = parsed["items"]
    expected_ids = [str(value) for value in task.get("item_ids", [])]
    actual_ids = [str(item.get("comment_id", "")) for item in items]
    if actual_ids != expected_ids:
        raise ValueError("semantic response must preserve every comment_id in order")
    numeric_labels = ("antagonism", "conflict_amplifying", "confidence")
    boolean_labels = (
        "constructive_disagreement",
        "counterspeech",
        "instruction_like_text",
    )
    for item in items:
        for label in numeric_labels:
            value = float(item[label])
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{label} must be in [0, 1]")
        for label in boolean_labels:
            if not isinstance(item[label], bool):
                raise ValueError(f"{label} must be boolean")


def summarize_multimodel_cache(cache_path: Path, output_path: Path) -> pd.DataFrame:
    frame = load_intent_response_frame(cache_path)
    if frame.empty:
        return frame
    summary = (
        frame.groupby(["family", "model", "community"])[
            ["reply", "antagonistic"]
        ]
        .mean()
        .reset_index()
    )
    summary.to_csv(output_path, index=False)
    return summary


def load_intent_response_frame(cache_path: Path) -> pd.DataFrame:
    rows = []
    if not cache_path.exists():
        return pd.DataFrame()
    for line in cache_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        response = _parse_json_object(item.get("response", "{}"))
        rows.append(
            {
                "task_id": item.get("task_id", ""),
                "family": item.get("family", ""),
                "model": item.get("model", ""),
                "community": item.get("community", ""),
                "post_id": str(item.get("post_id", "")),
                "leader": int(item.get("leader", 0)),
                "prompt_sha256": item.get("prompt_sha256", ""),
                "reply": str(response.get("action", "")).lower() == "reply",
                "antagonistic": str(response.get("polarity", "")).lower() == "antagonistic",
                "polarity": str(response.get("polarity", "")).lower(),
                "content_length": len(str(response.get("content", ""))),
            }
        )
    return pd.DataFrame(rows)
