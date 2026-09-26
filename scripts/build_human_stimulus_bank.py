from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOCIAL_ROOT = ROOT / "data" / "social_paper"
DEFAULT_OUTPUT = ROOT / "experiment_app" / "thread_fixture.json"
DEFAULT_AUDIT = ROOT / "artifacts" / "human_rct" / "stimulus_bank"
DEFAULT_TRANSLATION_CACHE = DEFAULT_AUDIT / "zh_cn_translation_cache.jsonl"

COMMUNITIES = ("AskReddit", "aww", "funny", "science", "worldnews")
RISK_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(?:suicide|self[- ]?harm|kill myself)\b",
        r"\b(?:rape|porn(?:ography)?|explicit sex|nudes?)\b",
        r"\bnsfw\b",
        r"\b(?:gore|behead(?:ed|ing)?|decapitat(?:ed|ion))\b",
        r"\b(?:kill|shoot|stab|hang)\s+(?:you|him|her|them)\b",
        r"\b(?:home address|phone number|doxx(?:ed|ing)?)\b",
    )
)
URL_PATTERN = re.compile(r"https?://\S+", re.IGNORECASE)
USER_PATTERN = re.compile(r"\bu/[A-Za-z0-9_-]+", re.IGNORECASE)
SPACE_PATTERN = re.compile(r"\s+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an anonymized, risk-screened human-study stimulus bank "
            "from the 500-post social-paper archive."
        )
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--comments-per-thread", type=int, default=10)
    parser.add_argument("--seed", type=int, default=30371)
    return parser.parse_args()


def stable_int(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:16], 16)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(value: str) -> str:
    text = URL_PATTERN.sub("[link removed]", str(value or ""))
    text = USER_PATTERN.sub("u/[anonymous]", text)
    return SPACE_PATTERN.sub(" ", text).strip()


def risk_reason(text: str) -> str | None:
    lowered = text.casefold()
    if lowered in {"[removed]", "[deleted]"}:
        return "removed_or_deleted"
    for index, pattern in enumerate(RISK_PATTERNS, start=1):
        if pattern.search(text):
            return f"high_risk_pattern_{index}"
    return None


def parse_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def created_timestamp(value: str, fallback: int) -> int:
    try:
        return int(datetime.fromisoformat(value).timestamp())
    except (TypeError, ValueError):
        return fallback


def community_paths(community: str) -> tuple[Path, Path]:
    folder = SOCIAL_ROOT / f"{community}_data"
    posts = folder / f"posts_features_final_enriched_safe_{community}.csv"
    comments = folder / f"comments_data_{community}.csv"
    return posts, comments


def post_conflict_score(row: dict[str, str]) -> float:
    conflict = min(
        1.0,
        parse_float(
            row.get("features_discourse_ecology_conflict_intensity")
        )
        / 3.0,
    )
    controversy = min(
        1.0,
        parse_float(
            row.get("features_post_catalyst_scores_controversiality")
        )
        / 5.0,
    )
    upvote_ratio = parse_float(row.get("final_upvote_ratio"), 1.0)
    vote_balance = max(0.0, 1.0 - abs(upvote_ratio - 0.5) / 0.5)
    ecology = str(
        row.get("features_discourse_ecology_ecology_type", "")
    ).casefold()
    ecology_signal = float(
        any(token in ecology for token in ("conflict", "toxic", "hostile"))
    )
    return round(
        0.42 * conflict
        + 0.28 * controversy
        + 0.20 * vote_balance
        + 0.10 * ecology_signal,
        6,
    )


def comment_priority(row: dict[str, str], seed: int) -> float:
    score = max(0, parse_int(row.get("score")))
    depth = max(0, parse_int(row.get("depth")))
    jitter = (stable_int(f"{seed}:{row.get('comment_id')}") % 1000) / 1000
    return math.log1p(score) + min(depth, 5) * 0.28 + jitter * 0.12


def select_comments(
    candidates: list[dict[str, Any]],
    count: int,
) -> list[dict[str, Any]]:
    ordered = sorted(
        candidates,
        key=lambda item: (item["priority"], item["score"]),
        reverse=True,
    )
    groups = {
        "shallow": [item for item in ordered if item["raw_depth"] == 0],
        "middle": [item for item in ordered if 1 <= item["raw_depth"] <= 2],
        "deep": [item for item in ordered if item["raw_depth"] >= 3],
    }
    quotas = {
        "shallow": max(4, count // 2),
        "middle": max(3, count // 3),
        "deep": max(1, count - max(4, count // 2) - max(3, count // 3)),
    }
    chosen: list[dict[str, Any]] = []
    seen: set[str] = set()
    for group in ("shallow", "middle", "deep"):
        for item in groups[group][: quotas[group]]:
            chosen.append(item)
            seen.add(item["comment_id"])
    for item in ordered:
        if len(chosen) >= count:
            break
        if item["comment_id"] not in seen:
            chosen.append(item)
            seen.add(item["comment_id"])
    return sorted(chosen[:count], key=lambda item: item["created_utc"])


def context_comment(
    item: dict[str, Any],
    *,
    context: str,
    position: int,
) -> dict[str, Any]:
    magnitude = max(8, min(240, round(12 + 17 * math.log1p(item["score"]))))
    token = stable_int(f"{item['comment_id']}:{context}")
    if context == "neutral":
        upvotes = magnitude + 18
        downvotes = 1 + token % max(2, round(upvotes * 0.10))
    else:
        total = 2 * (magnitude + 22)
        ratio = 0.44 + (token % 13) / 100.0
        upvotes = max(1, round(total * ratio))
        downvotes = max(1, total - upvotes)
    total_votes = upvotes + downvotes
    return {
        "comment_id": item["comment_id"],
        "author": f"匿名用户 {position + 1:02d}",
        "text": item["text"],
        "score": upvotes - downvotes,
        "upvotes": upvotes,
        "downvotes": downvotes,
        "agreement_ratio": round(upvotes / total_votes, 4),
        "created_utc": item["created_utc"],
        "depth": max(1, min(6, item["raw_depth"] + 1)),
    }


def build_thread(
    row: dict[str, str],
    comments: list[dict[str, Any]],
    *,
    community: str,
) -> dict[str, Any]:
    source_id = str(row["post_id"])
    source_hash = hashlib.sha256(source_id.encode("utf-8")).hexdigest()
    thread_id = f"src-{community.lower()}-{source_hash[:12]}"
    selected = select_comments(comments, 10)
    prompt = (
        "以下是经过匿名化的历史讨论内容。请按平时浏览评论区的方式阅读，"
        "并自主决定是否回复。"
    )
    variants = {}
    for context in ("neutral", "conflict"):
        variants[context] = {
            "prompt": prompt,
            "comments": [
                context_comment(item, context=context, position=index)
                for index, item in enumerate(selected)
            ],
        }
    return {
        "thread_id": thread_id,
        "community": community,
        "title": normalize_text(row.get("title", "")),
        "source": {
            "post_id_sha256": source_hash,
            "language": "en",
            "conflict_score": post_conflict_score(row),
            "conflict_band": "pending",
            "screening_status": "automated_demo_screen",
            "approved_for_production": False,
        },
        "variants": variants,
        "control_notice": {
            "author": "社区提示",
            "text": "请围绕帖子内容和论据交流。",
            "type": "none",
        },
        "corrections": [
            {
                "author": "社区成员提醒",
                "text": (
                    "强烈分歧不代表对方缺乏善意；回复时可以直接回应论据，"
                    "并补充可核查的信息。"
                ),
                "type": "counterspeech",
            },
            {
                "author": "版主说明",
                "text": (
                    "一条针对个人的攻击性内容已被移除。请继续围绕主题和"
                    "证据展开讨论。"
                ),
                "type": "moderator_explanation",
            },
        ],
    }


def assign_conflict_bands(threads: list[dict[str, Any]]) -> None:
    by_community: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for thread in threads:
        by_community[str(thread["community"])].append(thread)
    for community_threads in by_community.values():
        ordered = sorted(
            community_threads,
            key=lambda item: (
                float(item["source"]["conflict_score"]),
                str(item["thread_id"]),
            ),
        )
        size = len(ordered)
        for index, thread in enumerate(ordered):
            fraction = (index + 0.5) / size
            band = "low" if fraction <= 1 / 3 else (
                "medium" if fraction <= 2 / 3 else "high"
            )
            thread["source"]["conflict_band"] = band


def build_bank(comments_per_thread: int, seed: int) -> tuple[dict, list[dict]]:
    threads: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    exclusion_counts: Counter[str] = Counter()
    source_total = 0

    for community in COMMUNITIES:
        posts_path, comments_path = community_paths(community)
        with posts_path.open("r", encoding="utf-8-sig", newline="") as handle:
            posts = list(csv.DictReader(handle))
        source_total += len(posts)
        post_by_id: dict[str, dict[str, str]] = {}
        for row in posts:
            title = normalize_text(row.get("title", ""))
            reason = risk_reason(title)
            if reason:
                exclusion_counts[f"post_{reason}"] += 1
                continue
            if not 24 <= len(title) <= 260:
                exclusion_counts["post_title_length"] += 1
                continue
            post_by_id[str(row["post_id"])] = row

        heaps: dict[str, list[tuple[float, int, dict[str, Any]]]] = {
            post_id: [] for post_id in post_by_id
        }
        serial = 0
        with comments_path.open(
            "r", encoding="utf-8-sig", newline="", errors="replace"
        ) as handle:
            for row in csv.DictReader(handle):
                post_id = str(row.get("post_id", ""))
                if post_id not in heaps:
                    continue
                text = normalize_text(row.get("comment_text", ""))
                reason = risk_reason(text)
                if reason or not 35 <= len(text) <= 500:
                    continue
                score = parse_int(row.get("score"))
                if score < 0:
                    continue
                serial += 1
                item = {
                    "comment_id": "c_" + hashlib.sha256(
                        str(row.get("comment_id", serial)).encode("utf-8")
                    ).hexdigest()[:14],
                    "text": text,
                    "score": score,
                    "raw_depth": max(0, parse_int(row.get("depth"))),
                    "created_utc": created_timestamp(
                        str(row.get("created_utc", "")),
                        1_700_000_000 + serial,
                    ),
                    "priority": comment_priority(row, seed),
                }
                heap = heaps[post_id]
                entry = (float(item["priority"]), serial, item)
                if len(heap) < 36:
                    heapq.heappush(heap, entry)
                elif entry[:2] > heap[0][:2]:
                    heapq.heapreplace(heap, entry)

        for post_id, row in post_by_id.items():
            candidates = [entry[2] for entry in heaps[post_id]]
            if len(candidates) < comments_per_thread:
                exclusion_counts["insufficient_safe_comments"] += 1
                continue
            thread = build_thread(row, candidates, community=community)
            threads.append(thread)
            source_rows.append(
                {
                    "thread_id": thread["thread_id"],
                    "community": community,
                    "source_post_id": post_id,
                    "source_post_id_sha256": thread["source"]["post_id_sha256"],
                    "title": thread["title"],
                    "conflict_score": thread["source"]["conflict_score"],
                    "automated_screen": "pass",
                    "manual_review": "pending",
                }
            )

    assign_conflict_bands(threads)
    band_by_thread = {
        thread["thread_id"]: thread["source"]["conflict_band"]
        for thread in threads
    }
    for row in source_rows:
        row["conflict_band"] = band_by_thread[row["thread_id"]]

    bank = {
        "schema_version": 3,
        "source_total_posts": source_total,
        "eligible_thread_count": len(threads),
        "participant_trial_count": 20,
        "sampling_design": {
            "communities": list(COMMUNITIES),
            "per_community": {"low": 1, "medium": 2, "high": 1},
            "total": {"low": 5, "medium": 10, "high": 5},
        },
        "screening": {
            "source_records_are_unchanged": True,
            "participant_authors_are_anonymized": True,
            "production_requires_manual_review": True,
            "excluded_counts": dict(sorted(exclusion_counts.items())),
        },
        "threads": sorted(
            threads,
            key=lambda item: (
                COMMUNITIES.index(str(item["community"])),
                str(item["source"]["conflict_band"]),
                str(item["thread_id"]),
            ),
        ),
    }
    return bank, source_rows


def main() -> None:
    args = parse_args()
    bank, source_rows = build_bank(args.comments_per_thread, args.seed)
    if DEFAULT_TRANSLATION_CACHE.is_file():
        from translate_human_stimuli import apply_translations, load_cache

        translations = load_cache(DEFAULT_TRANSLATION_CACHE)
        bank = apply_translations(bank, translations, "qwen3.7-plus")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(bank, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.audit_dir.mkdir(parents=True, exist_ok=True)
    source_path = args.audit_dir / "stimulus_source_manifest.csv"
    with source_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(source_rows[0]))
        writer.writeheader()
        writer.writerows(source_rows)
    counts = Counter(
        (thread["community"], thread["source"]["conflict_band"])
        for thread in bank["threads"]
    )
    manifest = {
        "status": "complete",
        "source_total_posts": bank["source_total_posts"],
        "eligible_thread_count": bank["eligible_thread_count"],
        "bank_path": str(args.output),
        "bank_sha256": sha256_file(args.output),
        "source_manifest": str(source_path),
        "source_manifest_sha256": sha256_file(source_path),
        "community_band_counts": {
            f"{community}:{band}": count
            for (community, band), count in sorted(counts.items())
        },
        "screening": bank["screening"],
    }
    manifest_path = args.audit_dir / "stimulus_bank_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
