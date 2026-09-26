from __future__ import annotations

import hashlib
import json
import math
import mimetypes
import random
import secrets
import sqlite3
import threading
import time
import webbrowser
from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlsplit


RANKINGS = ("new", "top", "hot", "best", "controversial")
CONTEXTS = ("neutral", "conflict")
CORRECTIONS = ("none", "counterspeech", "moderator_explanation")
EVENT_TYPES = {
    "thread_open",
    "item_visible",
    "scroll",
    "target_selected",
    "reply_submitted",
    "reply_skipped",
    "intent_frozen",
    "trial_complete",
    "exit",
    "return",
    "study_complete",
    "result_exported",
}
SCHEMA_VERSION = 3


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def hash_secret(value: str, study_id: str) -> str:
    return sha256_text(f"{study_id}:{value.strip().upper()}")


def assignment_for_index(index: int, seed: int = 30371) -> dict[str, str]:
    """Return a reproducible permuted-block assignment over ten cells."""
    cells = [
        (ranking, context)
        for ranking in RANKINGS
        for context in CONTEXTS
    ]
    block = index // len(cells)
    within = index % len(cells)
    shuffled = cells.copy()
    random.Random(seed + block).shuffle(shuffled)
    ranking, context = shuffled[within]
    return {
        "ranking": ranking,
        "context": context,
        "correction": "within_participant",
        "arm": f"{ranking}_{context}",
    }


def trial_plan_for_index(
    index: int,
    threads: Iterable[str | dict[str, Any]],
    seed: int = 30371,
    *,
    trial_count: int | None = None,
    survey_item_ids: Iterable[str] = (),
    sampling_design: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return a reproducible, stratified participant-specific trial plan."""
    records = list(threads)
    if len(records) < len(CORRECTIONS):
        raise ValueError("At least three thread stimuli are required")
    rich_records = all(isinstance(item, dict) for item in records)
    desired = int(
        trial_count
        if trial_count is not None
        else (
            (sampling_design or {}).get("participant_trial_count")
            or len(CORRECTIONS)
        )
    )
    desired = max(len(CORRECTIONS), min(desired, len(records)))
    rng = random.Random(seed + 100_000 + index)

    selected: list[dict[str, Any]] = []
    if rich_records and sampling_design:
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for raw in records:
            record = dict(raw)
            key = (
                str(record.get("community", "")),
                str(record.get("source", {}).get("conflict_band", "")),
            )
            grouped.setdefault(key, []).append(record)
        quotas = sampling_design.get("per_community", {})
        communities = sampling_design.get("communities") or sorted(
            {key[0] for key in grouped}
        )
        for community in communities:
            for band, count in quotas.items():
                pool = list(grouped.get((str(community), str(band)), []))
                rng.shuffle(pool)
                if len(pool) < int(count):
                    raise ValueError(
                        f"Stimulus bank lacks {community}:{band} records"
                    )
                selected.extend(pool[: int(count)])
        if len(selected) != desired:
            remaining = [item for item in records if item not in selected]
            rng.shuffle(remaining)
            selected.extend(remaining[: max(0, desired - len(selected))])
            selected = selected[:desired]
    else:
        normalized = [
            dict(item) if isinstance(item, dict) else {"thread_id": str(item)}
            for item in records
        ]
        rng.shuffle(normalized)
        selected = normalized[:desired]

    rng.shuffle(selected)
    corrections = list(CORRECTIONS) * math.ceil(desired / len(CORRECTIONS))
    corrections = corrections[:desired]
    random.Random(seed + 200_000 + index).shuffle(corrections)
    questions = [str(item) for item in survey_item_ids]
    random.Random(seed + 300_000 + index).shuffle(questions)
    if questions and len(questions) < desired:
        questions = (questions * math.ceil(desired / len(questions)))[:desired]
    return [
        {
            "trial_index": trial_index,
            "thread_id": str(selected[trial_index]["thread_id"]),
            "community": str(selected[trial_index].get("community", "")),
            "stimulus_conflict_band": str(
                selected[trial_index].get("source", {}).get(
                    "conflict_band", ""
                )
            ),
            "correction": corrections[trial_index],
            "survey_item_id": (
                questions[trial_index] if questions else ""
            ),
        }
        for trial_index in range(desired)
    ]


def create_invite_codes(
    count: int,
    *,
    prefix: str = "RCT",
) -> list[str]:
    if count < 1:
        raise ValueError("Invite count must be positive")
    return [
        f"{prefix}-{secrets.token_hex(4).upper()}"
        for _ in range(count)
    ]


def invite_records(
    codes: Iterable[str],
    *,
    study_id: str,
) -> list[dict[str, Any]]:
    return [
        {
            "code_hash": hash_secret(code, study_id),
            "assignment_index": index,
        }
        for index, code in enumerate(codes)
    ]


def load_fixture(path: Path) -> dict[str, Any]:
    fixture = json.loads(path.read_text(encoding="utf-8"))
    threads = fixture.get("threads")
    if not isinstance(threads, list) or len(threads) < 3:
        raise ValueError("The study fixture must contain at least three threads")
    ids = [str(item.get("thread_id", "")) for item in threads]
    if any(not item for item in ids) or len(ids) != len(set(ids)):
        raise ValueError("Thread IDs must be non-empty and unique")
    for thread in threads:
        variants = thread.get("variants", {})
        for context in CONTEXTS:
            comments = variants.get(context, {}).get("comments", [])
            if len(comments) < 5:
                raise ValueError(
                    f"{thread['thread_id']}:{context} needs at least five comments"
                )
        source = thread.get("source", {})
        if source and source.get("conflict_band") not in {
            "low",
            "medium",
            "high",
        }:
            raise ValueError(
                f"{thread['thread_id']} has an invalid conflict band"
            )
    return fixture


def public_protocol(config: dict[str, Any]) -> dict[str, Any]:
    consent = config.get("consent", {})
    return {
        "study_id": str(config.get("study_id", "bdmtf-thread-study")),
        "study_title": str(config.get("study_title", "在线讨论体验研究")),
        "protocol_version": str(config.get("protocol_version", "draft")),
        "mode": str(config.get("mode", "demo")),
        "ethics_approval_id": str(config.get("ethics_approval_id", "")),
        "preregistration_url": str(config.get("preregistration_url", "")),
        "consent": {
            "version": str(consent.get("version", "draft")),
            "summary": str(consent.get("summary", "")),
            "duration_minutes": int(consent.get("duration_minutes", 12)),
            "contact": str(consent.get("contact", "")),
            "data_retention": str(consent.get("data_retention", "")),
            "compensation": str(consent.get("compensation", "")),
        },
        "trial_count": int(config.get("trials_per_participant", 3)),
        "mechanism_item_count": len(config.get("mechanism_items", [])),
        "enrollment_mode": str(config.get("enrollment_mode", "invite")),
        "intent_choices": [
            {
                "id": str(item.get("id", "")),
                "label": str(item.get("label", "")),
            }
            for item in config.get("intent_choices", [])
        ],
        "response_text_collected": bool(
            config.get("collect_response_text", True)
        ),
    }


def validate_protocol(
    config: dict[str, Any],
    *,
    allow_demo: bool = False,
) -> None:
    mode = str(config.get("mode", "demo"))
    if mode == "demo" and allow_demo:
        return
    missing = [
        name
        for name in (
            "study_id",
            "protocol_version",
            "ethics_approval_id",
            "preregistration_url",
            "frozen_primary_outcomes",
        )
        if not config.get(name)
    ]
    consent = config.get("consent", {})
    missing.extend(
        f"consent.{name}"
        for name in ("version", "summary", "contact", "data_retention")
        if not consent.get(name)
    )
    if missing:
        raise PermissionError(
            "Human experiment collection is locked until these approved "
            "protocol fields are complete: "
            + ", ".join(missing)
        )


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA busy_timeout=30000")
    return connection


def initialize_database(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _connect(path) as connection:
        connection.executescript(
            """
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS invites (
                code_hash TEXT PRIMARY KEY,
                assignment_index INTEGER NOT NULL,
                used_at TEXT,
                participant_id TEXT
            );
            CREATE TABLE IF NOT EXISTS participants (
                participant_id TEXT PRIMARY KEY,
                session_token_hash TEXT NOT NULL,
                invite_code_hash TEXT NOT NULL,
                enrolled_at TEXT NOT NULL,
                consent INTEGER NOT NULL,
                consent_version TEXT NOT NULL,
                eligible INTEGER NOT NULL,
                age_band TEXT NOT NULL,
                prior_platform_use INTEGER NOT NULL,
                baseline_conflict_tolerance INTEGER NOT NULL,
                ranking TEXT NOT NULL,
                context TEXT NOT NULL,
                correction TEXT NOT NULL,
                arm TEXT NOT NULL,
                assignment_index INTEGER NOT NULL DEFAULT 0,
                enrollment_mode TEXT NOT NULL DEFAULT 'invite',
                completed INTEGER NOT NULL DEFAULT 0,
                withdrawn INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS trials (
                trial_uuid TEXT PRIMARY KEY,
                participant_id TEXT NOT NULL,
                trial_index INTEGER NOT NULL,
                thread_id TEXT NOT NULL,
                community TEXT NOT NULL DEFAULT '',
                stimulus_conflict_band TEXT NOT NULL DEFAULT '',
                correction TEXT NOT NULL,
                survey_item_id TEXT NOT NULL DEFAULT '',
                survey_construct TEXT NOT NULL DEFAULT '',
                survey_response INTEGER,
                started_at TEXT,
                completed_at TEXT,
                perceived_conflict INTEGER,
                discussion_quality INTEGER,
                return_intent INTEGER,
                UNIQUE(participant_id, trial_index),
                FOREIGN KEY(participant_id) REFERENCES participants(participant_id)
            );
            CREATE TABLE IF NOT EXISTS intents (
                intent_uuid TEXT PRIMARY KEY,
                participant_id TEXT NOT NULL,
                trial_uuid TEXT NOT NULL UNIQUE,
                recorded_at TEXT NOT NULL,
                intent_choice TEXT NOT NULL,
                intent_text TEXT,
                intent_length INTEGER NOT NULL,
                FOREIGN KEY(participant_id) REFERENCES participants(participant_id),
                FOREIGN KEY(trial_uuid) REFERENCES trials(trial_uuid)
            );
            CREATE TABLE IF NOT EXISTS events (
                event_uuid TEXT PRIMARY KEY,
                participant_id TEXT NOT NULL,
                trial_uuid TEXT,
                recorded_at TEXT NOT NULL,
                event_type TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                content_id TEXT,
                content_depth INTEGER,
                viewport_position INTEGER,
                scroll_depth REAL,
                payload_json TEXT NOT NULL,
                FOREIGN KEY(participant_id) REFERENCES participants(participant_id),
                FOREIGN KEY(trial_uuid) REFERENCES trials(trial_uuid)
            );
            CREATE TABLE IF NOT EXISTS responses (
                response_uuid TEXT PRIMARY KEY,
                participant_id TEXT NOT NULL,
                trial_uuid TEXT NOT NULL,
                recorded_at TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                target_id TEXT,
                target_depth INTEGER,
                response_text TEXT,
                response_length INTEGER NOT NULL,
                skipped INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY(participant_id) REFERENCES participants(participant_id),
                FOREIGN KEY(trial_uuid) REFERENCES trials(trial_uuid)
            );
            """
        )
        trial_columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(trials)")
        }
        additions = {
            "community": "TEXT NOT NULL DEFAULT ''",
            "stimulus_conflict_band": "TEXT NOT NULL DEFAULT ''",
            "survey_item_id": "TEXT NOT NULL DEFAULT ''",
            "survey_construct": "TEXT NOT NULL DEFAULT ''",
            "survey_response": "INTEGER",
        }
        for name, definition in additions.items():
            if name not in trial_columns:
                connection.execute(
                    f"ALTER TABLE trials ADD COLUMN {name} {definition}"
                )
        participant_columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(participants)")
        }
        participant_additions = {
            "assignment_index": "INTEGER NOT NULL DEFAULT 0",
            "enrollment_mode": "TEXT NOT NULL DEFAULT 'invite'",
        }
        for name, definition in participant_additions.items():
            if name not in participant_columns:
                connection.execute(
                    f"ALTER TABLE participants ADD COLUMN {name} {definition}"
                )
        metadata = {
            "schema_version": str(SCHEMA_VERSION),
            "study_id": str(config.get("study_id", "")),
            "protocol_version": str(config.get("protocol_version", "")),
            "mode": str(config.get("mode", "demo")),
        }
        connection.executemany(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
            metadata.items(),
        )
        records = config.get("invite_records", [])
        connection.executemany(
            """
            INSERT OR IGNORE INTO invites(code_hash, assignment_index)
            VALUES (?, ?)
            """,
            [
                (
                    str(item["code_hash"]),
                    int(item["assignment_index"]),
                )
                for item in records
            ],
        )


@dataclass(frozen=True)
class Session:
    participant_id: str
    session_token: str
    assignment: dict[str, str]


def enroll(
    database: Path,
    config: dict[str, Any],
    fixture: dict[str, Any],
    invite_code: str = "",
    *,
    consent: bool,
    eligible: bool,
    age_band: str,
    prior_platform_use: int,
    baseline_conflict_tolerance: int,
) -> Session:
    if not consent or not eligible:
        raise PermissionError(
            "Only eligible participants who provide consent may continue"
        )
    if age_band not in {"18-24", "25-34", "35-44", "45-54", "55+"}:
        raise ValueError("A valid age band is required")
    if not 0 <= int(prior_platform_use) <= 7:
        raise ValueError("Prior platform use must be between 0 and 7")
    if not 1 <= int(baseline_conflict_tolerance) <= 7:
        raise ValueError("Conflict tolerance must be between 1 and 7")

    study_id = str(config["study_id"])
    enrollment_mode = str(config.get("enrollment_mode", "invite"))
    if enrollment_mode not in {"open", "invite"}:
        raise ValueError("enrollment_mode must be open or invite")
    code_hash = (
        hash_secret(invite_code, study_id)
        if enrollment_mode == "invite"
        else ""
    )
    seed = int(config.get("seed", 30371))
    thread_records = list(fixture["threads"])
    if bool(config.get("require_manual_stimulus_approval")):
        thread_records = [
            item
            for item in thread_records
            if bool(item.get("source", {}).get("approved_for_production"))
        ]
    required_trials = int(config.get("trials_per_participant", 3))
    if len(thread_records) < required_trials:
        raise PermissionError(
            "The approved stimulus bank is too small for this protocol"
        )
    mechanism_items = {
        str(item["id"]): dict(item)
        for item in config.get("mechanism_items", [])
    }
    survey_item_ids = list(mechanism_items)
    participant_id = "p_" + secrets.token_urlsafe(9)
    session_token = secrets.token_urlsafe(24)
    token_hash = sha256_text(session_token)

    with _connect(database) as connection:
        connection.execute("BEGIN IMMEDIATE")
        if enrollment_mode == "invite":
            invite = connection.execute(
                """
                SELECT assignment_index, used_at
                FROM invites WHERE code_hash = ?
                """,
                (code_hash,),
            ).fetchone()
            if invite is None:
                raise PermissionError("The invitation code is invalid")
            if invite["used_at"]:
                raise PermissionError(
                    "This invitation code has already been used; resume from "
                    "the same browser or contact the study coordinator"
                )
            index = int(invite["assignment_index"])
        else:
            index = secrets.randbelow(2_000_000_000)
        assignment = assignment_for_index(index, seed)
        consent_version = str(
            config.get("consent", {}).get("version", "draft")
        )
        connection.execute(
            """
            INSERT INTO participants(
                participant_id, session_token_hash, invite_code_hash,
                enrolled_at, consent, consent_version, eligible, age_band,
                prior_platform_use, baseline_conflict_tolerance, ranking,
                context, correction, arm, assignment_index, enrollment_mode
            ) VALUES (?, ?, ?, ?, 1, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                participant_id,
                token_hash,
                code_hash,
                utc_now(),
                consent_version,
                age_band,
                int(prior_platform_use),
                int(baseline_conflict_tolerance),
                assignment["ranking"],
                assignment["context"],
                assignment["correction"],
                assignment["arm"],
                index,
                enrollment_mode,
            ),
        )
        for item in trial_plan_for_index(
            index,
            thread_records,
            seed,
            trial_count=required_trials,
            survey_item_ids=survey_item_ids,
            sampling_design=fixture.get("sampling_design"),
        ):
            survey_item = mechanism_items.get(item["survey_item_id"], {})
            connection.execute(
                """
                INSERT INTO trials(
                    trial_uuid, participant_id, trial_index,
                    thread_id, community, stimulus_conflict_band,
                    correction, survey_item_id, survey_construct
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    secrets.token_hex(16),
                    participant_id,
                    item["trial_index"],
                    item["thread_id"],
                    item["community"],
                    item["stimulus_conflict_band"],
                    item["correction"],
                    item["survey_item_id"],
                    str(survey_item.get("construct", "")),
                ),
            )
        if enrollment_mode == "invite":
            connection.execute(
                """
                UPDATE invites
                SET used_at = ?, participant_id = ?
                WHERE code_hash = ?
                """,
                (utc_now(), participant_id, code_hash),
            )
    return Session(participant_id, session_token, assignment)


def _participant(
    connection: sqlite3.Connection,
    participant_id: str,
    session_token: str,
) -> sqlite3.Row:
    participant = connection.execute(
        "SELECT * FROM participants WHERE participant_id = ?",
        (participant_id,),
    ).fetchone()
    if participant is None or not secrets.compare_digest(
        str(participant["session_token_hash"]),
        sha256_text(session_token),
    ):
        raise PermissionError("The local study session is not authorized")
    if int(participant["withdrawn"]):
        raise PermissionError("This participation record has been withdrawn")
    return participant


def rank_thread(
    comments: list[dict[str, Any]],
    ranking: str,
    *,
    now: float | None = None,
) -> list[dict[str, Any]]:
    if ranking not in RANKINGS:
        raise ValueError(f"Unsupported ranking: {ranking}")
    current = float(
        now if now is not None else datetime.now(timezone.utc).timestamp()
    )

    def score(comment: dict[str, Any]) -> tuple[float, str]:
        created = float(comment.get("created_utc", current))
        age_hours = max(0.0, (current - created) / 3600.0)
        up = float(comment.get("upvotes", comment.get("score", 0)))
        down = float(comment.get("downvotes", 0))
        net = up - down
        total = max(1.0, up + down)
        controversy = min(up, down) / total
        if ranking == "new":
            value = created
        elif ranking == "top":
            value = net
        elif ranking == "hot":
            value = net / ((age_hours + 2.0) ** 1.5)
        elif ranking == "best":
            value = (up + 1.0) / (total + 2.0)
        else:
            value = controversy * total**0.5
        return value, str(comment.get("comment_id", ""))

    return sorted(comments, key=score, reverse=True)


def _thread_by_id(
    fixture: dict[str, Any],
    thread_id: str,
) -> dict[str, Any]:
    for thread in fixture["threads"]:
        if str(thread["thread_id"]) == thread_id:
            return thread
    raise ValueError(f"Unknown thread fixture: {thread_id}")


def record_intent(
    database: Path,
    config: dict[str, Any],
    participant_id: str,
    session_token: str,
    item: dict[str, Any],
) -> None:
    """Freeze the participant's semantic intent before interaction cues appear."""
    intent_uuid = str(item.get("intent_uuid", "")).strip()
    trial_uuid = str(item.get("trial_uuid", "")).strip()
    choice = str(item.get("intent_choice", "")).strip()
    text = str(item.get("intent_text", "")).strip()
    allowed = {
        str(option.get("id", ""))
        for option in config.get("intent_choices", [])
        if option.get("id")
    }
    if not allowed:
        allowed = {
            "support",
            "challenge",
            "question",
            "add_information",
            "no_reply",
        }
    if not intent_uuid or not trial_uuid:
        raise ValueError("intent_uuid and trial_uuid are required")
    if choice not in allowed:
        raise ValueError("Choose one valid initial response intention")
    if len(text) > 500:
        raise ValueError("The initial thought exceeds 500 characters")

    with _connect(database) as connection:
        _participant(connection, participant_id, session_token)
        trial = connection.execute(
            """
            SELECT completed_at FROM trials
            WHERE trial_uuid = ? AND participant_id = ?
            """,
            (trial_uuid, participant_id),
        ).fetchone()
        if trial is None:
            raise PermissionError(
                "The intent does not belong to this participant's trial"
            )
        if trial["completed_at"]:
            raise ValueError("A completed trial cannot accept a new intent")
        existing = connection.execute(
            """
            SELECT intent_choice, intent_text FROM intents
            WHERE trial_uuid = ? AND participant_id = ?
            """,
            (trial_uuid, participant_id),
        ).fetchone()
        if existing is not None:
            if (
                str(existing["intent_choice"]) == choice
                and str(existing["intent_text"] or "") == text
            ):
                return
            raise ValueError(
                "The initial intention is already frozen for this trial"
            )
        connection.execute(
            """
            INSERT INTO intents(
                intent_uuid, participant_id, trial_uuid, recorded_at,
                intent_choice, intent_text, intent_length
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                intent_uuid,
                participant_id,
                trial_uuid,
                utc_now(),
                choice,
                text or None,
                len(text),
            ),
        )


def next_trial(
    database: Path,
    config: dict[str, Any],
    fixture: dict[str, Any],
    participant_id: str,
    session_token: str,
) -> dict[str, Any]:
    with _connect(database) as connection:
        participant = _participant(
            connection,
            participant_id,
            session_token,
        )
        trial = connection.execute(
            """
            SELECT * FROM trials
            WHERE participant_id = ? AND completed_at IS NULL
            ORDER BY trial_index
            LIMIT 1
            """,
            (participant_id,),
        ).fetchone()
        counts = connection.execute(
            """
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN completed_at IS NOT NULL THEN 1 ELSE 0 END)
                    AS completed
            FROM trials WHERE participant_id = ?
            """,
            (participant_id,),
        ).fetchone()
        total = int(counts["total"] or 0)
        completed = int(counts["completed"] or 0)
        if trial is None:
            return {
                "complete": True,
                "completed_trials": completed,
                "total_trials": total,
            }
        if not trial["started_at"]:
            connection.execute(
                "UPDATE trials SET started_at = ? WHERE trial_uuid = ?",
                (utc_now(), trial["trial_uuid"]),
            )
        source = _thread_by_id(fixture, str(trial["thread_id"]))
        variant = dict(source["variants"][participant["context"]])
        intent = connection.execute(
            """
            SELECT intent_choice, intent_text, recorded_at
            FROM intents
            WHERE trial_uuid = ? AND participant_id = ?
            """,
            (trial["trial_uuid"], participant_id),
        ).fetchone()
        common = {
            "complete": False,
            "trial_uuid": str(trial["trial_uuid"]),
            "trial_index": int(trial["trial_index"]),
            "completed_trials": completed,
            "total_trials": total,
            "community": str(source["community"]),
            "stimulus_conflict_band": str(trial["stimulus_conflict_band"]),
        }
        if intent is None:
            return {
                **common,
                "phase": "intent",
                "thread": {
                    "thread_id": str(source["thread_id"]),
                    "community": str(source["community"]),
                    "title": str(source["title"]),
                    "prompt": (
                        "请只根据帖子标题，先记录你最初想如何回应。"
                        "下一步才会显示评论、投票和平台提示。"
                    ),
                },
                "intent_choices": [
                    {
                        "id": str(option.get("id", "")),
                        "label": str(option.get("label", "")),
                    }
                    for option in config.get("intent_choices", [])
                ],
            }
        comments = rank_thread(
            [dict(item) for item in variant["comments"]],
            str(participant["ranking"]),
            now=float(config.get("ranking_reference_utc", 1767300000)),
        )
        if trial["correction"] == "none":
            notice = dict(source.get("control_notice", {}))
        else:
            notice = next(
                (
                    dict(option)
                    for option in source.get("corrections", [])
                    if option.get("type") == trial["correction"]
                ),
                {},
            )
        item_by_id = {
            str(option.get("id", "")): option
            for option in config.get("mechanism_items", [])
        }
        survey = item_by_id.get(str(trial["survey_item_id"]), {})
        return {
            **common,
            "phase": "interaction",
            "frozen_intent": {
                "choice": str(intent["intent_choice"]),
                "text": str(intent["intent_text"] or ""),
            },
            "notice": {
                "type": str(notice.get("type", trial["correction"])),
                "author": str(notice.get("author", "社区提示")),
                "text": str(notice.get("text", "请围绕帖子内容交流。")),
            },
            "survey_item": {
                "id": str(survey.get("id", "")),
                "construct": str(survey.get("construct", "")),
                "text": str(survey.get("text", "")),
                "left_anchor": str(survey.get("left_anchor", "完全不同意")),
                "right_anchor": str(survey.get("right_anchor", "完全同意")),
                "reverse_scored": bool(survey.get("reverse_scored", False)),
            },
            "thread": {
                "thread_id": str(source["thread_id"]),
                "community": str(source["community"]),
                "title": str(source["title"]),
                "prompt": str(variant.get("prompt", "")),
                "comments": comments,
            },
        }


def record_event(
    database: Path,
    participant_id: str,
    session_token: str,
    item: dict[str, Any],
) -> None:
    event_type = str(item.get("event_type", ""))
    if event_type not in EVENT_TYPES:
        raise ValueError("Unsupported event_type")
    event_uuid = str(item.get("event_uuid", ""))
    if not event_uuid:
        raise ValueError("event_uuid is required for idempotency")
    payload = item.get("payload", {})
    if len(canonical_json(payload)) > 8_000:
        raise ValueError("Event payload is too large")
    with _connect(database) as connection:
        _participant(connection, participant_id, session_token)
        connection.execute(
            """
            INSERT OR IGNORE INTO events(
                event_uuid, participant_id, trial_uuid, recorded_at,
                event_type, thread_id, content_id, content_depth,
                viewport_position, scroll_depth, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_uuid,
                participant_id,
                str(item.get("trial_uuid") or "") or None,
                utc_now(),
                event_type,
                str(item.get("thread_id", "")),
                str(item.get("content_id") or "") or None,
                item.get("content_depth"),
                item.get("viewport_position"),
                item.get("scroll_depth"),
                canonical_json(payload),
            ),
        )


def record_response(
    database: Path,
    config: dict[str, Any],
    participant_id: str,
    session_token: str,
    item: dict[str, Any],
) -> None:
    response_uuid = str(item.get("response_uuid", "")).strip()
    trial_uuid = str(item.get("trial_uuid", "")).strip()
    thread_id = str(item.get("thread_id", "")).strip()
    text = str(item.get("response_text", "")).strip()
    skipped = bool(item.get("skipped"))
    if not response_uuid:
        raise ValueError("response_uuid is required for idempotency")
    if not trial_uuid or not thread_id:
        raise ValueError("trial_uuid and thread_id are required")
    if not skipped and not text:
        raise ValueError("Enter a reply or choose not to reply")
    if not skipped and not str(item.get("target_id", "")).strip():
        raise ValueError("Select a reply target before submitting")
    if len(text) > 1_200:
        raise ValueError("The reply exceeds 1,200 characters")
    stored_text = (
        text if bool(config.get("collect_response_text", True)) else None
    )
    with _connect(database) as connection:
        _participant(connection, participant_id, session_token)
        trial = connection.execute(
            """
            SELECT thread_id, survey_item_id, completed_at FROM trials
            WHERE trial_uuid = ? AND participant_id = ?
            """,
            (trial_uuid, participant_id),
        ).fetchone()
        if trial is None or str(trial["thread_id"]) != thread_id:
            raise PermissionError(
                "The response does not belong to this participant's trial"
            )
        if trial["completed_at"]:
            raise ValueError("A completed trial cannot accept a response")
        if trial["survey_item_id"]:
            intent_exists = connection.execute(
                """
                SELECT 1 FROM intents
                WHERE trial_uuid = ? AND participant_id = ?
                """,
                (trial_uuid, participant_id),
            ).fetchone()
            if intent_exists is None:
                raise ValueError(
                    "Freeze the initial intention before viewing the discussion"
                )
        connection.execute(
            """
            INSERT OR IGNORE INTO responses(
                response_uuid, participant_id, trial_uuid, recorded_at,
                thread_id, target_id, target_depth, response_text,
                response_length, skipped
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                response_uuid,
                participant_id,
                trial_uuid,
                utc_now(),
                thread_id,
                str(item.get("target_id") or "") or None,
                item.get("target_depth"),
                stored_text,
                len(text),
                int(skipped),
            ),
        )


def complete_trial(
    database: Path,
    participant_id: str,
    session_token: str,
    item: dict[str, Any],
) -> bool:
    with _connect(database) as connection:
        _participant(connection, participant_id, session_token)
        trial = connection.execute(
            """
            SELECT survey_item_id FROM trials
            WHERE trial_uuid = ? AND participant_id = ?
            LIMIT 1
            """,
            (str(item["trial_uuid"]), participant_id),
        ).fetchone()
        if trial is None:
            raise PermissionError(
                "The trial does not belong to this participant"
            )
        response_exists = connection.execute(
            """
            SELECT 1 FROM responses
            WHERE trial_uuid = ? AND participant_id = ?
            LIMIT 1
            """,
            (str(item["trial_uuid"]), participant_id),
        ).fetchone()
        if response_exists is None:
            raise ValueError(
                "Submit a reply or explicitly skip before completing the trial"
            )
        expected_item = str(trial["survey_item_id"] or "")
        if expected_item:
            submitted_item = str(item.get("survey_item_id", ""))
            rating = int(item.get("survey_response", 0))
            if submitted_item != expected_item:
                raise ValueError("The mechanism item does not match this trial")
            if not 1 <= rating <= 7:
                raise ValueError("The mechanism rating must be between 1 and 7")
            updated = connection.execute(
                """
                UPDATE trials
                SET completed_at = ?, survey_response = ?
                WHERE trial_uuid = ? AND participant_id = ?
                """,
                (
                    utc_now(),
                    rating,
                    str(item["trial_uuid"]),
                    participant_id,
                ),
            )
        else:
            ratings = {
                "perceived_conflict": int(item.get("perceived_conflict", 0)),
                "discussion_quality": int(item.get("discussion_quality", 0)),
                "return_intent": int(item.get("return_intent", 0)),
            }
            if any(not 1 <= value <= 7 for value in ratings.values()):
                raise ValueError(
                    "Every post-task rating must be between 1 and 7"
                )
            updated = connection.execute(
                """
                UPDATE trials
                SET completed_at = ?, perceived_conflict = ?,
                    discussion_quality = ?, return_intent = ?
                WHERE trial_uuid = ? AND participant_id = ?
                """,
                (
                    utc_now(),
                    ratings["perceived_conflict"],
                    ratings["discussion_quality"],
                    ratings["return_intent"],
                    str(item["trial_uuid"]),
                    participant_id,
                ),
            )
        if updated.rowcount != 1:
            raise PermissionError(
                "The trial does not belong to this participant"
            )
        remaining = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM trials
                WHERE participant_id = ? AND completed_at IS NULL
                """,
                (participant_id,),
            ).fetchone()[0]
        )
        if remaining == 0:
            connection.execute(
                """
                UPDATE participants SET completed = 1
                WHERE participant_id = ?
                """,
                (participant_id,),
            )
    return remaining == 0


def withdraw(
    database: Path,
    participant_id: str,
    session_token: str,
    *,
    redact: bool,
) -> None:
    with _connect(database) as connection:
        _participant(connection, participant_id, session_token)
        connection.execute(
            """
            UPDATE participants SET withdrawn = 1, completed = 0
            WHERE participant_id = ?
            """,
            (participant_id,),
        )
        if redact:
            connection.execute(
                """
                UPDATE responses
                SET response_text = NULL
                WHERE participant_id = ?
                """,
                (participant_id,),
            )
            connection.execute(
                """
                UPDATE events
                SET payload_json = '{}'
                WHERE participant_id = ?
                """,
                (participant_id,),
            )


def _clean_participant(row: sqlite3.Row) -> dict[str, Any]:
    excluded = {
        "session_token_hash",
        "invite_code_hash",
    }
    return {
        key: row[key]
        for key in row.keys()
        if key not in excluded
    }


def export_participant_bundle(
    database: Path,
    config: dict[str, Any],
    participant_id: str,
    session_token: str,
) -> dict[str, Any]:
    with _connect(database) as connection:
        participant = _participant(
            connection,
            participant_id,
            session_token,
        )
        if not int(participant["completed"]):
            raise PermissionError(
                "Complete all study tasks before exporting the result"
            )
        trials = [
            dict(row)
            for row in connection.execute(
                """
                SELECT * FROM trials WHERE participant_id = ?
                ORDER BY trial_index
                """,
                (participant_id,),
            )
        ]
        events = [
            dict(row)
            for row in connection.execute(
                """
                SELECT * FROM events WHERE participant_id = ?
                ORDER BY recorded_at, event_uuid
                """,
                (participant_id,),
            )
        ]
        responses = [
            dict(row)
            for row in connection.execute(
                """
                SELECT * FROM responses WHERE participant_id = ?
                ORDER BY recorded_at, response_uuid
                """,
                (participant_id,),
            )
        ]
        intents = [
            dict(row)
            for row in connection.execute(
                """
                SELECT * FROM intents WHERE participant_id = ?
                ORDER BY recorded_at, intent_uuid
                """,
                (participant_id,),
            )
        ]
    body = {
        "schema_version": SCHEMA_VERSION,
        "study_id": str(config["study_id"]),
        "protocol_version": str(config["protocol_version"]),
        "mode": str(config.get("mode", "demo")),
        "exported_at": utc_now(),
        "participant": _clean_participant(participant),
        "design": {
            "expected_trial_count": int(
                config.get("trials_per_participant", len(trials))
            ),
            "mechanism_item_count": len(config.get("mechanism_items", [])),
            "mechanism_items": [
                {
                    "id": str(item.get("id", "")),
                    "construct": str(item.get("construct", "")),
                    "reverse_scored": bool(item.get("reverse_scored", False)),
                }
                for item in config.get("mechanism_items", [])
            ],
            "intent_frozen_before_interaction": True,
        },
        "trials": trials,
        "intents": intents,
        "events": events,
        "responses": responses,
    }
    body["payload_sha256"] = sha256_text(canonical_json(body))
    return body


def _trial_outcomes(
    participant: sqlite3.Row,
    trial: sqlite3.Row,
    events: list[sqlite3.Row],
    responses: list[sqlite3.Row],
    intent: sqlite3.Row | None,
) -> dict[str, Any]:
    scroll = [
        float(row["scroll_depth"])
        for row in events
        if row["scroll_depth"] is not None
    ]
    submitted = [row for row in responses if not int(row["skipped"])]
    targets = [
        int(row["target_depth"])
        for row in submitted
        if row["target_depth"] is not None
    ]
    return {
        "participant_id": str(participant["participant_id"]),
        "trial_uuid": str(trial["trial_uuid"]),
        "trial_index": int(trial["trial_index"]),
        "arm": str(participant["arm"]),
        "ranking": str(participant["ranking"]),
        "context": str(participant["context"]),
        "correction": str(trial["correction"]),
        "community": str(trial["community"]),
        "stimulus_conflict_band": str(trial["stimulus_conflict_band"]),
        "survey_item_id": str(trial["survey_item_id"]),
        "survey_construct": str(trial["survey_construct"]),
        "survey_response": trial["survey_response"],
        "intent_choice": str(intent["intent_choice"]) if intent else "",
        "intent_length": int(intent["intent_length"]) if intent else 0,
        "intent_activation": int(
            bool(intent) and str(intent["intent_choice"]) != "no_reply"
        ),
        "consent": bool(participant["consent"]),
        "eligible": bool(participant["eligible"]),
        "completed": bool(trial["completed_at"]),
        "age_band": str(participant["age_band"]),
        "prior_platform_use": int(participant["prior_platform_use"]),
        "baseline_conflict_tolerance": int(
            participant["baseline_conflict_tolerance"]
        ),
        "scroll_depth": max(scroll, default=0.0),
        "reply_selected": int(bool(submitted)),
        "target_depth": (
            sum(targets) / len(targets) if targets else 0.0
        ),
        "exit": int(any(row["event_type"] == "exit" for row in events)),
        "reparticipation": int(
            any(row["event_type"] == "return" for row in events)
        ),
        "response_length": sum(
            int(row["response_length"]) for row in responses
        ),
        "perceived_conflict": trial["perceived_conflict"],
        "discussion_quality": trial["discussion_quality"],
        "return_intent": trial["return_intent"],
    }


def export_analysis_tables(
    database: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    trial_records: list[dict[str, Any]] = []
    with _connect(database) as connection:
        participants = connection.execute(
            """
            SELECT * FROM participants
            WHERE withdrawn = 0
            ORDER BY participant_id
            """
        ).fetchall()
        for participant in participants:
            trials = connection.execute(
                """
                SELECT * FROM trials WHERE participant_id = ?
                ORDER BY trial_index
                """,
                (participant["participant_id"],),
            ).fetchall()
            for trial in trials:
                events = connection.execute(
                    """
                    SELECT * FROM events
                    WHERE participant_id = ? AND trial_uuid = ?
                    """,
                    (
                        participant["participant_id"],
                        trial["trial_uuid"],
                    ),
                ).fetchall()
                responses = connection.execute(
                    """
                    SELECT * FROM responses
                    WHERE participant_id = ? AND trial_uuid = ?
                    """,
                    (
                        participant["participant_id"],
                        trial["trial_uuid"],
                    ),
                ).fetchall()
                intent = connection.execute(
                    """
                    SELECT * FROM intents
                    WHERE participant_id = ? AND trial_uuid = ?
                    """,
                    (
                        participant["participant_id"],
                        trial["trial_uuid"],
                    ),
                ).fetchone()
                trial_records.append(
                    _trial_outcomes(
                        participant,
                        trial,
                        events,
                        responses,
                        intent,
                    )
                )
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in trial_records:
        grouped.setdefault(record["participant_id"], []).append(record)
    participant_records = []
    for participant_id, rows in grouped.items():
        first = rows[0]
        participant_records.append(
            {
                key: first[key]
                for key in (
                    "participant_id",
                    "arm",
                    "ranking",
                    "context",
                    "consent",
                    "eligible",
                    "age_band",
                    "prior_platform_use",
                    "baseline_conflict_tolerance",
                )
            }
            | {
                "correction": "within_participant",
                "completed": all(row["completed"] for row in rows),
                "scroll_depth": sum(
                    row["scroll_depth"] for row in rows
                )
                / len(rows),
                "reply_selected": sum(
                    row["reply_selected"] for row in rows
                )
                / len(rows),
                "target_depth": sum(
                    row["target_depth"] for row in rows
                )
                / len(rows),
                "exit": int(any(row["exit"] for row in rows)),
                "reparticipation": int(
                    any(row["reparticipation"] for row in rows)
                ),
            }
        )
    return participant_records, trial_records


class LocalStudyServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        address: tuple[str, int],
        handler: type[BaseHTTPRequestHandler],
        *,
        database: Path,
        config: dict[str, Any],
        fixture: dict[str, Any],
        app_root: Path,
    ) -> None:
        super().__init__(address, handler)
        self.database = database
        self.config = config
        self.fixture = fixture
        self.app_root = app_root


def make_handler() -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server: LocalStudyServer

        def _headers(self, content_type: str, length: int) -> None:
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(length))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; img-src 'self' data:; "
                "style-src 'self'; script-src 'self'; connect-src 'self'",
            )

        def _json(self, status: int, payload: Any) -> None:
            body = canonical_json(payload).encode("utf-8")
            self.send_response(status)
            self._headers("application/json; charset=utf-8", len(body))
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> dict[str, Any]:
            size = int(self.headers.get("Content-Length", "0"))
            if size > 64_000:
                raise ValueError("Request body is too large")
            return json.loads(self.rfile.read(size) or b"{}")

        def _session(self, item: dict[str, Any]) -> tuple[str, str]:
            return (
                str(item.get("participant_id", "")),
                str(item.get("session_token", "")),
            )

        def do_GET(self) -> None:
            parsed = urlsplit(self.path)
            if parsed.path == "/api/protocol":
                self._json(
                    HTTPStatus.OK,
                    public_protocol(self.server.config),
                )
                return
            relative = (
                "index.html"
                if parsed.path in {"", "/"}
                else parsed.path.lstrip("/")
            )
            path = (self.server.app_root / relative).resolve()
            app_root = self.server.app_root.resolve()
            if app_root not in path.parents and path != app_root:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            if not path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            body = path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self._headers(
                mimetypes.guess_type(path.name)[0]
                or "application/octet-stream",
                len(body),
            )
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            try:
                item = self._body()
                if self.path == "/api/enroll":
                    session = enroll(
                        self.server.database,
                        self.server.config,
                        self.server.fixture,
                        str(item.get("invite_code", "")),
                        consent=bool(item.get("consent")),
                        eligible=bool(item.get("eligible")),
                        age_band=str(item.get("age_band", "")),
                        prior_platform_use=int(
                            item.get("prior_platform_use", -1)
                        ),
                        baseline_conflict_tolerance=int(
                            item.get(
                                "baseline_conflict_tolerance",
                                0,
                            )
                        ),
                    )
                    self._json(
                        HTTPStatus.OK,
                        {
                            "participant_id": session.participant_id,
                            "session_token": session.session_token,
                            "trial": next_trial(
                                self.server.database,
                                self.server.config,
                                self.server.fixture,
                                session.participant_id,
                                session.session_token,
                            ),
                        },
                    )
                    return
                participant_id, session_token = self._session(item)
                if self.path == "/api/resume":
                    trial = next_trial(
                        self.server.database,
                        self.server.config,
                        self.server.fixture,
                        participant_id,
                        session_token,
                    )
                    if not trial["complete"]:
                        record_event(
                            self.server.database,
                            participant_id,
                            session_token,
                            {
                                "event_uuid": secrets.token_hex(16),
                                "event_type": "return",
                                "trial_uuid": trial["trial_uuid"],
                                "thread_id": trial["thread"]["thread_id"],
                            },
                        )
                    self._json(HTTPStatus.OK, {"trial": trial})
                    return
                if self.path == "/api/event":
                    record_event(
                        self.server.database,
                        participant_id,
                        session_token,
                        item,
                    )
                    self._json(HTTPStatus.OK, {"status": "recorded"})
                    return
                if self.path == "/api/intent":
                    record_intent(
                        self.server.database,
                        self.server.config,
                        participant_id,
                        session_token,
                        item,
                    )
                    self._json(
                        HTTPStatus.OK,
                        {
                            "status": "frozen",
                            "trial": next_trial(
                                self.server.database,
                                self.server.config,
                                self.server.fixture,
                                participant_id,
                                session_token,
                            ),
                        },
                    )
                    return
                if self.path == "/api/response":
                    record_response(
                        self.server.database,
                        self.server.config,
                        participant_id,
                        session_token,
                        item,
                    )
                    self._json(HTTPStatus.OK, {"status": "recorded"})
                    return
                if self.path == "/api/trial-complete":
                    complete = complete_trial(
                        self.server.database,
                        participant_id,
                        session_token,
                        item,
                    )
                    trial = next_trial(
                        self.server.database,
                        self.server.config,
                        self.server.fixture,
                        participant_id,
                        session_token,
                    )
                    self._json(
                        HTTPStatus.OK,
                        {"study_complete": complete, "trial": trial},
                    )
                    return
                if self.path == "/api/withdraw":
                    withdraw(
                        self.server.database,
                        participant_id,
                        session_token,
                        redact=bool(
                            self.server.config.get(
                                "withdrawal_redacts_text",
                                True,
                            )
                        ),
                    )
                    self._json(HTTPStatus.OK, {"status": "withdrawn"})
                    return
                if self.path == "/api/participant-export":
                    bundle = export_participant_bundle(
                        self.server.database,
                        self.server.config,
                        participant_id,
                        session_token,
                    )
                    self._json(HTTPStatus.OK, bundle)
                    return
                if self.path == "/api/shutdown":
                    export_participant_bundle(
                        self.server.database,
                        self.server.config,
                        participant_id,
                        session_token,
                    )
                    self._json(HTTPStatus.OK, {"status": "closing"})
                    threading.Thread(
                        target=self.server.shutdown,
                        daemon=True,
                    ).start()
                    return
                self._json(
                    HTTPStatus.NOT_FOUND,
                    {"error": "Unknown endpoint"},
                )
            except (
                KeyError,
                TypeError,
                ValueError,
                PermissionError,
                json.JSONDecodeError,
            ) as exc:
                self._json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": str(exc)},
                )

        def log_message(self, format: str, *args: Any) -> None:
            return

    return Handler


def serve_human_experiment(
    root: Path,
    config: dict[str, Any],
    host: str = "127.0.0.1",
    port: int = 8765,
    database: Path | None = None,
    *,
    app_root: Path | None = None,
    open_browser: bool = False,
) -> None:
    if host not in {"127.0.0.1", "localhost"}:
        raise PermissionError(
            "The offline study server may only bind to localhost"
        )
    validate_protocol(config, allow_demo=True)
    app_root = app_root or root / "experiment_app"
    fixture = load_fixture(app_root / "thread_fixture.json")
    database = database or root / "data" / "private" / "human_rct.sqlite"
    initialize_database(database, config)
    server = LocalStudyServer(
        (host, port),
        make_handler(),
        database=database,
        config=config,
        fixture=fixture,
        app_root=app_root,
    )
    actual_port = int(server.server_address[1])
    if open_browser:
        threading.Thread(
            target=lambda: (
                time.sleep(0.7),
                webbrowser.open(f"http://127.0.0.1:{actual_port}/"),
            ),
            daemon=True,
        ).start()
    server.serve_forever()
