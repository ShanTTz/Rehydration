from __future__ import annotations

from dataclasses import MISSING, asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

import pandas as pd


def _utc(value: Any) -> datetime:
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Invalid UTC timestamp: {value!r}")
    return parsed.to_pydatetime()


def _required(value: Any, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must not be empty")
    return text


@dataclass(frozen=True)
class PlatformEvent:
    platform: str
    community_id: str
    content_id: str
    event_id: str
    parent_event_id: str
    author_id: str
    created_at: datetime
    event_type: str
    text: str = ""
    url: str = ""
    title: str = ""
    score: float = 0.0
    removed: bool = False
    locked: bool = False
    source_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "platform", _required(self.platform, "platform").lower())
        object.__setattr__(self, "community_id", _required(self.community_id, "community_id"))
        object.__setattr__(self, "content_id", _required(self.content_id, "content_id"))
        object.__setattr__(self, "event_id", _required(self.event_id, "event_id"))
        object.__setattr__(self, "created_at", _utc(self.created_at))
        object.__setattr__(self, "event_type", _required(self.event_type, "event_type").lower())

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["created_at"] = self.created_at.astimezone(timezone.utc).isoformat()
        return record


@dataclass(frozen=True)
class InterventionEvent:
    platform: str
    intervention_id: str
    unit_id: str
    community_id: str
    intervention_type: str
    occurred_at: datetime
    active: bool
    moderator_id: str = ""
    reason: str = ""
    content_id: str = ""
    event_id: str = ""
    source_id: str = ""
    excluded_reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "platform", _required(self.platform, "platform").lower())
        object.__setattr__(self, "intervention_id", _required(self.intervention_id, "intervention_id"))
        object.__setattr__(self, "unit_id", _required(self.unit_id, "unit_id"))
        object.__setattr__(self, "community_id", _required(self.community_id, "community_id"))
        object.__setattr__(
            self,
            "intervention_type",
            _required(self.intervention_type, "intervention_type").lower(),
        )
        object.__setattr__(self, "occurred_at", _utc(self.occurred_at))

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["occurred_at"] = self.occurred_at.astimezone(timezone.utc).isoformat()
        return record


@dataclass(frozen=True)
class MatchedStory:
    match_id: str
    left_platform: str
    left_content_id: str
    right_platform: str
    right_content_id: str
    match_type: str
    canonical_url: str
    title_similarity: float
    time_distance_hours: float
    created_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "match_id", _required(self.match_id, "match_id"))
        object.__setattr__(self, "left_platform", _required(self.left_platform, "left_platform").lower())
        object.__setattr__(self, "right_platform", _required(self.right_platform, "right_platform").lower())
        object.__setattr__(self, "left_content_id", _required(self.left_content_id, "left_content_id"))
        object.__setattr__(self, "right_content_id", _required(self.right_content_id, "right_content_id"))
        object.__setattr__(self, "match_type", _required(self.match_type, "match_type").lower())
        object.__setattr__(self, "created_at", _utc(self.created_at))
        if self.match_type not in {"exact_url", "semantic_event"}:
            raise ValueError("match_type must be exact_url or semantic_event")
        if not 0.0 <= float(self.title_similarity) <= 1.0:
            raise ValueError("title_similarity must be in [0, 1]")

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["created_at"] = self.created_at.astimezone(timezone.utc).isoformat()
        return record


@dataclass(frozen=True)
class OutcomePanel:
    platform: str
    unit_id: str
    community_id: str
    period: datetime
    relative_period: int
    treated: bool
    treatment_cohort: str
    outcome: str
    value: float
    intervention_id: str = ""
    source_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "platform", _required(self.platform, "platform").lower())
        object.__setattr__(self, "unit_id", _required(self.unit_id, "unit_id"))
        object.__setattr__(self, "community_id", _required(self.community_id, "community_id"))
        object.__setattr__(self, "period", _utc(self.period))
        object.__setattr__(self, "outcome", _required(self.outcome, "outcome"))

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["period"] = self.period.astimezone(timezone.utc).isoformat()
        return record


def records_frame(items: Iterable[PlatformEvent | InterventionEvent | MatchedStory | OutcomePanel]) -> pd.DataFrame:
    return pd.DataFrame([item.to_record() for item in items])


def validate_platform_events(frame: pd.DataFrame) -> dict[str, Any]:
    required = {
        "platform",
        "community_id",
        "content_id",
        "event_id",
        "parent_event_id",
        "author_id",
        "created_at",
        "event_type",
    }
    missing = sorted(required.difference(frame.columns))
    issues: list[str] = []
    if missing:
        issues.append(f"missing columns: {', '.join(missing)}")
    if not missing:
        timestamps = pd.to_datetime(frame["created_at"], utc=True, errors="coerce")
        if timestamps.isna().any():
            issues.append("created_at contains invalid timestamps")
        if frame.duplicated(["platform", "event_id"]).any():
            issues.append("event_id is not unique within platform")
    return {
        "status": "ready" if not issues else "invalid",
        "claim_allowed": not issues,
        "n_events": int(len(frame)),
        "issues": issues,
    }


def event_from_mapping(record: Mapping[str, Any]) -> PlatformEvent:
    fields = PlatformEvent.__dataclass_fields__
    values: dict[str, Any] = {}
    missing: list[str] = []
    for name, field in fields.items():
        if name in record:
            values[name] = record[name]
        elif field.default is not MISSING:
            values[name] = field.default
        elif field.default_factory is not MISSING:
            values[name] = field.default_factory()
        else:
            missing.append(name)
    if missing:
        raise ValueError(f"Missing PlatformEvent fields: {', '.join(missing)}")
    return PlatformEvent(**values)
