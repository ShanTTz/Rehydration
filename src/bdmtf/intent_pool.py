from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional

import pandas as pd

from .features import estimate_toxicity
from .schema import Intent


DEFAULT_SUPPORTIVE = [
    "I agree with this and think it adds something useful to the discussion.",
    "This is a fair point. I would like to see more context before judging it.",
    "Thanks for sharing this. The thread is more useful when people stay specific.",
]

DEFAULT_ANTAGONISTIC = [
    "This is exactly the kind of take that makes the thread go in circles.",
    "People keep missing the obvious problem here, and it is frustrating.",
    "That sounds convenient, but it ignores the part everyone is arguing about.",
]


class FrozenIntentPool:
    """Immutable semantic catalog used to create branch-local replay sessions."""

    def __init__(self, intents: Iterable[Intent]):
        self._by_agent_polarity: DefaultDict[tuple[Optional[int], str], List[Intent]] = defaultdict(list)
        self._fallback: DefaultDict[str, List[Intent]] = defaultdict(list)
        for intent in intents:
            self._by_agent_polarity[(intent.agent_id, intent.polarity)].append(intent)
            self._fallback[intent.polarity].append(intent)

        for polarity, templates in {
            "supportive": DEFAULT_SUPPORTIVE,
            "antagonistic": DEFAULT_ANTAGONISTIC,
        }.items():
            if not self._fallback[polarity]:
                for idx, text in enumerate(templates):
                    intent = Intent(
                        intent_id=f"default_{polarity}_{idx}",
                        agent_id=None,
                        polarity=polarity,
                        content=text,
                        metadata={"source": "default_template"},
                    )
                    self._fallback[polarity].append(intent)
        self._legacy_session: Optional[IntentPoolSession] = None

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "FrozenIntentPool":
        intents: List[Intent] = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                raw = json.loads(line)
                intents.append(
                    Intent(
                        intent_id=str(raw["intent_id"]),
                        agent_id=raw.get("agent_id"),
                        polarity=str(raw.get("polarity", "supportive")),
                        content=str(raw.get("content", "")),
                        metadata=dict(raw.get("metadata", {})),
                    )
                )
        return cls(intents)

    @classmethod
    def from_comments(
        cls,
        comments_df: pd.DataFrame,
        max_intents: int = 2000,
        seed: int = 0,
    ) -> "FrozenIntentPool":
        if comments_df is None or comments_df.empty or "comment_text" not in comments_df.columns:
            return cls([])
        rng = random.Random(seed)
        sample = comments_df.dropna(subset=["comment_text"]).copy()
        if len(sample) > max_intents:
            sample = sample.sample(n=max_intents, random_state=seed)
        intents: List[Intent] = []
        for idx, row in enumerate(sample.itertuples(index=False)):
            text = str(getattr(row, "comment_text", "")).strip()
            if len(text) < 12:
                continue
            toxicity = estimate_toxicity(text)
            polarity = "antagonistic" if toxicity >= 0.30 else "supportive"
            digest = hashlib.sha1(f"{idx}:{text}".encode("utf-8", errors="ignore")).hexdigest()[:12]
            intents.append(
                Intent(
                    intent_id=f"comment_{digest}",
                    agent_id=None,
                    polarity=polarity,
                    content=text[:500],
                    metadata={"source": "comments_csv", "toxicity": toxicity},
                )
            )
        rng.shuffle(intents)
        return cls(intents)

    def save_jsonl(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        seen = set()
        with path.open("w", encoding="utf-8") as handle:
            for items in self._fallback.values():
                for intent in items:
                    if intent.intent_id in seen:
                        continue
                    seen.add(intent.intent_id)
                    handle.write(json.dumps(intent.__dict__, ensure_ascii=False) + "\n")

    def start_session(
        self,
        capacity_multiplier: int = 1,
        exhaustion_policy: str = "no_op",
    ) -> "IntentPoolSession":
        return IntentPoolSession(
            pool=self,
            capacity_multiplier=capacity_multiplier,
            exhaustion_policy=exhaustion_policy,
        )

    def retrieve(self, agent_id: int, polarity: str) -> Intent:
        """Backward-compatible cyclic access for legacy callers.

        New simulations must use ``start_session`` so branch state and
        exhaustion are explicit.
        """
        if self._legacy_session is None:
            self._legacy_session = self.start_session(exhaustion_policy="cycle")
        intent = self._legacy_session.retrieve(agent_id, polarity)
        if intent is None:  # pragma: no cover - cycle mode never exhausts
            raise RuntimeError("Legacy cyclic intent session unexpectedly exhausted")
        return intent

    def _resolve(
        self, agent_id: int, polarity: str
    ) -> tuple[tuple[Optional[int], str], List[Intent]]:
        requested_key = (agent_id, polarity)
        items = self._by_agent_polarity.get(requested_key)
        if items:
            return requested_key, items
        resolved_polarity = polarity if self._fallback.get(polarity) else "supportive"
        return (None, resolved_polarity), self._fallback[resolved_polarity]

    def unique_intents(self) -> List[Intent]:
        by_id: Dict[str, Intent] = {}
        for items in self._fallback.values():
            for intent in items:
                by_id.setdefault(intent.intent_id, intent)
        return list(by_id.values())


@dataclass
class IntentPoolSession:
    """Branch-local consumption state for an immutable frozen catalog."""

    pool: FrozenIntentPool
    capacity_multiplier: int = 1
    exhaustion_policy: str = "no_op"

    def __post_init__(self) -> None:
        if self.capacity_multiplier < 1:
            raise ValueError("capacity_multiplier must be at least 1")
        if self.exhaustion_policy not in {"no_op", "cycle"}:
            raise ValueError("exhaustion_policy must be 'no_op' or 'cycle'")
        self._cursor: Dict[tuple[Optional[int], str], int] = {}
        self._uses: DefaultDict[str, int] = defaultdict(int)
        self._requests: DefaultDict[str, int] = defaultdict(int)
        self._successes: DefaultDict[str, int] = defaultdict(int)
        self._exhausted: DefaultDict[str, int] = defaultdict(int)

    def retrieve(self, agent_id: int, polarity: str) -> Optional[Intent]:
        label = str(polarity)
        self._requests[label] += 1
        key, items = self.pool._resolve(agent_id, label)
        cursor = self._cursor.get(key, 0)

        if self.exhaustion_policy == "cycle":
            intent = items[cursor % len(items)]
            self._cursor[key] = cursor + 1
            self._successes[label] += 1
            return intent

        for offset in range(len(items)):
            index = (cursor + offset) % len(items)
            intent = items[index]
            used = self._uses[intent.intent_id]
            if used >= self.capacity_multiplier:
                continue
            self._uses[intent.intent_id] = used + 1
            self._cursor[key] = index + 1
            self._successes[label] += 1
            if self.capacity_multiplier == 1:
                return intent
            return Intent(
                intent_id=f"{intent.intent_id}#op{used + 1}",
                agent_id=intent.agent_id,
                polarity=intent.polarity,
                content=intent.content,
                metadata=dict(
                    intent.metadata,
                    base_intent_id=intent.intent_id,
                    opportunity_index=used + 1,
                ),
            )

        self._exhausted[label] += 1
        return None

    def diagnostics(self) -> Dict[str, Any]:
        intents = self.pool.unique_intents()
        labels = sorted(set(self._requests) | set(self._successes) | set(self._exhausted))
        requests = int(sum(self._requests.values()))
        exhausted = int(sum(self._exhausted.values()))
        by_polarity = {
            label: {
                "requests": int(self._requests[label]),
                "successes": int(self._successes[label]),
                "exhausted": int(self._exhausted[label]),
                "exhaustion_rate": (
                    float(self._exhausted[label] / self._requests[label])
                    if self._requests[label]
                    else 0.0
                ),
            }
            for label in labels
        }
        return {
            "exhaustion_policy": self.exhaustion_policy,
            "capacity_multiplier": int(self.capacity_multiplier),
            "unique_intents": int(len(intents)),
            "opportunity_capacity": int(len(intents) * self.capacity_multiplier),
            "requests": requests,
            "successes": int(sum(self._successes.values())),
            "exhausted": exhausted,
            "exhaustion_rate": float(exhausted / requests) if requests else 0.0,
            "unique_intents_consumed": int(len(self._uses)),
            "by_polarity": by_polarity,
        }
