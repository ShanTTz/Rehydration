from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Mapping, Sequence


class TraitMode(str, Enum):
    COUPLED = "coupled"
    INDEPENDENT = "independent"
    AGGRESSIVE_CONSTRUCTIVE = "aggressive_constructive"
    SHUFFLED = "shuffled"


@dataclass(frozen=True)
class BehavioralTraits:
    antagonism: float
    attention: float
    prosocial: float


@dataclass
class EventNode:
    node_id: str
    parent_id: str | None
    depth: int
    created_minute: float
    score: float = 0.0
    likes: int = 0
    dislikes: int = 0
    toxicity: float = 0.0
    removed: bool = False
    author_id: str = ""
    metadata: dict[str, object] = field(default_factory=dict)


class ActivationPolicy(ABC):
    @abstractmethod
    def event_count(self, step: int, history: Sequence[EventNode], rng: random.Random) -> int:
        raise NotImplementedError


class TargetPolicy(ABC):
    @abstractmethod
    def choose_parent(self, visible: Sequence[EventNode], history: Sequence[EventNode], rng: random.Random) -> EventNode | None:
        raise NotImplementedError


class RankingPolicy(ABC):
    @abstractmethod
    def rank(self, nodes: Sequence[EventNode], now_minute: float) -> list[EventNode]:
        raise NotImplementedError


class CorrectionPolicy(ABC):
    @abstractmethod
    def apply(self, node: EventNode, rng: random.Random) -> str:
        raise NotImplementedError


class ConnectivityGraph(ABC):
    @abstractmethod
    def neighbors(self, agent_id: str) -> frozenset[str]:
        raise NotImplementedError


class NullConnectivityGraph(ConnectivityGraph):
    """Explicitly represents unavailable Reddit follower data."""

    def neighbors(self, agent_id: str) -> frozenset[str]:
        return frozenset()


class AffiliationGraph(ConnectivityGraph):
    """Co-participation graph; never interpreted as a follower network."""

    def __init__(self, adjacency: Mapping[str, Iterable[str]]) -> None:
        self._adjacency = {key: frozenset(value) for key, value in adjacency.items()}

    def neighbors(self, agent_id: str) -> frozenset[str]:
        return self._adjacency.get(agent_id, frozenset())


class RedditRankingPolicy(RankingPolicy):
    MODES = {"new", "top", "hot", "best", "controversial"}

    def __init__(self, mode: str = "best", position_decay: float = 0.82) -> None:
        if mode not in self.MODES:
            raise ValueError(f"Unsupported ranking mode: {mode}")
        self.mode = mode
        self.position_decay = position_decay

    def rank(self, nodes: Sequence[EventNode], now_minute: float) -> list[EventNode]:
        return sorted(nodes, key=lambda node: self._score(node, now_minute), reverse=True)

    def _score(self, node: EventNode, now_minute: float) -> float:
        age_hours = max(0.0, now_minute - node.created_minute) / 60.0
        net = node.score + node.likes - node.dislikes
        total = max(1, node.likes + node.dislikes)
        if self.mode == "new":
            return node.created_minute
        if self.mode == "top":
            return net
        if self.mode == "hot":
            order = math.log10(max(abs(net), 1.0))
            sign = 1.0 if net > 0 else -1.0 if net < 0 else 0.0
            return sign * order - age_hours / 12.5
        if self.mode == "controversial":
            balance = 1.0 - abs(node.likes - node.dislikes) / (total + 1.0)
            return math.log1p(total) * balance - 0.02 * age_hours
        up = max(0, node.likes + max(0, int(node.score)))
        n = up + max(0, node.dislikes)
        if n == 0:
            return -0.001 * age_hours
        z = 1.281551565545
        phat = up / n
        return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n) - 0.001 * age_hours


class EmpiricalCorrectionPolicy(CorrectionPolicy):
    def __init__(self, moderation_probability: float = 0.0, counterspeech_probability: float = 0.0, hostile_dropout_probability: float = 0.0) -> None:
        self.moderation_probability = moderation_probability
        self.counterspeech_probability = counterspeech_probability
        self.hostile_dropout_probability = hostile_dropout_probability

    def apply(self, node: EventNode, rng: random.Random) -> str:
        if node.toxicity < 0.30:
            return "keep"
        draw = rng.random()
        if draw < self.moderation_probability:
            node.removed = True
            return "remove"
        draw -= self.moderation_probability
        if draw < self.counterspeech_probability:
            return "counterspeech"
        draw -= self.counterspeech_probability
        if draw < self.hostile_dropout_probability:
            return "dropout"
        return "keep"


def sample_traits(count: int, mode: TraitMode | str, seed: int) -> list[BehavioralTraits]:
    mode = TraitMode(mode)
    rng = random.Random(seed)
    antagonism = [rng.betavariate(2.0, 4.0) for _ in range(count)]
    attention = [rng.betavariate(2.2, 3.2) for _ in range(count)]
    independent_prosocial = [rng.betavariate(3.0, 2.5) for _ in range(count)]
    if mode is TraitMode.COUPLED:
        prosocial = [max(0.0, 1.0 - 1.5 * value) for value in antagonism]
    elif mode is TraitMode.AGGRESSIVE_CONSTRUCTIVE:
        prosocial = independent_prosocial
        for index in range(max(1, count // 5)):
            antagonism[index] = max(0.75, antagonism[index])
            prosocial[index] = max(0.75, prosocial[index])
    elif mode is TraitMode.SHUFFLED:
        prosocial = [max(0.0, 1.0 - 1.5 * value) for value in antagonism]
        rng.shuffle(prosocial)
    else:
        prosocial = independent_prosocial
    return [BehavioralTraits(a, t, p) for a, t, p in zip(antagonism, attention, prosocial)]
