from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class RankingPolicy(str, Enum):
    BEST = "best"
    TOP = "top"
    CONTROVERSIAL = "controversial"


@dataclass(frozen=True)
class SimulationConfig:
    num_agents: int = 50
    steps: int = 72
    viewport_k: int = 5
    leader_fraction: float = 0.2
    base_impulse: float = 0.20
    alpha_conflict: float = 1.8
    beta_heat: float = 1.0
    gamma_consensus: float = 0.6
    polarity_kappa: float = 5.0
    threshold_base: float = 0.52
    noise_width: float = 0.1
    action_probability: float = 0.32
    action_probability_decay: float = 0.998
    baseline_action_multiplier: float = 1.0
    toxic_action_multiplier: float = 3.0
    toxic_activation_boost: float = 0.25
    toxic_reply_bonus: float = 0.04
    early_engagement_median: float = 100.0
    initial_engagement_signal: float = 0.0
    external_lambda: float = 80.0
    external_decay: float = 0.998
    enable_external_traffic: bool = True
    enable_conflict_channel: bool = True
    enable_heat_channel: bool = True
    enable_depth_targeting: bool = True
    max_external_reactions_per_step: int = 8000
    constructive_depth_lambda: float = -1.8
    antagonistic_depth_lambda: float = 3.5
    baseline_depth_lambda_floor: float = -100.0
    toxic_depth_lambda_floor: float = -100.0
    baseline_depth_fatigue_scale: float = 6.0
    toxic_depth_fatigue_scale: float = 6.0
    baseline_depth_collapse_start: float = 13.0
    toxic_depth_collapse_start: float = 13.0
    baseline_depth_collapse_relative_to_max: bool = False
    toxic_depth_collapse_relative_to_max: bool = False
    baseline_depth_collapse_scale: float = 0.5
    toxic_depth_collapse_scale: float = 0.5
    root_target_penalty: float = 0.005
    baseline_root_target_penalty: float = -1.0
    toxic_root_target_penalty: float = -1.0
    baseline_direct_root_reply_probability: float = 0.0
    toxic_direct_root_reply_probability: float = 0.0
    refresh_visibility_per_action: bool = False
    visibility_refresh_interval_actions: int = 1
    baseline_max_comment_depth: int = 0
    toxic_max_comment_depth: int = 0
    baseline_max_comment_depth_jitter: int = 0
    toxic_max_comment_depth_jitter: int = 0
    intent_pool_capacity_multiplier: int = 1
    intent_pool_exhaustion_policy: str = "no_op"
    semantic_payload_mode: str = "verbatim"
    community_calibration: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass(frozen=True)
class Intervention:
    name: str
    core: str = "baseline"
    ranking: RankingPolicy = RankingPolicy.BEST
    context: str = "neutral"
    hostile_seed_count: int = 3
    positive_seed_count: int = 3

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "Intervention":
        return cls(
            name=str(raw.get("name", "condition")),
            core=str(raw.get("core", "baseline")),
            ranking=RankingPolicy(str(raw.get("ranking", "best")).lower()),
            context=str(raw.get("context", "neutral")),
            hostile_seed_count=int(raw.get("hostile_seed_count", 3)),
            positive_seed_count=int(raw.get("positive_seed_count", 3)),
        )


@dataclass
class AgentProfile:
    agent_id: int
    antagonism: float
    attention: float
    prosocial: float
    threshold: float
    is_leader: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dark_tetrad(
        cls,
        agent_id: int,
        scores: Dict[str, float],
        threshold_base: float = 0.52,
        is_leader: bool = False,
    ) -> "AgentProfile":
        mach = _score_to_unit(scores.get("machiavellianism", 2.0))
        narc = _score_to_unit(scores.get("narcissism", 2.0))
        psy = _score_to_unit(scores.get("psychopathy", 2.0))
        sad = _score_to_unit(scores.get("sadism", 2.0))
        antagonism = min(1.0, 0.45 * sad + 0.35 * psy + 0.20 * mach)
        attention = min(1.0, 0.70 * narc + 0.30 * mach)
        prosocial = max(0.0, 1.0 - 1.5 * antagonism)
        threshold = max(0.05, min(0.95, threshold_base - 0.15 * attention))
        return cls(
            agent_id=agent_id,
            antagonism=antagonism,
            attention=attention,
            prosocial=prosocial,
            threshold=threshold,
            is_leader=is_leader,
            metadata={"source_scores": dict(scores)},
        )

    def with_core_shift(self, core: str) -> "AgentProfile":
        if core == "baseline":
            return self
        if core == "toxic":
            antagonism = min(1.0, self.antagonism * 1.35 + 0.25)
            attention = min(1.0, self.attention * 1.10 + 0.05)
        elif core == "healthy":
            antagonism = max(0.0, self.antagonism * 0.45)
            attention = self.attention
        else:
            raise ValueError(f"Unknown core intervention: {core}")
        prosocial = max(0.0, 1.0 - 1.5 * antagonism)
        threshold = max(0.05, min(0.95, self.threshold - 0.04 * attention))
        return AgentProfile(
            agent_id=self.agent_id,
            antagonism=antagonism,
            attention=attention,
            prosocial=prosocial,
            threshold=threshold,
            is_leader=self.is_leader,
            metadata=dict(self.metadata, core_shift=core),
        )


@dataclass
class Intent:
    intent_id: str
    agent_id: Optional[int]
    polarity: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CommentNode:
    node_id: str
    author_id: int
    content: str
    parent_id: Optional[str]
    depth: int
    created_step: int
    likes: int = 0
    dislikes: int = 0
    reply_count: int = 0
    toxicity: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ThreadState:
    post_id: str
    title: str
    comments: Dict[str, CommentNode] = field(default_factory=dict)
    root_comment_ids: List[str] = field(default_factory=list)
    next_node_index: int = 0
    external_reactions: int = 0

    def add_root_post(self, content: str = "", step: int = -1) -> CommentNode:
        node_id = "post"
        if node_id in self.comments:
            return self.comments[node_id]
        node = CommentNode(
            node_id=node_id,
            author_id=-9999,
            content=content or self.title,
            parent_id=None,
            depth=0,
            created_step=step,
            toxicity=0.0,
            metadata={"root_post": True},
        )
        self.comments[node_id] = node
        return node

    def add_comment(
        self,
        author_id: int,
        content: str,
        parent_id: Optional[str],
        step: int,
        toxicity: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CommentNode:
        node_id = f"c{self.next_node_index}"
        self.next_node_index += 1
        if parent_id and parent_id in self.comments:
            depth = self.comments[parent_id].depth + 1
            self.comments[parent_id].reply_count += 1
        else:
            parent_id = None
            depth = 1
            self.root_comment_ids.append(node_id)
        node = CommentNode(
            node_id=node_id,
            author_id=author_id,
            content=content,
            parent_id=parent_id,
            depth=depth,
            created_step=step,
            toxicity=toxicity,
            metadata=metadata or {},
        )
        self.comments[node_id] = node
        return node


def _score_to_unit(score: float) -> float:
    try:
        value = float(score)
    except (TypeError, ValueError):
        value = 2.0
    # Paper Appendix A.1.2, Eq. 6: D(theta) = sigmoid((theta - mu) / mu), mu=2.5.
    return 1.0 / (1.0 + math.exp(-((value - 2.5) / 2.5)))
