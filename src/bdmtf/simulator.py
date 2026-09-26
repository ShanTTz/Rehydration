from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from .features import estimate_toxicity
from .intent_pool import FrozenIntentPool
from .semantic_intent import render_frozen_intent
from .schema import AgentProfile, CommentNode, Intervention, RankingPolicy, SimulationConfig, ThreadState


@dataclass
class StepTrace:
    step: int
    active_agents: int
    replies: int
    likes: int
    dislikes: int
    external_reactions: int
    intent_requests: int
    pool_exhausted: int
    supportive_intent_requests: int
    antagonistic_intent_requests: int
    supportive_pool_exhausted: int
    antagonistic_pool_exhausted: int


class BDMTFSimulator:
    """Mechanism-first simulator matching the BDMTF paper design."""

    def __init__(
        self,
        agents: Iterable[AgentProfile],
        intent_pool: FrozenIntentPool,
        config: Optional[SimulationConfig] = None,
        seed: int = 0,
        post_core_agent_transform: Optional[
            Callable[[List[AgentProfile], Intervention], List[AgentProfile]]
        ] = None,
    ) -> None:
        self.base_agents = list(agents)
        self.intent_pool = intent_pool
        self.config = config or SimulationConfig()
        self.rng = random.Random(seed)
        self.depth_rng = random.Random(seed ^ 0x42444D54)
        self.post_core_agent_transform = post_core_agent_transform
        self.intent_pool_diagnostics: Dict[str, object] = {}

    def run(
        self,
        post_id: str,
        title: str,
        intervention: Intervention,
        initial_text: str = "",
    ) -> Tuple[ThreadState, List[StepTrace]]:
        agents = [agent.with_core_shift(intervention.core) for agent in self.base_agents]
        if self.post_core_agent_transform is not None:
            agents = self.post_core_agent_transform(agents, intervention)
        state = ThreadState(post_id=post_id, title=title)
        traces: List[StepTrace] = []

        root = state.add_root_post(content=initial_text or title)
        root.likes = int(max(0.0, self.config.initial_engagement_signal))
        self._apply_context_seed(state, intervention)
        max_comment_depth = self._max_comment_depth(intervention)
        intent_session = self.intent_pool.start_session(
            capacity_multiplier=self.config.intent_pool_capacity_multiplier,
            exhaustion_policy=self.config.intent_pool_exhaustion_policy,
        )

        for step in range(self.config.steps):
            visible = self.visible_nodes(
                state,
                intervention.ranking,
                max_comment_depth=max_comment_depth,
            )
            step_counts = {
                "active": 0,
                "replies": 0,
                "likes": 0,
                "dislikes": 0,
                "external": 0,
                "intent_requests": 0,
                "pool_exhausted": 0,
                "supportive_intent_requests": 0,
                "antagonistic_intent_requests": 0,
                "supportive_pool_exhausted": 0,
                "antagonistic_pool_exhausted": 0,
            }

            refresh_interval = max(
                1,
                self.config.visibility_refresh_interval_actions,
            )
            for action_index, agent in enumerate(agents):
                if (
                    self.config.refresh_visibility_per_action
                    and action_index % refresh_interval == 0
                ):
                    visible = self.visible_nodes(
                        state,
                        intervention.ranking,
                        max_comment_depth=max_comment_depth,
                    )
                impulse, aggregate_signals = self._compute_impulse(agent, visible)
                leader_bootstrap = step == 0 and agent.is_leader
                activation_threshold = agent.threshold
                if intervention.core == "toxic":
                    activation_threshold = max(0.05, activation_threshold - self.config.toxic_activation_boost)
                if not leader_bootstrap and impulse <= activation_threshold:
                    continue
                action_probability = self.config.action_probability * (self.config.action_probability_decay**step)
                if intervention.core == "toxic":
                    action_probability *= self.config.toxic_action_multiplier
                else:
                    action_probability *= self.config.baseline_action_multiplier
                if not leader_bootstrap and self.rng.random() >= min(1.0, action_probability):
                    continue
                step_counts["active"] += 1
                target = self._select_target(
                    agent,
                    visible,
                    intervention,
                    max_comment_depth=max_comment_depth,
                )
                target_signals = self._signals_for_node(target) if target else aggregate_signals
                polarity = self._sample_polarity(agent, target_signals["controversy"])

                if self._should_reply(agent, target_signals, intervention):
                    step_counts["intent_requests"] += 1
                    step_counts[f"{polarity}_intent_requests"] += 1
                    intent = intent_session.retrieve(agent.agent_id, polarity)
                    if intent is None:
                        step_counts["pool_exhausted"] += 1
                        step_counts[f"{polarity}_pool_exhausted"] += 1
                        continue
                    parent_id = target.node_id if target and not _is_root_post(target) else _root_node_id(state)
                    toxicity = float(
                        intent.metadata.get(
                            "toxicity", estimate_toxicity(intent.content)
                        )
                    )
                    content = intent.content
                    semantic_metadata: Dict[str, object] = {}
                    if self.config.semantic_payload_mode == "frame_rendered":
                        rendered = render_frozen_intent(intent, target or root, title)
                        content = rendered.text
                        semantic_metadata = rendered.metadata()
                    elif self.config.semantic_payload_mode != "verbatim":
                        raise ValueError(
                            "semantic_payload_mode must be 'verbatim' or "
                            "'frame_rendered'"
                        )
                    state.add_comment(
                        author_id=agent.agent_id,
                        content=content,
                        parent_id=parent_id,
                        step=step,
                        toxicity=toxicity,
                        metadata={
                            "polarity": polarity,
                            "intent_id": intent.intent_id,
                            "condition": intervention.name,
                            "semantic_payload_mode": self.config.semantic_payload_mode,
                            "frozen_payload": intent.content,
                            **semantic_metadata,
                        },
                    )
                    step_counts["replies"] += 1
                elif target:
                    if polarity == "antagonistic":
                        target.dislikes += 1
                        step_counts["dislikes"] += 1
                    else:
                        target.likes += 1
                        step_counts["likes"] += 1

            if self.config.enable_external_traffic:
                step_counts["external"] = self._inject_external_traffic(
                    state,
                    step,
                    likes_this_step=step_counts["likes"],
                    replies_this_step=step_counts["replies"],
                )

            traces.append(
                StepTrace(
                    step=step,
                    active_agents=step_counts["active"],
                    replies=step_counts["replies"],
                    likes=step_counts["likes"],
                    dislikes=step_counts["dislikes"],
                    external_reactions=step_counts["external"],
                    intent_requests=step_counts["intent_requests"],
                    pool_exhausted=step_counts["pool_exhausted"],
                    supportive_intent_requests=step_counts[
                        "supportive_intent_requests"
                    ],
                    antagonistic_intent_requests=step_counts[
                        "antagonistic_intent_requests"
                    ],
                    supportive_pool_exhausted=step_counts[
                        "supportive_pool_exhausted"
                    ],
                    antagonistic_pool_exhausted=step_counts[
                        "antagonistic_pool_exhausted"
                    ],
                )
            )

        self.intent_pool_diagnostics = intent_session.diagnostics()
        return state, traces

    def visible_nodes(
        self,
        state: ThreadState,
        ranking: RankingPolicy,
        max_comment_depth: int = 0,
    ) -> List[CommentNode]:
        nodes = list(state.comments.values())
        root_nodes = [node for node in nodes if _is_root_post(node)]
        comment_nodes = [node for node in nodes if not _is_root_post(node)]
        if max_comment_depth > 0:
            comment_nodes = [
                node for node in comment_nodes if node.depth < max_comment_depth
            ]
        if not comment_nodes:
            return root_nodes
        ranked = sorted(comment_nodes, key=lambda node: self._ranking_score(node, ranking), reverse=True)
        frontier = sorted(comment_nodes, key=lambda node: (node.depth, node.created_step), reverse=True)
        recent = sorted(comment_nodes, key=lambda node: node.created_step, reverse=True)
        frontier_quota = max(1, self.config.viewport_k - 1)
        candidates = root_nodes + ranked[:1] + frontier[:frontier_quota] + recent[:1] + ranked
        # The post provides S0 signals but does not consume one of the k comment slots.
        return _unique_nodes(candidates)[: self.config.viewport_k + len(root_nodes)]

    def _max_comment_depth(self, intervention: Intervention) -> int:
        if not self.config.enable_depth_targeting:
            return 0
        if intervention.core == "toxic":
            depth = self.config.toxic_max_comment_depth
            jitter = self.config.toxic_max_comment_depth_jitter
        else:
            depth = self.config.baseline_max_comment_depth
            jitter = self.config.baseline_max_comment_depth_jitter
        if depth <= 0 or jitter <= 0:
            return depth
        return max(1, depth + self.depth_rng.randint(-jitter, jitter))

    def _apply_context_seed(self, state: ThreadState, intervention: Intervention) -> None:
        if intervention.context == "neutral":
            return
        if intervention.context == "hostile":
            for idx in range(intervention.hostile_seed_count):
                state.add_comment(
                    author_id=-(idx + 1),
                    content="This is already a mess and people are refusing to admit the obvious problem.",
                    parent_id=_root_node_id(state),
                    step=-1,
                    toxicity=0.55,
                    metadata={"context_seed": "hostile"},
                )
        elif intervention.context in {"positive", "love"}:
            for idx in range(intervention.positive_seed_count):
                state.add_comment(
                    author_id=-(idx + 1),
                    content="This is a constructive start. I hope the thread stays specific and helpful.",
                    parent_id=_root_node_id(state),
                    step=-1,
                    toxicity=0.0,
                    metadata={"context_seed": "positive"},
                )
        else:
            raise ValueError(f"Unknown context intervention: {intervention.context}")

    def _compute_impulse(
        self, agent: AgentProfile, visible: List[CommentNode]
    ) -> Tuple[float, Dict[str, float]]:
        if not visible:
            signals = {"controversy": 0.0, "consensus": 0.4, "heat": 0.0}
        else:
            node_signals = [self._signals_for_node(node) for node in visible]
            signals = {
                "controversy": max(item["controversy"] for item in node_signals),
                "consensus": max(item["consensus"] for item in node_signals),
                "heat": max(item["heat"] for item in node_signals),
            }
        noise = self.rng.uniform(-self.config.noise_width, self.config.noise_width)
        conflict_drive = (
            self.config.alpha_conflict * agent.antagonism * signals["controversy"]
            if self.config.enable_conflict_channel
            else 0.0
        )
        heat_drive = (
            self.config.beta_heat * agent.attention * signals["heat"]
            if self.config.enable_heat_channel
            else 0.0
        )
        impulse = (
            self.config.base_impulse
            + conflict_drive
            + self.config.gamma_consensus * agent.prosocial * signals["consensus"]
            + heat_drive
            + noise
        )
        return impulse, signals

    def _signals_for_node(self, node: CommentNode) -> Dict[str, float]:
        total = node.likes + node.dislikes
        if total == 0:
            controversy = 0.0
        else:
            controversy = 1.0 - abs(node.likes - node.dislikes) / (total + 1.0)
        consensus = 1.0 - controversy
        heat = min(
            1.0,
            math.log(node.likes + node.reply_count + 1.0)
            / math.log(1.0 + max(2.0, self.config.early_engagement_median)),
        )
        return {"controversy": controversy, "consensus": consensus, "heat": heat}

    def _sample_polarity(self, agent: AgentProfile, controversy: float) -> str:
        conflict_signal = controversy if self.config.enable_conflict_channel else 0.0
        p_dislike = _sigmoid(
            self.config.polarity_kappa * agent.antagonism * conflict_signal
        )
        return "antagonistic" if self.rng.random() < p_dislike else "supportive"

    def _should_reply(
        self,
        agent: AgentProfile,
        signals: Dict[str, float],
        intervention: Intervention,
    ) -> bool:
        heat_term = 0.30 * signals["heat"] if self.config.enable_heat_channel else 0.0
        conflict_term = (
            0.24 * signals["controversy"]
            if self.config.enable_conflict_channel
            else 0.0
        )
        p_reply = 0.12 + heat_term + conflict_term + 0.12 * agent.attention
        if agent.is_leader:
            p_reply += 0.08
        if intervention.core == "toxic":
            p_reply += self.config.toxic_reply_bonus
        p_reply = max(0.05, min(0.72, p_reply))
        return self.rng.random() < p_reply

    def _select_target(
        self,
        agent: AgentProfile,
        visible: List[CommentNode],
        intervention: Intervention,
        max_comment_depth: int = 0,
    ) -> Optional[CommentNode]:
        if not visible:
            return None
        if not self.config.enable_depth_targeting:
            return self.rng.choice(visible)
        ranking = intervention.ranking
        lam = (
            self.config.antagonistic_depth_lambda
            if agent.antagonism >= 0.3
            else self.config.constructive_depth_lambda
        )
        if intervention.core == "toxic":
            lam = max(lam, self.config.toxic_depth_lambda_floor)
            fatigue_scale = self.config.toxic_depth_fatigue_scale
            collapse_start = self.config.toxic_depth_collapse_start
            collapse_scale = self.config.toxic_depth_collapse_scale
            root_target_penalty = self.config.toxic_root_target_penalty
            direct_root_reply_probability = (
                self.config.toxic_direct_root_reply_probability
            )
            collapse_relative_to_max = (
                self.config.toxic_depth_collapse_relative_to_max
            )
        else:
            lam = max(lam, self.config.baseline_depth_lambda_floor)
            fatigue_scale = self.config.baseline_depth_fatigue_scale
            collapse_start = self.config.baseline_depth_collapse_start
            collapse_scale = self.config.baseline_depth_collapse_scale
            root_target_penalty = self.config.baseline_root_target_penalty
            direct_root_reply_probability = (
                self.config.baseline_direct_root_reply_probability
            )
            collapse_relative_to_max = (
                self.config.baseline_depth_collapse_relative_to_max
            )
        if collapse_relative_to_max and max_comment_depth > 0:
            collapse_start = float(max_comment_depth - 1)
        if root_target_penalty < 0:
            root_target_penalty = self.config.root_target_penalty
        root = next((node for node in visible if _is_root_post(node)), None)
        if (
            direct_root_reply_probability > 0
            and root is not None
            and len(visible) > 1
            and self.rng.random() < direct_root_reply_probability
        ):
            return root
        if max_comment_depth > 0:
            eligible = [
                node
                for node in visible
                if _is_root_post(node) or node.depth < max_comment_depth
            ]
            if eligible:
                visible = eligible
        weights = []
        for node in visible:
            signals = self._signals_for_node(node)
            surface_pressure = _ranking_surface_pressure(ranking)
            if self.config.enable_conflict_channel:
                surface_pressure += signals["controversy"]
            if self.config.enable_heat_channel:
                surface_pressure += 0.5 * signals["heat"]
            effective_lam = lam - surface_pressure
            depth = max(1.0, float(node.depth))
            depth_fatigue = math.exp(
                -max(0.0, depth - 1.0) / max(1e-6, fatigue_scale)
            )
            depth_fatigue *= math.exp(
                -max(0.0, depth - collapse_start) / max(1e-6, collapse_scale)
            )
            root_penalty = (
                root_target_penalty
                if _is_root_post(node) and len(visible) > 1
                else 1.0
            )
            weights.append(root_penalty * max(1e-9, (depth**effective_lam) * depth_fatigue))
        return self.rng.choices(visible, weights=weights, k=1)[0]

    def _inject_external_traffic(
        self,
        state: ThreadState,
        step: int,
        likes_this_step: int,
        replies_this_step: int,
    ) -> int:
        if not state.comments:
            return 0
        nodes = list(state.comments.values())
        comment_nodes = [node for node in nodes if not _is_root_post(node)]
        momentum = 1.0 + 0.25 * (likes_this_step + 3 * replies_this_step)
        conflict_boost = 1.0
        if self.config.enable_conflict_channel:
            conflict_boost += 3.5 * _safe_divide(
                replies_this_step,
                likes_this_step + replies_this_step,
            )
        survival = self._survival_gate(len(comment_nodes), replies_this_step)
        raw = (
            self.config.external_lambda
            * self._log_potency()
            * momentum
            * conflict_boost
            * survival
            * (self.config.external_decay**step)
            * self.rng.uniform(0.8, 1.2)
        )
        n = int(max(0, min(self.config.max_external_reactions_per_step, raw)))
        if n == 0:
            return 0

        root = state.comments.get("post")
        if root is None:
            return 0
        dislike_probability = _safe_divide(
            sum(node.dislikes for node in comment_nodes),
            sum(node.likes + node.dislikes for node in comment_nodes),
        )
        for _ in range(n):
            if self.rng.random() < dislike_probability:
                root.dislikes += 1
            else:
                root.likes += 1
        state.external_reactions += n
        return n

    def _log_potency(self) -> float:
        signal = max(0.0, float(self.config.initial_engagement_signal))
        if signal <= 0:
            return 1.0
        denom = math.log(1.0 + max(2.0, self.config.early_engagement_median))
        return max(0.05, math.log(signal + 1.0) / denom)

    @staticmethod
    def _survival_gate(volume: int, replies_this_step: int) -> float:
        if volume < 3 and replies_this_step == 0:
            return 0.05
        if volume < 8:
            return 0.2
        if replies_this_step >= 5:
            return 1.3
        return 1.0

    @staticmethod
    def _ranking_score(node: CommentNode, ranking: RankingPolicy) -> float:
        if ranking == RankingPolicy.BEST:
            return node.likes - node.dislikes
        if ranking == RankingPolicy.TOP:
            return node.likes
        if ranking == RankingPolicy.CONTROVERSIAL:
            return min(node.likes, node.dislikes)
        raise ValueError(f"Unsupported ranking: {ranking}")


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _safe_divide(num: float, denom: float) -> float:
    return 0.0 if denom == 0 else num / denom


def _is_root_post(node: CommentNode) -> bool:
    return bool(node.metadata.get("root_post"))


def _root_node_id(state: ThreadState) -> Optional[str]:
    return "post" if "post" in state.comments else None


def _unique_nodes(nodes: Iterable[CommentNode]) -> List[CommentNode]:
    seen = set()
    out: List[CommentNode] = []
    for node in nodes:
        if node.node_id in seen:
            continue
        seen.add(node.node_id)
        out.append(node)
    return out


def _ranking_surface_pressure(ranking: RankingPolicy) -> float:
    if ranking == RankingPolicy.BEST:
        return -1.0
    if ranking == RankingPolicy.TOP:
        return 1.2
    if ranking == RankingPolicy.CONTROVERSIAL:
        return 0.5
    return 0.0
