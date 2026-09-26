from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from .schema import CommentNode, Intent


_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9'-]{2,}")
_STOPWORDS = {
    "about",
    "after",
    "again",
    "also",
    "because",
    "being",
    "claim",
    "comment",
    "conclusion",
    "could",
    "disagree",
    "does",
    "drawn",
    "from",
    "have",
    "honestly",
    "into",
    "just",
    "like",
    "more",
    "people",
    "point",
    "really",
    "should",
    "still",
    "that",
    "their",
    "there",
    "they",
    "think",
    "this",
    "unsupported",
    "very",
    "what",
    "when",
    "where",
    "which",
    "with",
    "would",
    "your",
}


@dataclass(frozen=True)
class SemanticFrame:
    dialogue_act: str
    stance: str
    tone: str
    argument_basis: str
    target_type: str
    source_topic_terms: tuple[str, ...]


@dataclass(frozen=True)
class RenderedIntent:
    text: str
    frame: SemanticFrame
    parent_topic_terms: tuple[str, ...]
    source_topic_overlap: float

    def metadata(self) -> dict[str, Any]:
        return {
            "semantic_frame": asdict(self.frame),
            "parent_topic_terms": list(self.parent_topic_terms),
            "source_topic_overlap": self.source_topic_overlap,
            "renderer": "deterministic_frame_v1",
        }


def topic_terms(text: str, limit: int = 4) -> tuple[str, ...]:
    terms: list[str] = []
    seen: set[str] = set()
    for token in _TOKEN.findall(text.lower()):
        if token in _STOPWORDS or token in seen:
            continue
        seen.add(token)
        terms.append(token)
        if len(terms) >= limit:
            break
    return tuple(terms)


def _argument_basis(text: str) -> str:
    lowered = text.lower()
    categories = (
        ("evidence", ("evidence", "source", "data", "study", "proof")),
        ("feasibility", ("work", "practical", "implement", "feasible", "how")),
        ("causality", ("cause", "effect", "lead", "result", "because")),
        ("fairness", ("fair", "equal", "right", "accountab", "justice")),
        ("experience", ("experience", "seen", "happened", "personally")),
    )
    for label, markers in categories:
        if any(marker in lowered for marker in markers):
            return label
    return "clarification"


def frame_from_intent(intent: Intent) -> SemanticFrame:
    stored = intent.metadata.get("semantic_frame")
    if isinstance(stored, Mapping):
        return SemanticFrame(
            dialogue_act=str(stored.get("dialogue_act", "respond")),
            stance=str(stored.get("stance", intent.polarity)),
            tone=str(stored.get("tone", "constructive")),
            argument_basis=str(stored.get("argument_basis", "clarification")),
            target_type=str(stored.get("target_type", "reply_or_root")),
            source_topic_terms=tuple(
                str(value) for value in stored.get("source_topic_terms", ())
            ),
        )
    antagonistic = intent.polarity == "antagonistic"
    return SemanticFrame(
        dialogue_act="challenge" if antagonistic else "acknowledge_and_extend",
        stance="oppose" if antagonistic else "support_or_qualify",
        tone="confrontational" if antagonistic else "constructive",
        argument_basis=_argument_basis(intent.content),
        target_type="reply_or_root",
        source_topic_terms=topic_terms(intent.content),
    )


def _anchor(parent: CommentNode, thread_title: str) -> tuple[str, tuple[str, ...]]:
    terms = topic_terms(parent.content)
    if not terms:
        terms = topic_terms(thread_title)
    if not terms:
        return "this point", ()
    if len(terms) == 1:
        return terms[0], terms
    return " ".join(terms[:3]), terms


def _overlap(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    if not left or not right:
        return 0.0
    return len(set(left) & set(right)) / len(set(left) | set(right))


def render_frozen_intent(
    intent: Intent,
    parent: CommentNode,
    thread_title: str,
) -> RenderedIntent:
    frame = frame_from_intent(intent)
    anchor, parent_terms = _anchor(parent, thread_title)
    supportive = {
        "evidence": [
            "The point about {anchor} is useful; a source or concrete example would make it stronger.",
            "I can see the argument about {anchor}. It would help to state the supporting evidence explicitly.",
        ],
        "feasibility": [
            "The idea about {anchor} is worth considering, especially if the practical steps are made clear.",
            "I see the point about {anchor}; a concrete implementation example would help the discussion.",
        ],
        "causality": [
            "The connection involving {anchor} is plausible, though the causal steps should be stated more clearly.",
            "I follow the point about {anchor}. Separating correlation from cause would strengthen it.",
        ],
        "fairness": [
            "The fairness concern around {anchor} deserves attention, with the same standard applied consistently.",
            "I understand the concern about {anchor}; spelling out who bears the costs would improve the argument.",
        ],
        "experience": [
            "The experience involving {anchor} adds useful context, though it may help to compare other cases.",
            "That perspective on {anchor} is helpful. A broader example could show how typical it is.",
        ],
        "clarification": [
            "I see the point about {anchor}. Could you clarify the main assumption behind it?",
            "The comment about {anchor} is worth unpacking; a more specific example would help.",
        ],
    }
    antagonistic = {
        "evidence": [
            "I disagree with the claim about {anchor}; it needs stronger evidence than what is given here.",
            "The point about {anchor} is not convincing without a source or a concrete countercheck.",
        ],
        "feasibility": [
            "The proposal about {anchor} skips the practical obstacles that would determine whether it works.",
            "I do not think the claim about {anchor} survives a realistic implementation check.",
        ],
        "causality": [
            "The argument about {anchor} treats correlation as cause and leaves out competing explanations.",
            "I disagree with the causal claim around {anchor}; the intermediate steps have not been shown.",
        ],
        "fairness": [
            "The fairness claim about {anchor} applies an uneven standard and needs to address who bears the cost.",
            "I disagree with the framing of {anchor}; it ignores an important distributional tradeoff.",
        ],
        "experience": [
            "One experience involving {anchor} is not enough to support such a broad conclusion.",
            "The example about {anchor} may be real, but it does not establish the general claim.",
        ],
        "clarification": [
            "I disagree with the point about {anchor}; the central assumption is still unsupported.",
            "The claim about {anchor} is too vague to carry the conclusion being drawn from it.",
        ],
    }
    bank = antagonistic if intent.polarity == "antagonistic" else supportive
    templates = bank.get(frame.argument_basis, bank["clarification"])
    digest = hashlib.sha256(
        f"{intent.intent_id}:{parent.node_id}".encode("utf-8")
    ).digest()
    template = templates[digest[0] % len(templates)]
    return RenderedIntent(
        text=template.format(anchor=anchor),
        frame=frame,
        parent_topic_terms=parent_terms,
        source_topic_overlap=_overlap(frame.source_topic_terms, parent_terms),
    )
