from __future__ import annotations

import re
from typing import Any, Dict, Iterable

import pandas as pd


DARK_SIGNAL_LEXICONS = {
    "strategic_manipulation": {
        "manipulate": 3,
        "deceive": 3,
        "exploit": 4,
        "agenda": 2,
        "scheme": 2,
        "leverage": 2,
        "tactic": 2,
        "puppet": 3,
    },
    "self_focus": {
        "i am the best": 4,
        "superior": 3,
        "elite": 2,
        "genius": 2,
        "i deserve": 3,
        "look at me": 3,
        "alpha": 3,
        "sigma": 3,
    },
    "callous_impulsive": {
        "i don't care": 2,
        "no remorse": 4,
        "get over it": 2,
        "stop whining": 2,
        "no empathy": 4,
        "callous": 3,
        "reckless": 2,
        "do it anyway": 2,
    },
    "conflict_amplifying": {
        "troll": 2,
        "owned": 2,
        "triggered": 2,
        "cry more": 3,
        "snowflake": 2,
        "cope": 2,
        "seethe": 2,
        "rekt": 2,
        "u mad": 2,
        "skill issue": 2,
        "suffer": 4,
        "humiliate": 4,
        "pathetic": 3,
        "trash": 2,
        "kys": 5,
    },
}

POSITIVE_LEXICON = [
    "sorry",
    "apologize",
    "my bad",
    "thanks",
    "thank you",
    "appreciate",
    "grateful",
    "agree",
    "good point",
    "valid",
    "understand",
    "respect",
    "love",
    "great",
    "support",
]

_WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)


def sanitize_text(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace("\n", " ").replace("\r", " ").lower()
    return re.sub(r"\s+", " ", text).strip()


def count_phrase(text: str, phrase: str) -> int:
    if " " in phrase:
        return text.count(phrase)
    return len(re.findall(rf"\b{re.escape(phrase)}\b", text))


def text_risk_level(text: Any) -> float:
    cleaned = sanitize_text(text)
    if not cleaned:
        return 0.0
    score = 0.0
    for lexicon in DARK_SIGNAL_LEXICONS.values():
        for phrase, weight in lexicon.items():
            score += count_phrase(cleaned, phrase) * float(weight)
    return score


def estimate_toxicity(text: Any) -> float:
    cleaned = sanitize_text(text)
    if not cleaned:
        return 0.0
    risk = text_risk_level(cleaned)
    pos = sum(count_phrase(cleaned, phrase) for phrase in POSITIVE_LEXICON)
    return max(0.0, min(1.0, (risk / 8.0) - 0.04 * pos))


def discourse_priors(texts: Iterable[str]) -> Dict[str, Any]:
    all_text = sanitize_text(" ".join(str(t) for t in texts if t))
    words = _WORD_RE.findall(all_text)
    total_words = len(words)
    if total_words < 10:
        return {
            f"{name}_tool_score": 1 for name in DARK_SIGNAL_LEXICONS
        } | {"pos_factor_per1k": 0.0, "total_words": total_words}

    pos_count = sum(count_phrase(all_text, phrase) for phrase in POSITIVE_LEXICON)
    pos_factor = (pos_count / total_words) * 1000.0

    result: Dict[str, Any] = {}
    for trait, lexicon in DARK_SIGNAL_LEXICONS.items():
        raw = 0.0
        for phrase, weight in lexicon.items():
            raw += count_phrase(all_text, phrase) * float(weight)
        per_1k = (raw / total_words) * 1000.0
        adjusted = _apply_prosocial_counterweight(trait, per_1k, pos_factor)
        result[f"{trait}_tool_score"] = _map_to_1_5(adjusted)
        result[f"raw_{trait}_per1k"] = per_1k

    result["pos_factor_per1k"] = pos_factor
    result["total_words"] = total_words
    return result


def priors_from_author_frame(author_df: pd.DataFrame) -> Dict[str, Any]:
    if author_df is None or author_df.empty or "text" not in author_df.columns:
        return discourse_priors([])
    return discourse_priors(author_df["text"].dropna().astype(str).tolist())


def _apply_prosocial_counterweight(trait: str, score: float, pos_factor: float) -> float:
    p = min(pos_factor, 15.0) / 15.0
    if trait in {"callous_impulsive", "conflict_amplifying"}:
        return score * (1.0 - 0.30 * p)
    if trait == "strategic_manipulation":
        return score * (1.0 - 0.10 * p)
    return score


def _map_to_1_5(score_per_1k: float) -> int:
    if score_per_1k <= 1.0:
        return 1
    if score_per_1k < 6.0:
        return int(max(1, min(5, round(1.0 + (score_per_1k / 6.0) * 2.0))))
    capped = min(score_per_1k, 18.0)
    return int(max(1, min(5, round(3.0 + ((capped - 6.0) / 12.0) * 2.0))))
