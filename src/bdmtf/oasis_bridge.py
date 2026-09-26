from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .schema import AgentProfile, ThreadState


def oasis_available() -> bool:
    try:
        import oasis  # noqa: F401

        return True
    except Exception:
        return False


def oasis_shim_path() -> Path:
    return Path(__file__).resolve().parents[2] / "vendor" / "oasis" / "shim"


def ensure_oasis_runtime(prefer_shim: bool = False) -> str:
    """Make either installed OASIS or the local compatibility shim importable."""
    has_oasis = importlib.util.find_spec("oasis") is not None
    has_camel = importlib.util.find_spec("camel") is not None
    if prefer_shim or not (has_oasis and has_camel):
        shim = oasis_shim_path()
        if shim.exists() and str(shim) not in sys.path:
            sys.path.insert(0, str(shim))
        return "shim"
    return "installed"


def profiles_to_oasis_json(agents: Iterable[AgentProfile]) -> List[Dict[str, Any]]:
    """Export BDMTF agents to the OASIS profile schema used by the old social code."""
    profiles: List[Dict[str, Any]] = []
    for agent in agents:
        profiles.append(
            {
                "username": f"bdmtf_agent_{agent.agent_id}",
                "realname": f"BDMTF Agent {agent.agent_id}",
                "bio": "Synthetic agent for controlled discourse simulation.",
                "persona": (
                    f"BDMTF behavioral priors: antagonism={agent.antagonism:.2f}, "
                    f"attention={agent.attention:.2f}, prosocial={agent.prosocial:.2f}."
                ),
                "age": 30,
                "gender": "unknown",
                "mbti": "unknown",
                "country": "unknown",
                "profession": "synthetic participant",
                "interested_topics": ["online discourse", "social media"],
                "dark_tetrad_scores": agent.metadata.get("source_scores", {}),
            }
        )
    return profiles


def state_to_oasis_like_trace(state: ThreadState) -> List[Dict[str, Any]]:
    """Convert a BDMTF replay state to a platform-neutral action trace."""
    rows: List[Dict[str, Any]] = []
    for node in sorted(state.comments.values(), key=lambda item: (item.created_step, item.node_id)):
        rows.append(
            {
                "action": "CREATE_COMMENT",
                "post_id": state.post_id,
                "comment_id": node.node_id,
                "parent_comment_id": node.parent_id,
                "user_id": node.author_id,
                "content": node.content,
                "created_step": node.created_step,
                "likes": node.likes,
                "dislikes": node.dislikes,
                "depth": node.depth,
                "toxicity": node.toxicity,
            }
        )
    return rows


class OasisBridgeNote:
    """Documentation object for the intended OASIS integration boundary.

    The paper-critical mechanisms live in BDMTF core rather than the default
    OASIS platform because BDMTF needs comment-level ranking, reply parent
    control, and frozen semantic replay. OASIS can still host exported profiles
    or consume the action trace after replay.
    """

    required_oasis_extensions = [
        "comment.parent_comment_id field",
        "comment-level recommendation hook",
        "manual action replay without live LLM generation",
        "trace import/export for BDMTF state",
    ]
