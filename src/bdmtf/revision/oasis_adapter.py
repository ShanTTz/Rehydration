from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from bdmtf.revision.policies import EventNode, RedditRankingPolicy


def vendored_oasis_root(repository_root: Path) -> Path:
    return repository_root / "vendor" / "oasis" / "upstream"


def capability_manifest(repository_root: Path) -> dict[str, Any]:
    root = vendored_oasis_root(repository_root)
    package = root / "oasis"
    required = {
        "actions": package / "social_agent" / "agent_action.py",
        "agent_graph": package / "social_agent" / "agent_graph.py",
        "platform": package / "social_platform" / "platform.py",
        "recommender": package / "social_platform" / "recsys.py",
        "database": package / "social_platform" / "database.py",
        "comment_schema": package / "social_platform" / "schema" / "comment.sql",
        "license": root / "LICENSE",
    }
    return {
        "root": str(root),
        "python_compatibility": ">=3.10,<3.12",
        "core_runtime_required": False,
        "components": {name: {"path": str(path.relative_to(root)), "present": path.exists()} for name, path in required.items()},
        "complete": all(path.exists() for path in required.values()),
    }


def map_event_to_manual_action(node: EventNode, post_id: int = 1) -> dict[str, Any]:
    if node.metadata.get("root_post"):
        return {"action_type": "CREATE_POST", "arguments": {"content": str(node.metadata.get("content", "post"))}}
    return {
        "action_type": "CREATE_COMMENT",
        "arguments": {"post_id": post_id, "content": str(node.metadata.get("content", node.node_id)), "parent_id": node.parent_id},
    }


def oasis_rank_compatibility(nodes: list[EventNode], mode: str, now_minute: float) -> list[str]:
    return [node.node_id for node in RedditRankingPolicy(mode).rank(nodes, now_minute)]


def import_vendored_oasis(repository_root: Path):
    root = vendored_oasis_root(repository_root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import oasis

    return oasis
