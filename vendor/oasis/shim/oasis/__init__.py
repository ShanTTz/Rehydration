from __future__ import annotations

import asyncio
import json
import sqlite3
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Tuple


class ActionType(str, Enum):
    CREATE_POST = "CREATE_POST"
    LIKE_POST = "LIKE_POST"
    DISLIKE_POST = "DISLIKE_POST"
    CREATE_COMMENT = "CREATE_COMMENT"
    LIKE_COMMENT = "LIKE_COMMENT"
    DISLIKE_COMMENT = "DISLIKE_COMMENT"
    DO_NOTHING = "DO_NOTHING"


class DefaultPlatformType(str, Enum):
    REDDIT = "REDDIT"


@dataclass(frozen=True)
class ManualAction:
    action_type: ActionType
    action_args: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LLMAction:
    action_type: ActionType
    action_args: Dict[str, Any] = field(default_factory=dict)


class SocialAgent:
    def __init__(self, social_agent_id: int, profile: Dict[str, Any]):
        self.social_agent_id = int(social_agent_id)
        self.profile = dict(profile)
        persona = self.profile.get("persona") or self.profile.get("bio") or "A social media user."
        self.system_message = SimpleNamespace(content=str(persona))

    def __hash__(self) -> int:
        return hash(self.social_agent_id)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, SocialAgent) and other.social_agent_id == self.social_agent_id


class AgentGraph:
    def __init__(self, agents: Iterable[SocialAgent]):
        self._agents = list(agents)
        self._by_id = {agent.social_agent_id: agent for agent in self._agents}

    def get_agent(self, agent_id: int) -> SocialAgent:
        return self._by_id[int(agent_id)]

    def get_agents(self) -> List[Tuple[int, SocialAgent]]:
        return [(agent.social_agent_id, agent) for agent in self._agents]


async def generate_reddit_agent_graph(
    profile_path: str | Path,
    model: Any = None,
    available_actions: Optional[List[ActionType]] = None,
) -> AgentGraph:
    del model, available_actions
    profiles = json.loads(Path(profile_path).read_text(encoding="utf-8"))
    if not isinstance(profiles, list):
        raise ValueError("OASIS profile file must contain a JSON list.")
    agents = [SocialAgent(idx, profile) for idx, profile in enumerate(profiles)]
    return AgentGraph(agents)


class SimpleRedditDatabase:
    def __init__(self, database_path: str | Path):
        self.path = Path(database_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(self.path))
        self.connection.row_factory = sqlite3.Row
        self._create_schema()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS post (
                id INTEGER PRIMARY KEY,
                author_id INTEGER,
                content TEXT,
                likes INTEGER DEFAULT 0,
                dislikes INTEGER DEFAULT 0,
                created_step INTEGER
            );
            CREATE TABLE IF NOT EXISTS comment (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                post_id INTEGER,
                parent_comment_id INTEGER,
                author_id INTEGER,
                content TEXT,
                likes INTEGER DEFAULT 0,
                dislikes INTEGER DEFAULT 0,
                created_step INTEGER
            );
            CREATE TABLE IF NOT EXISTS action_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                step INTEGER,
                agent_id INTEGER,
                action_type TEXT,
                action_args TEXT
            );
            """
        )
        self.connection.commit()

    def get_table_size(self, table: str) -> int:
        if table not in {"post", "comment", "action_log"}:
            raise ValueError(f"Unsupported table: {table}")
        row = self.connection.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()
        return int(row["n"])

    def query_comments(self, post_id: int) -> List[SimpleNamespace]:
        rows = self.connection.execute(
            "SELECT id, post_id, parent_comment_id, author_id, content, likes, dislikes, created_step "
            "FROM comment WHERE post_id = ? ORDER BY created_step, id",
            (int(post_id),),
        ).fetchall()
        return [SimpleNamespace(**dict(row)) for row in rows]

    def close(self) -> None:
        self.connection.close()


class RedditEnvironment:
    def __init__(self, agent_graph: AgentGraph, database_path: str | Path):
        self.agent_graph = agent_graph
        self.database = SimpleRedditDatabase(database_path)
        self.step_index = 0

    async def reset(self) -> None:
        await asyncio.sleep(0)
        self.step_index = 0

    async def step(self, actions: Dict[SocialAgent, ManualAction | LLMAction]) -> None:
        await asyncio.sleep(0)
        for agent, action in actions.items():
            action_type = ActionType(action.action_type)
            args = dict(action.action_args or {})
            self._apply_action(agent, action_type, args)
            self.database.connection.execute(
                "INSERT INTO action_log(step, agent_id, action_type, action_args) VALUES (?, ?, ?, ?)",
                (self.step_index, agent.social_agent_id, action_type.value, json.dumps(args, ensure_ascii=False)),
            )
        self.database.connection.commit()
        self.step_index += 1

    async def to_text_prompt(self, social_agent_id: int) -> str:
        await asyncio.sleep(0)
        posts = self.database.connection.execute(
            "SELECT id, content, likes, dislikes FROM post ORDER BY id LIMIT 5"
        ).fetchall()
        comments = self.database.connection.execute(
            "SELECT id, post_id, content, likes, dislikes FROM comment ORDER BY created_step DESC, id DESC LIMIT 10"
        ).fetchall()
        lines = [f"Agent {social_agent_id} observes a Reddit-like thread."]
        for post in posts:
            lines.append(f"Post {post['id']}: {post['content']} [likes={post['likes']}, dislikes={post['dislikes']}]")
        for comment in comments:
            lines.append(
                f"Comment {comment['id']} on post {comment['post_id']}: {comment['content']} "
                f"[likes={comment['likes']}, dislikes={comment['dislikes']}]"
            )
        return "\n".join(lines)

    async def close(self) -> None:
        await asyncio.sleep(0)
        self.database.close()

    def _apply_action(self, agent: SocialAgent, action_type: ActionType, args: Dict[str, Any]) -> None:
        if action_type == ActionType.DO_NOTHING:
            return
        if action_type == ActionType.CREATE_POST:
            post_id = int(args.get("post_id", 1))
            content = str(args.get("content", ""))
            self.database.connection.execute(
                "INSERT OR REPLACE INTO post(id, author_id, content, created_step) VALUES (?, ?, ?, ?)",
                (post_id, agent.social_agent_id, content, self.step_index),
            )
            return
        if action_type == ActionType.CREATE_COMMENT:
            post_id = int(args.get("post_id", 1))
            content = str(args.get("content", ""))
            parent = args.get("parent_comment_id")
            parent_id = int(parent) if parent is not None else None
            self.database.connection.execute(
                "INSERT INTO comment(post_id, parent_comment_id, author_id, content, created_step) "
                "VALUES (?, ?, ?, ?, ?)",
                (post_id, parent_id, agent.social_agent_id, content, self.step_index),
            )
            return
        if action_type in {ActionType.LIKE_POST, ActionType.DISLIKE_POST}:
            post_id = int(args.get("post_id", 1))
            column = "likes" if action_type == ActionType.LIKE_POST else "dislikes"
            self.database.connection.execute(f"UPDATE post SET {column} = {column} + 1 WHERE id = ?", (post_id,))
            return
        if action_type in {ActionType.LIKE_COMMENT, ActionType.DISLIKE_COMMENT}:
            comment_id = int(args.get("comment_id"))
            column = "likes" if action_type == ActionType.LIKE_COMMENT else "dislikes"
            self.database.connection.execute(f"UPDATE comment SET {column} = {column} + 1 WHERE id = ?", (comment_id,))
            return
        raise ValueError(f"Unsupported action type: {action_type}")


def make(agent_graph: AgentGraph, platform: DefaultPlatformType, database_path: str | Path) -> RedditEnvironment:
    if DefaultPlatformType(platform) != DefaultPlatformType.REDDIT:
        raise ValueError("The local OASIS shim only implements the Reddit platform subset.")
    return RedditEnvironment(agent_graph=agent_graph, database_path=database_path)


__all__ = [
    "ActionType",
    "AgentGraph",
    "DefaultPlatformType",
    "LLMAction",
    "ManualAction",
    "RedditEnvironment",
    "SocialAgent",
    "generate_reddit_agent_graph",
    "make",
]
