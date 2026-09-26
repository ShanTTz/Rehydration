from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SHIM = ROOT / "vendor" / "oasis" / "shim"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def ensure_oasis_path(prefer_shim: bool = False) -> str:
    has_oasis = importlib.util.find_spec("oasis") is not None
    has_camel = importlib.util.find_spec("camel") is not None
    if prefer_shim or not (has_oasis and has_camel):
        if str(SHIM) not in sys.path:
            sys.path.insert(0, str(SHIM))
        return "shim"
    return "installed"


async def run_smoke(prefer_shim: bool = False) -> dict:
    runtime = ensure_oasis_path(prefer_shim=prefer_shim)
    import oasis
    from oasis import ActionType, ManualAction, generate_reddit_agent_graph

    profiles = [
        {
            "username": "seed",
            "persona": "A seed author.",
            "dark_tetrad_scores": {"machiavellianism": 1, "narcissism": 1, "psychopathy": 1, "sadism": 1},
        },
        {
            "username": "reply",
            "persona": "A concise replier.",
            "dark_tetrad_scores": {"machiavellianism": 2, "narcissism": 2, "psychopathy": 2, "sadism": 2},
        },
    ]
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        profile_path = tmp_path / "profiles.json"
        db_path = tmp_path / "shim_oasis.db"
        profile_path.write_text(json.dumps(profiles), encoding="utf-8")
        graph = await generate_reddit_agent_graph(profile_path=profile_path, model=None, available_actions=[])
        env = oasis.make(agent_graph=graph, platform=oasis.DefaultPlatformType.REDDIT, database_path=db_path)
        await env.reset()
        seed_agent = graph.get_agent(0)
        reply_agent = graph.get_agent(1)
        await env.step({seed_agent: ManualAction(ActionType.CREATE_POST, {"post_id": 1, "content": "Hello OASIS"})})
        await env.step({reply_agent: ManualAction(ActionType.CREATE_COMMENT, {"post_id": 1, "content": "A reply"})})
        prompt = await env.to_text_prompt(reply_agent.social_agent_id)
        comments = env.database.get_table_size("comment")
        actions = env.database.get_table_size("action_log")
        await env.close()
    return {
        "runtime": runtime,
        "comments": comments,
        "actions": actions,
        "prompt_contains_post": "Hello OASIS" in prompt,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Check installed OASIS or local OASIS shim integration.")
    parser.add_argument("--prefer-shim", action="store_true", help="Force the local vendor/oasis/shim runtime.")
    args = parser.parse_args()
    result = asyncio.run(run_smoke(prefer_shim=args.prefer_shim))
    print(json.dumps(result, indent=2))
    ok = result["comments"] == 1 and result["actions"] == 2 and result["prompt_contains_post"]
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
