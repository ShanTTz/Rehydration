import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SHIM = ROOT / "vendor" / "oasis" / "shim"
sys.path.insert(0, str(SHIM))


class OasisShimTest(unittest.TestCase):
    def test_reddit_environment_smoke(self):
        import oasis
        from oasis import ActionType, ManualAction, generate_reddit_agent_graph

        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                profile_path = tmp_path / "profiles.json"
                db_path = tmp_path / "oasis.db"
                profile_path.write_text(
                    json.dumps([
                        {"username": "a", "persona": "Author"},
                        {"username": "b", "persona": "Reply user"},
                    ]),
                    encoding="utf-8",
                )
                graph = await generate_reddit_agent_graph(profile_path)
                env = oasis.make(graph, oasis.DefaultPlatformType.REDDIT, db_path)
                await env.reset()
                await env.step({graph.get_agent(0): ManualAction(ActionType.CREATE_POST, {"content": "post"})})
                await env.step({
                    graph.get_agent(1): ManualAction(
                        ActionType.CREATE_COMMENT,
                        {"post_id": 1, "content": "comment"},
                    )
                })
                self.assertEqual(env.database.get_table_size("comment"), 1)
                text = await env.to_text_prompt(1)
                self.assertIn("post", text)
                await env.close()

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
