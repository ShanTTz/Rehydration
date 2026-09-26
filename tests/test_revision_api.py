import unittest
from types import SimpleNamespace

from bdmtf.revision.api_intents import (
    _participant_context,
    _post_context,
    _validate_annotation_response,
    _validate_intent_response,
    select_models,
)


class RevisionApiTest(unittest.TestCase):
    def test_selects_three_distinct_model_families(self):
        config = {
            "preferred_models": [
                {"family": "openai", "patterns": ["gpt-4o-mini"]},
                {"family": "deepseek", "patterns": ["deepseek"]},
                {"family": "qwen", "patterns": ["qwen"]},
            ],
            "require_distinct_families": 3,
        }
        selected = select_models(["gpt-4o-mini", "deepseek-chat", "qwen-plus"], config)
        self.assertEqual({item["family"] for item in selected}, {"openai", "deepseek", "qwen"})

    def test_refuses_duplicate_family_substitution(self):
        config = {
            "preferred_models": [{"family": "openai", "patterns": ["gpt"]}],
            "require_distinct_families": 3,
        }
        with self.assertRaises(RuntimeError):
            select_models(["gpt-a", "gpt-b", "gpt-c"], config)

    def test_validates_frozen_intent_contract(self):
        _validate_intent_response(
            '{"action":"reply","polarity":"constructive",'
            '"content":"A reasoned response."}'
        )
        with self.assertRaises(ValueError):
            _validate_intent_response(
                '{"action":"like","polarity":"constructive","content":""}'
            )

    def test_validates_annotation_contract_and_order(self):
        task = {"item_ids": ["c1", "c2"]}
        _validate_annotation_response(
            '{"items":['
            '{"comment_id":"c1","toxicity":0.1,"emotion":"neutral",'
            '"topic":"science","counterspeech":false},'
            '{"comment_id":"c2","toxicity":0.8,"emotion":"anger",'
            '"topic":"politics","counterspeech":true}'
            "]}",
            task,
        )
        with self.assertRaises(ValueError):
            _validate_annotation_response(
                '{"items":['
                '{"comment_id":"c2","toxicity":0.1,"emotion":"neutral",'
                '"topic":"science","counterspeech":false}'
                "]}",
                task,
            )

    def test_post_context_falls_back_from_nan_body_to_title(self):
        post = SimpleNamespace(title="Visible title", full_text=float("nan"))
        self.assertEqual(_post_context(post), "Visible title")

    def test_participant_context_contains_persona(self):
        context = _participant_context(
            {
                "persona": "A careful reader who asks for evidence.",
                "bio": "Researcher",
                "interested_topics": ["science", "methods"],
            },
            3,
        )
        self.assertIn("Participant index: 3", context)
        self.assertIn("careful reader", context)
        self.assertIn("science, methods", context)


if __name__ == "__main__":
    unittest.main()
