from __future__ import annotations

import unittest

from bdmtf.schema import CommentNode, Intent
from bdmtf.semantic_intent import frame_from_intent, render_frozen_intent


class SemanticIntentTest(unittest.TestCase):
    def test_parent_slot_changes_surface_not_frozen_frame(self) -> None:
        intent = Intent(
            "i1",
            None,
            "supportive",
            "A source would make this evidence clearer.",
        )
        first = CommentNode("c1", 1, "The vaccine trial used a small sample.", None, 1, 0)
        second = CommentNode("c2", 2, "The housing policy may increase rent.", None, 1, 0)
        rendered_first = render_frozen_intent(intent, first, "Policy")
        rendered_second = render_frozen_intent(intent, second, "Policy")
        self.assertEqual(rendered_first.frame, rendered_second.frame)
        self.assertEqual(rendered_first.frame, frame_from_intent(intent))
        self.assertNotEqual(rendered_first.text, rendered_second.text)
        self.assertIn("vaccine", rendered_first.text.lower())
        self.assertIn("housing", rendered_second.text.lower())


if __name__ == "__main__":
    unittest.main()

