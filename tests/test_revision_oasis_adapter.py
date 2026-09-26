import unittest
from pathlib import Path

from bdmtf.revision.oasis_adapter import capability_manifest, map_event_to_manual_action
from bdmtf.revision.policies import EventNode


class RevisionOasisAdapterTest(unittest.TestCase):
    def test_vendored_capabilities_are_present(self):
        root = Path(__file__).resolve().parents[1]
        manifest = capability_manifest(root)
        self.assertTrue(manifest["complete"])

    def test_event_maps_to_comment_action(self):
        event = EventNode("c1", "post", 1, 1.0)
        action = map_event_to_manual_action(event)
        self.assertEqual(action["action_type"], "CREATE_COMMENT")


if __name__ == "__main__":
    unittest.main()
